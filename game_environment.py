from bot import PlanningSeeker, Oracle
import re
import pandas as pd
import math
import random

df = pd.read_csv('countries.csv')

class GameEnvironment:
    """
    Manages the state and turn loop for a single game.

    Holds the active candidate DataFrame, runs the Seeker–Oracle exchange
    until the question budget is exhausted or only one candidate remains,
    and on every turn computes the information-theoretic optimal predicate
    alongside the Seeker's chosen predicate. Both are logged so the Seeker's
    decisions can be benchmarked.
    """

    EXCLUDED_COLUMNS = {"country"}

    def __init__(self, seeker, oracle, df, experiment_id=None, game_id=None, variant=None, attribute_moves=None):        
        self.seeker = seeker
        self.oracle = oracle

        self.seeker.game = self
        self.oracle.game = self

        self.remaining_df = df.copy()
        self.remaining_df.columns = self.remaining_df.columns.str.strip().str.lower()
        self.remaining_df = self.remaining_df.astype(str).apply(lambda col: col.str.strip())
        self.remaining_df = self.remaining_df.replace("nan", "None")

        self._current_action_space = self.action_space_summary  

        self.seeker.candidate_count = len(self.remaining_df)
        self.seeker.remaining_candidates = self.remaining_df["country"].tolist()

        self.total_countries = len(self.seeker.country_choice)

        self.question_budget = seeker.question_budget
        self.game_over = False
        self.guess = None
        self.correct = None
        self.question = None
        self.answer = None
        self.turn_history = []

        self.experiment_id = experiment_id
        self.game_id = game_id
        self.variant = variant
        self.attribute_moves = attribute_moves

        self.turn_metrics = []
        self.transcript_lines = []
        self.retry_count = 0
        self.early_exit = False
        self.early_exit_turn = None

        self.score = self._compute_score()

    def _compute_score(self) -> float:
        """Score = % of original country pool still in the candidate set."""
        return (len(self.remaining_df) / self.total_countries) * 100

    @property
    def askable_attributes(self) -> list[str]:
        """All DataFrame columns that the Seeker is allowed to ask about."""
        return [
            column for column in self.remaining_df.columns
            if column not in self.EXCLUDED_COLUMNS
        ]

    @property
    def action_space_summary(self) -> str:
        attributes = sorted([
            col for col in self.remaining_df.columns
            if col not in self.EXCLUDED_COLUMNS
        ])
        random.shuffle(attributes) 
        lines = []
        for attribute in attributes:
            values = sorted(
                self.remaining_df[attribute].dropna().unique().astype(str).tolist()
            )
            lines.append(f"- {attribute} | values: {', '.join(values)}")
        return "\n".join(lines)

    def run(self):
        """
        Main turn loop. On each turn:
           1. Compute the optimal predicate over the current candidate space.
           2. Let the Seeker choose its predicate.
           3. Ask the Oracle, parse the response, filter the candidate set.
           4. Record both the Seeker's IG and the optimal IG for that turn.
        """
        turn = 1

        self.log(f"Experiment: {self.experiment_id}")
        self.log(f"Variant: {self.variant}")
        self.log(f"Game ID: {self.game_id}")
        self.log(f"Hidden country: {self.oracle.hidden_country}")
        self.log(f"Question budget: {self.seeker.question_budget}")
        self.log("Oracle: I've chosen my country, ask your first question...\n")

        while self.seeker.questions_asked < self.question_budget:

            if len(self.remaining_df) == 1:
                break

            self._current_action_space = self.action_space_summary
            optimal = self.calculate_best_question()

            question = self.seeker.act()
            if question is None:
                break

            seeker_attribute = self.extract_attribute(question)
            seeker_value = self.extract_value(question)
            seeker_question = self.extract_question(question)

            seeker_attribute = self.extract_attribute(question)

            if seeker_attribute not in self.attribute_moves:
                self.log(f"INVALID ATTRIBUTE: {seeker_attribute}. Retrying...")
                question = self.seeker.act()

            self.log(f"\n=== TURN {turn} ===")
            self.log(f"[QUESTION]: ATTRIBUTE: {seeker_attribute} VALUE: {seeker_value}")
            self.log(f"[QUESTION TEXT]: {seeker_question}")

            self.question = question

            response = self.oracle.act()
            parsed_response = self.extract_response(response)

            self.log(f"[ORACLE]: {parsed_response}")

            self.seeker.update_history(seeker_attribute, seeker_value, parsed_response)
            self.oracle.update_history(seeker_attribute, seeker_value, parsed_response)

            seeker_ig = self._lookup_seeker_ig(
                seeker_attribute, seeker_value, optimal["all_scores"]
            )

            before = len(self.remaining_df)

            self.remaining_df = self.filter_candidates(
                self.remaining_df,
                question,
                response
            )

            after = len(self.remaining_df)

            realised_info = self.realised_information(before, after)
            ig_gap = optimal["best_ig"] - seeker_ig
            ig_ratio = seeker_ig / optimal["best_ig"] if optimal["best_ig"] > 0 else 0

            self.log(
                f"[IG] Seeker: {seeker_ig:.4f} | "
                f"Optimal: {optimal['best_ig']:.4f} | "
                f"Gap: {ig_gap:.4f} | "
                f"Ratio: {ig_ratio:.4f}"
            )

            self.log(
                f"[OPTIMAL]: ATTRIBUTE: {optimal['best_attribute']} "
                f"VALUE: {optimal['best_value']}"
            )

            self.log(f"Candidates: {before} -> {after}")

            self.seeker.candidate_count = after
            self.seeker.remaining_candidates = self.remaining_df["country"].tolist()
            self.score = self._compute_score()

            hidden = self.oracle.hidden_country.strip().lower()
            countries = self.remaining_df["country"].str.strip().str.lower().values

            if hidden not in countries:
                self.log("WARNING: Hidden country was eliminated.")
                self.log(f"Hidden country: {hidden}")
                self.log(f"Question: {question}")
                self.log(f"Answer: {response}")

            self.turn_history.append({
                "variant": self.seeker.variant_name,
                "turn": turn,
                "question": question,
                "answer": response,
                "seeker_attribute": seeker_attribute,
                "seeker_value": seeker_value,
                "seeker_ig": seeker_ig,
                "optimal_attribute": optimal["best_attribute"],
                "optimal_value": optimal["best_value"],
                "optimal_ig": optimal["best_ig"],
                "all_ig_scores": optimal["all_scores"],
                "before": before,
                "after": after,
                "score": self.score,
            })

            self.turn_metrics.append({
                "experiment_id": self.experiment_id,
                "game_id": self.game_id,
                "variant": self.variant,
                "hidden_country": self.oracle.hidden_country,
                "turn": turn,
                "seeker_attribute": seeker_attribute,
                "seeker_value": seeker_value,
                "seeker_ig": seeker_ig,
                "optimal_attribute": optimal["best_attribute"],
                "optimal_value": optimal["best_value"],
                "optimal_ig": optimal["best_ig"],
                "ig_gap": ig_gap,
                "ig_ratio": ig_ratio,
                "candidates_before": before,
                "candidates_after": after,
                "realised_info": realised_info,
                "answer": parsed_response,
                "questions_remaining": self.question_budget - self.seeker.questions_asked,
                "early_exit": self.early_exit,
            })

            turn += 1

        self.guess = self.seeker.make_guess()
        self.correct = self.guess.lower().strip() == self.oracle.hidden_country.lower().strip()
        self.game_over = True

        winner = "Seeker" if self.correct else "Oracle"

        self.log("\n=== RESULT ===")
        self.log(f"Won: {'Yes' if self.correct else 'No'}")
        self.log(f"Guess: {self.guess}")
        self.log(f"Correct: {self.oracle.hidden_country}")
        self.log(f"Questions used: {self.seeker.questions_asked}")
        self.log(f"Candidates at guess: {len(self.remaining_df)}")
        self.log(f"Winner: {winner}\n\n")

        return {
            "turn_metrics": self.turn_metrics,
            "game_summary": self.game_summary(),
            "transcript": "\n".join(self.transcript_lines)
        }

    def log(self, text):
        print(text)
        self.transcript_lines.append(text)

    def realised_information(self, before, after):
        if before > 0 and after > 0:
            return math.log2(before / after)
        return 0

    def result(self) -> dict:
        if not self.game_over:
            return None
        return {
            "guess": self.guess,
            "correct_answer": self.oracle.hidden_country,
            "correct": self.correct,
            "question_asked": self.seeker.questions_asked,
            "score": self.score
        }

    def extract_attribute(self, question_text: str) -> str | None:
        for line in question_text.split("\n"):
            if line.startswith("ATTRIBUTE:"):
                raw = line.replace("ATTRIBUTE:", "").strip().lower()
                return raw.split("|")[0].strip()
        return None

    def extract_value(self, question_text: str) -> str | None:
        for line in question_text.split("\n"):
            if line.startswith("VALUE:"):
                return line.replace("VALUE:", "").strip()
        return None

    def extract_question(self, question_text: str) -> str | None:
        for line in question_text.split("\n"):
            if line.startswith("QUESTION:"):
                return line.replace("QUESTION:", "").strip()
        return None

    def extract_response(self, answer_text: str) -> str | None:
        for line in answer_text.split("\n"):
            if line.startswith("RESPONSE:"):
                return line.replace("RESPONSE:", "").strip()
        return None

    def filter_candidates(self, df, question, answer):
        attribute = self.extract_attribute(question)
        value = self.extract_value(question)
        response = self.extract_response(answer)

        if attribute is None or attribute not in df.columns:
            print(f"WARNING: Could not parse attribute from question; skipping filter.")
            return df

        if response == "Yes":
            return df[df[attribute] == value]
        if response == "No":
            return df[df[attribute] != value]

        return df

    def calculate_best_question(self) -> dict:
        """
        Score every (attribute, value) predicate the Seeker could ask against
        the current candidate space, and return the one with highest information
        gain.

        Each predicate is a binary question of the form "Is <attribute> == <value>?".
        IG is the reduction in Shannon entropy when the two possible post-answer
        states are weighted by their probability of occurring.
        """
        candidate_df = self.remaining_df
        askable_attributes = self.askable_attributes
        total_candidates = len(candidate_df)

        if total_candidates <= 1:
            return {
                "best_attribute": None,
                "best_value": None,
                "best_ig": 0.0,
                "all_scores": [],
            }

        current_entropy = math.log2(total_candidates)
        scored_predicates = []

        for attribute in askable_attributes:
            value_counts = candidate_df[attribute].value_counts(dropna=True)

            for value, yes_count in value_counts.items():
                no_count = total_candidates - yes_count

                if yes_count == 0 or no_count == 0:
                    information_gain = 0.0
                else:
                    yes_probability = yes_count / total_candidates
                    no_probability = no_count / total_candidates
                    expected_entropy_after = (
                        yes_probability * math.log2(yes_count)
                        + no_probability * math.log2(no_count)
                    )
                    information_gain = current_entropy - expected_entropy_after

                scored_predicates.append((attribute, value, information_gain))

        scored_predicates.sort(key=lambda row: row[2], reverse=True)
        best_attribute, best_value, best_ig = scored_predicates[0]

        print(
            f"(IG = {best_ig:.4f} bits, candidates = {total_candidates})"
        )

        return {
            "best_attribute": best_attribute,
            "best_value": best_value,
            "best_ig": best_ig,
            "all_scores": scored_predicates,
        }

    def _lookup_seeker_ig(
        self,
        seeker_attribute: str | None,
        seeker_value: str | None,
        scored_predicates: list,
    ) -> float:
        """
        Find the IG of the Seeker's chosen (attribute, value) predicate within
        the pre-computed action space scoring.

        Returns 0.0 if the predicate is not found.
        """
        if seeker_attribute is None or seeker_value is None:
            return 0.0

        for attribute, value, information_gain in scored_predicates:
            if attribute == seeker_attribute and value == seeker_value:
                return information_gain

        return 0.0

    def game_summary(self):
        if self.turn_metrics:
            mean_ig_gap = sum(r["ig_gap"] for r in self.turn_metrics) / len(self.turn_metrics)
            mean_seeker_ig = sum(r["seeker_ig"] for r in self.turn_metrics) / len(self.turn_metrics)
            mean_realised_info = sum(r["realised_info"] for r in self.turn_metrics) / len(self.turn_metrics)
            total_zero_bit_turns = sum(1 for r in self.turn_metrics if r["seeker_ig"] == 0)
        else:
            mean_ig_gap = 0
            mean_seeker_ig = 0
            mean_realised_info = 0
            total_zero_bit_turns = 0

        return {
            "experiment_id": self.experiment_id,
            "game_id": self.game_id,
            "variant": self.variant,
            "hidden_country": self.oracle.hidden_country,
            "won": self.correct,
            "guess": self.guess,
            "questions_asked": self.seeker.questions_asked,
            "candidates_at_guess": len(self.remaining_df),
            "mean_ig_gap": mean_ig_gap,
            "mean_seeker_ig": mean_seeker_ig,
            "mean_realised_info": mean_realised_info,
            "total_zero_bit_turns": total_zero_bit_turns,
            "early_exit": self.early_exit,
            "early_exit_turn": self.early_exit_turn,
            "valid": len(self.remaining_df) > 0,
            "retry_count": self.retry_count,
        }