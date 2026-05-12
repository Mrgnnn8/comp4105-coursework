import os
import json
import random
import pandas as pd
from openai import OpenAI
from abc import ABC, abstractmethod
from country import country_choice
from attribute_criteria import ATTRIBUTE_MOVES


#load dataset
def initial_count(df):
    df= df.iloc[:, 0]
    inital_count = df.count()
    print(inital_count)
    return initial_count

class Brain(ABC):
    """
    Abstract base class for all agents in the game.

    Defines the four-module architecture shared by every agent:
    profile (who am I and what is my goal), memory (what has happened
    so far), planning (what should I do next), and action (commit to
    an output). The act() method orchestrates these in sequence on
    every turn.
    """
    def __init__(self, client: str, role: str, question_budget: int, model: str, attribute_moves):
        self.api_client = client
        self.model = model

        self.role = role
        self.history = [] 
        self.max_history = 10
        self.question_budget = question_budget
        self.questions_asked = 0
        self.questions_remaining = self.question_budget - self.questions_asked
        self.attribute_moves = attribute_moves
        self.country_choice = country_choice

    @abstractmethod
    def profile(self) -> str:
        """
        Defines who the agent is and what its goal is.
        Returns a system prompt string.
        """
        pass

    def memory(self) -> str:
        """
        Retrieves and formats the conversation history for the injection into the prompt.
        Respects max_history to avoid overflowing context window.
        """
        recent = self.history[-self.max_history:]
        if not recent:
            return "No questions have been asked yet."

        formatted = []
        for i, exchange in enumerate(recent, start=1):
            formatted.append(
                f"Turn {i}: asked '{exchange['attribute']} == {exchange['value']}' "
                f"-> answer: {exchange['response']}"
            )
        return "\n".join(formatted)

    @abstractmethod
    def planning(self, context: str, history: str) -> str:
        """
        Reasons about what to do next given profile context and memory.
        Makes a separate LLM call so reasoning is observable and loggable.
        Returns a plan string that is passed to the action module.
        """
        pass

    @abstractmethod
    def action(self, plan: str, history: str) -> str:
        """
        Takes the plan and produces the actual game output.
        Returns a question (Seeker) or answer (Oracle).
        """
        pass

    def act(self) -> str:
        """
        Orchestrates the four modules in sequence to build a prompt which is fed into the model.
        """
        context = self.profile()
        history = self.memory()
        self.last_plan = self.planning(context, history)
        output = self.action(self.last_plan, history)
        return output
    
    def call_llm(self, input: str) -> str:
        response = self.api_client.responses.create(
            model=self.model,
            instructions=self.profile(),
            input=input
        )
        response = response.output_text.strip()
        return response

    def update_history(self, attribute: str, value: str, response: str):
        self.history.append({
            "attribute": attribute,
            "value": value,
            "response": response,
        })


class PlanningSeeker(Brain):
    """
    Seeker variant that adds an explicit planning step before each question.

    On every turn, the agent first runs a reasoning pass over the game
    history, remaining candidates, and action space to produce a short
    natural-language plan. The plan is then passed to a separate action
    step that commits to a single (ATTRIBUTE, VALUE) predicate in the
    required format.
    """
    def __init__(self, client: OpenAI, model: str, question_budget: int, attribute_moves):
        super().__init__(
            client=client,
            role="PlanningSeeker",
            model=model,
            question_budget=question_budget,
            attribute_moves=attribute_moves,
        )

        self.remaining_candidates = self.country_choice
        self.candidate_count = len(self.remaining_candidates)

    @property
    def variant_name(self) -> str:
        return self.__class__.__name__

    def profile(self) -> str:
        budget_remaining = self.question_budget - self.questions_asked
        attributes = (
            f"You may only ask questions about the following attributes: {', '.join(self.game._current_action_space)}. "
            if self.game._current_action_space else
            "You may ask yes/no questions about the country. "
        )
        return (
            f"You are playing the role of a seeker in a country guessing game"
            f"You are trying to identify a hidden country. "
            f"Your goal is to gain the most information and reduce the most uncertainty possible at a given state."
            f"You do this by choosing the attribute-value pair which achieves this."
            f"Its important you reason over your choice."
            f"After your questions are used up, you must commit to a single guess. "
            f"If your guess is wrong, you lose."
        )

    def planning(self, context: str, history: str) -> str:
        user = (
            f"Game history so far:\n{history}\n\n"
            f"Remaining candidates ({self.candidate_count}): {self.remaining_candidates}\n"
            f"Questions remaining: {self.question_budget - self.questions_asked}\n\n"
            f"Action space (attributes you may ask about and their possible values):\n"
            f"{self.game._current_action_space}\n\n"
            f"Reason about what you know so far and what your next question should be. "
            f"Be brief — 3-5 sentences."
        )
        plan = self.call_llm(user)
        return plan

    def action(self, plan: str, history: str) -> str:
        user = (
            f"Game history so far:\n{history}\n\n"
            f"Your reasoning and strategy:\n{plan}\n\n"
            "Based on this, commit to the next question.\n\n"
            "Do NOT repeat any (ATTRIBUTE, VALUE) pair you have already asked about — "
            "look at the game history above and avoid duplicates.\n\n"
            "You must follow these rules:\n"
            "- Select ONE attribute from the action space\n"
            "- You must choose ATTRIBUTE exactly from this list. Do not invent, rename, or paraphrase attributes."
            "- If the attribute has multiple values, specify which VALUE you are testing\n"
            "- Your question must correspond EXACTLY to (ATTRIBUTE == VALUE)\n"
            "- The question must be answerable with Yes or No\n\n"
            f"Action space:\n{self.game._current_action_space}\n\n"
            "Return ONLY in this exact format:\n"
            "ATTRIBUTE: <column name>\n"
            "VALUE: <value being tested>\n"
            "QUESTION: <natural language yes/no question>\n\n"
            "Do not include any explanation or extra text."
        )

        self.questions_asked += 1
        return self.call_llm(user)

    def make_guess(self) -> str:

        candidates_str = ", ".join(self.remaining_candidates)
        user = (
            f"Game history:\n{self.memory()}\n\n"
            f"Remaining possible countries: {candidates_str}\n\n"
            f"Based on the game history and remaining candidates above, "
            f"which country is it? Respond with only the country name, nothing else."
        )
        return self.call_llm(user)


class ZeroShotSeeker(PlanningSeeker):
    """
    Inherits from PlanningSeeker but overrides planning() to return an
    empty string, collapsing the two-step plan-then-act pipeline into a
    single LLM call.
    """
    def __init__(self, client: OpenAI, model: str, question_budget: int, attribute_moves):
        super().__init__(
            client=client,
            model=model,
            question_budget=question_budget,
            attribute_moves=attribute_moves,
        )

    def planning(self, context: str, history: str) -> str:
        # No reasoning step — return empty plan so action() handles
        # everything in one LLM call.
        return ""

    def action(self, plan: str, history: str) -> str:
        plan_section = (
            f"Your reasoning and strategy:\n{plan}\n\n"
            if plan.strip()
            else ""
        )
        
        user = (
            f"Game history so far:\n{history}\n\n"
            f"{plan_section}"
            "Based on the history above, commit to the next question.\n\n"
            "Do NOT repeat any (ATTRIBUTE, VALUE) pair you have already asked about — "
            "look at the game history above and avoid duplicates.\n\n"
            "You must follow these rules:\n"
            "- Select ONE attribute from the action space\n"
            "- You must choose ATTRIBUTE exactly from this list. Do not invent, rename, or paraphrase attributes."
            "- If the attribute has multiple values, specify which VALUE you are testing\n"
            "- Your question must correspond EXACTLY to (ATTRIBUTE == VALUE)\n"
            "- The question must be answerable with Yes or No\n\n"
            f"Action space:\n{self.game._current_action_space}\n\n"
            "Return ONLY in this exact format:\n"
            "ATTRIBUTE: <column name>\n"
            "VALUE: <value being tested>\n"
            "QUESTION: <natural language yes/no question>\n\n"
            "Do not include any explanation or extra text."
        )

        self.questions_asked += 1
        return self.call_llm(user)

class ToTSeeker(PlanningSeeker):
    """
    Seeker variant that uses Tree of Thoughts reasoning. The planning module
    generates N candidate questions in parallel LLM calls, then selects the
    branch whose proposed predicate has the highest information gain against
    the current candidate state.
    """

    def __init__(self, client: OpenAI, model: str, question_budget: int, attribute_moves):
        super().__init__(
            client=client,
            model=model,
            question_budget=question_budget,
            attribute_moves=attribute_moves,
        )
        self.n_branches = 5  # number of parallel reasoning branches per turn. Can be changed to produce more or less branches.

    def tree_of_thought(self, history: str) -> list[dict]:
        """
        Generate N branches of independent reasoning, each producing a candidate
        (attribute, value) predicate. Each branch is shown the proposals from
        prior branches to enforce diversity across reasoning paths.
        Returns the raw branches; selection happens in planning().
        """
        branches = []

        for branch_index in range(1, self.n_branches + 1):
            already_proposed = (
                "The following (attribute, value) pairs have already been proposed in earlier branches "
                "— you MUST NOT propose any of these:\n" +
                "\n".join(f"- {b['attribute']} == {b['value']}" for b in branches) + "\n\n"
            ) if branches else ""

            user = (
                f"You are exploring reasoning branch {branch_index} of {self.n_branches} "
                f"for the next question to ask.\n\n"
                f"Game history so far:\n{history}\n\n"
                f"Remaining candidates ({self.candidate_count}): {self.remaining_candidates}\n\n"
                f"{already_proposed}"
                f"Questions remaining: {self.question_budget - self.questions_asked}\n\n"
                f"Action space (attributes you may ask about and their possible values):\n"
                f"{self.game._current_action_space}\n\n"
                f"Propose ONE candidate question for this branch. Reason briefly about "
                f"why this question is a reasonable choice given what is known. "
                f"Different branches should explore different reasoning paths.\n\n"
                f"Return in this exact format:\n"
                f"REASONING: <2-3 sentences>\n"
                f"ATTRIBUTE: <column name from the action space>\n"
                f"VALUE: <value being tested>\n"
            )

            response = self.call_llm(user)

            attribute = self._parse_field(response, "ATTRIBUTE:")
            value = self._parse_field(response, "VALUE:")
            reasoning = self._parse_field(response, "REASONING:")

            branches.append({
                "branch_number": branch_index,
                "reasoning": reasoning,
                "attribute": attribute,
                "value": value,
            })

        return branches

    def _parse_field(self, response: str, prefix: str) -> str | None:
        """Extract a single labelled field from an LLM response."""
        for line in response.split("\n"):
            if line.startswith(prefix):
                return line.replace(prefix, "").strip()
        return None

    def _select_best_branch(self, branches: list[dict], history: str) -> dict | None:
        """
        Ask the LLM to evaluate the candidate branches and select the best one.
        This keeps the selection within the LLM's reasoning rather than using
        external information-theoretic scoring.
        """
        if not branches:
            return None

        branch_summary = "\n".join([
            f"Branch {b['branch_number']}: {b['attribute']} == {b['value']}\n"
            f"  Reasoning: {b['reasoning']}"
            for b in branches
        ])

        user = (
            f"Game history so far:\n{history}\n\n"
            f"Remaining candidates ({self.candidate_count}): {self.remaining_candidates}\n\n"
            f"You have generated {len(branches)} candidate questions:\n\n"
            f"{branch_summary}\n\n"
            f"Select the single best question from these branches. "
            f"Consider which question would most effectively narrow down the remaining candidates. "
            f"Reply with ONLY the branch number.\n"
            f"SELECTED_BRANCH: <number>"
        )

        response = self.call_llm(user)

        try:
            selected = int(
                [l for l in response.split("\n") if l.startswith("SELECTED_BRANCH:")][0]
                .replace("SELECTED_BRANCH:", "").strip()
            )
            chosen = next((b for b in branches if b["branch_number"] == selected), None)
            return chosen if chosen else branches[0]
        except (IndexError, ValueError):
            return branches[0]

    def _log_branches(self, branches: list[dict], chosen: dict | None):
        """Append the round's branches to a transcript file for later inspection."""
        with open("tree_of_thoughts.txt", "a", encoding="utf-8") as f:
            f.write(
                f"\n--- Question {self.questions_asked + 1}/{self.question_budget} "
                f"| Candidates: {self.candidate_count} | Branches: {len(branches)} ---\n"
            )
            for branch in branches:
                ig = branch.get("information_gain")
                ig_str = f"{ig:.4f}" if ig is not None else "n/a"
                marker = "CHOSEN" if branch is chosen else ""
                f.write(
                    f"Branch {branch['branch_number']}: "
                    f"{branch['attribute']} == {branch['value']} "
                    f"  Reasoning: {branch['reasoning']}\n"
                )

    def planning(self, context: str, history: str) -> str:
        """
        Generate parallel reasoning branches, select the highest-IG branch
        deterministically, and return its reasoning as the plan that action()
        will commit to.
        """
        branches = self.tree_of_thought(history)
        chosen = self._select_best_branch(branches, history)

        self._log_branches(branches, chosen)

        if chosen is None or chosen["attribute"] is None:
            return super().planning(context, history)

        plan = (
            f"After considering {self.n_branches} candidate questions, "
            f"the best option targets {chosen['attribute']} == {chosen['value']}.\n"
            f"Reasoning: {chosen['reasoning']}"
        )
        
        return plan


class Oracle(Brain):
    """
    Deterministic Oracle for the country-guessing game.

    Inherits from Brain for interface compatibility with the Seeker variants,
    but does not use the LLM-powered profile/memory/planning/action modules.
    Instead, every question is answered by a direct lookup against the hidden
    country's row in the ground-truth DataFrame.

    In future work, this class can added to, to give variation to the Oracle.
    """

    def __init__(self, client: OpenAI, model: str, question_budget: int,
                 country_choice: list, attribute_moves, df):
        super().__init__(
            client=client,
            role="oracle",
            model=model,
            question_budget=question_budget,
            attribute_moves=attribute_moves,
        )

        self.df = df.copy()
        self.df.columns = self.df.columns.str.strip().str.lower()
        self.df["country"] = self.df["country"].astype(str).str.strip().str.lower()

        self.hidden_country = None
        self.hidden_row = None

    def set_hidden_country(self, country: str) -> None:
        self.hidden_country = self.normalise(country)
        self.hidden_row = self.df[self.df["country"] == self.hidden_country].iloc[0]

    def normalise(self, value) -> str:
        if value is None:
            return "none"
        return str(value).strip().lower()

    def truthful_answer(self, attribute: str, value: str) -> str:
        if attribute not in self.hidden_row.index:
            raise ValueError(
                f"Invalid attribute '{attribute}'. "
                f"Valid columns are: {list(self.hidden_row.index)}"
            )

        hidden_value = self.normalise(self.hidden_row[attribute])
        asked_value = self.normalise(value)

        return "Yes" if hidden_value == asked_value else "No"

    def act(self) -> str:
        question = self.game.question
        attribute = self.game.extract_attribute(question)
        value = self.game.extract_value(question)
        truthful = self.truthful_answer(attribute, value)
        return f"RESPONSE: {truthful}"
        
    def profile(self) -> str:
        return ""

    def planning(self, context: str, history: str) -> str:
        return ""

    def action(self, plan: str, history: str) -> str:
        return ""
        
