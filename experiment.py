import os
import json
import csv
import random
from datetime import date

import pandas as pd

from game_environment import GameEnvironment
from bot import Oracle


class ExperimentRunner:
    """
    Orchestrates a full experimental run across one or more Seeker variants.

    For each variant in seeker_classes, plays n_games against a
    Oracle using a shared set of randomly-sampled hidden countries (so variants
    are compared on identical games, not independent samples). Writes
    per-turn metrics, per-game summaries, full transcripts, and a config
    snapshot to a timestamped subdirectory of output_dir.
    """

    def __init__(self, experiment_id, client, model, df, seeker_classes, country_choice, attribute_moves, question_budget, n_games, seed=42, output_dir="experiments"):
        self.experiment_id = experiment_id
        self.client = client
        self.model = model
        self.df = df.copy()
        self.seeker_classes = seeker_classes
        self.country_choice = country_choice
        self.attribute_moves = attribute_moves
        self.question_budget = question_budget
        self.n_games = n_games
        self.seed = seed
        self.output_dir = output_dir

        random.seed(seed)

        self.run_dir = os.path.join(output_dir, experiment_id)
        self.transcript_dir = os.path.join(self.run_dir, "transcripts")

        os.makedirs(self.transcript_dir, exist_ok=True)

        self.metrics_path = os.path.join(self.run_dir, "metrics.csv")
        self.games_path = os.path.join(self.run_dir, "games.csv")
        self.config_path = os.path.join(self.run_dir, "config.json")

    def write_config(self, hidden_country):
        config = {
            "experiment_id": self.experiment_id,
            "date": str(date.today()),
            "model": self.model,
            "question_budget": self.question_budget,
            "n_games": self.n_games,
            "seed": self.seed,
            "variants": [cls.__name__ for cls in self.seeker_classes],
            "hidden_countries": hidden_country,
        }

        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=4)

    def run(self):
        hidden_countries = random.sample(self.country_choice, self.n_games)
        self.write_config(hidden_countries)

        all_turn_rows = []
        all_game_rows = []

        game_counter = 1

        for seeker_class in self.seeker_classes:
            variant = seeker_class.__name__

            for hidden_country in hidden_countries:
                game_id = f"{game_counter:03d}"

                seeker = seeker_class(
                    client=self.client,
                    model=self.model,
                    question_budget=self.question_budget,
                    attribute_moves=self.attribute_moves
                )

                oracle = Oracle(
                    client=self.client,
                    model=self.model,
                    country_choice=[hidden_country],

                    question_budget=self.question_budget,
                    attribute_moves=list(self.attribute_moves.keys()),
                    df =self.df
                )

                oracle.set_hidden_country(hidden_country)

                game = GameEnvironment(
                    seeker=seeker,
                    oracle=oracle,
                    df=self.df,
                    experiment_id=self.experiment_id,
                    game_id=game_id,
                    variant=variant,
                    attribute_moves=self.attribute_moves,
                )

                seeker.game = game
                oracle.game = game
                
                result = game.run()

                all_turn_rows.extend(result["turn_metrics"])
                all_game_rows.append(result["game_summary"])

                transcript_path = os.path.join(
                    self.transcript_dir,
                    f"game_{game_id}.txt"
                )

                with open(transcript_path, "w", encoding="utf-8") as f:
                    f.write(result["transcript"])

                game_counter += 1

        self.write_metrics(all_turn_rows)
        self.write_games(all_game_rows)


    def write_metrics(self, rows):
        fieldnames = [
            "experiment_id", "game_id", "variant", "hidden_country", "turn",
            "seeker_attribute", "seeker_value", "seeker_ig",
            "optimal_attribute", "optimal_value", "optimal_ig",
            "ig_gap", "ig_ratio",
            "candidates_before", "candidates_after", "realised_info",
            "answer", "questions_remaining", "early_exit"
        ]

        self.write_csv(self.metrics_path, rows, fieldnames)

    def write_games(self, rows):
        fieldnames = [
            "experiment_id", "game_id", "variant", "hidden_country",
            "won", "guess", "questions_asked", "candidates_at_guess",
            "mean_ig_gap", "mean_seeker_ig", "mean_realised_info",
            "total_zero_bit_turns", "early_exit", "early_exit_turn",
            "valid", "retry_count"
        ]

        self.write_csv(self.games_path, rows, fieldnames)

    def write_csv(self, path, rows, fieldnames):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for row in rows:
                writer.writerow(row)