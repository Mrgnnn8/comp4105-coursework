from openai import OpenAI
from bot import PlanningSeeker, Oracle, ZeroShotSeeker, ToTSeeker
from country import country_choice
import os
import pandas as pd
from attribute_criteria import ATTRIBUTE_MOVES
from experiment import ExperimentRunner


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

model = "gpt-5.4-nano" 
questions = 10 # Sets the number of questions the seeker can ask per game.
number_of_games = 1 # Sets the number of games each seeker variant plays.

df = pd.read_csv("countries.csv", keep_default_na=False, na_values=[""]) # Loads the dataset containing ground truth values

runner = ExperimentRunner(
    experiment_id="exp_001",
    client=client,
    model=model,
    df=df,
    seeker_classes=[ZeroShotSeeker, PlanningSeeker, ToTSeeker], # Determines which seeker variants are in tested in the experiment
    country_choice=country_choice,
    attribute_moves=ATTRIBUTE_MOVES,
    question_budget=questions,
    n_games=number_of_games,
    seed=42,
    output_dir="experiments"
)

runner.run()