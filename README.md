# Country Guessing Game — An Adversarial Multi-Agent LLM System

An LLM-agent system implementing a country guessing game, used to investigate whether LLM agents operate at the information-theoretically optimal decision boundary. Three Seeker variants (Zero-Shot, Planning, Tree of Thoughts) compete against a deterministic Oracle, with performance benchmarked against an information-theoretic optimum.

Submitted as coursework for COMP4105 Designing Intelligent Agents, University of Nottingham, 2025–26.

## Research Questions

- **RQ1:** How does the level of reasoning scaffolding affect the information-seeking efficiency of an LLM Seeker agent, measured against an information-theoretic optimum?
- **RQ2:** Do increased levels of reasoning scaffolding translate into measurable improvements in game outcomes?
- **RQ3:** How do increasing levels of reasoning scaffolding affect attribute-space exploration and category-level question distribution in an LLM Seeker agent?

## Installation

'''bash
python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
'''

### API key

The code reads `OPENAI_API_KEY` from the system environment. Set it before running:

```bash
export OPENAI_API_KEY="sk-..."    # macOS/Linux
# setx OPENAI_API_KEY "sk-..."    # Windows 
```

## Running Experiments

Experiment parameters (Seeker variant, number of games, question budget, etc.) are configured directly in `initialise.py`. Open the file and edit the configuration block at the top before running.

'run python initialise.py'

## Output

Running `initialise.py` creates an `experiments/` folder with a timestamped subdirectory for each run, containing:

- `games.csv` — per-game results
- `metrics.csv` — per-turn metrics
- A config JSON capturing the parameters used
- Full game transcripts for every game played

### Expected runtime and cost

A full 150-game experimental run (50 games × 3 Seeker variants) takes approximately **3 hours** and costs roughly **$5** in OpenAI API usage. Single-game test runs complete in seconds and cost fractions of a cent.

## Repository Layout

attribute_criteria.py   Scoring criteria for country attributes
attributes.py           Attribute taxonomy used by the Oracle
bot.py                  Seeker and Oracle agent classes (Brain base + subclasses)
country.py              Country object representation
countries.csv           Ground-truth country dataset
experiment.py           Experiment loop and run orchestration
game_environment.py     Game state, turn management, and scoring
initialise.py           Entry point — configures and launches an experiment
requirements.txt        Python dependencies
AI_USAGE.md             Declaration of AI tool usage

## Scope of this repository

This repository ships the experimental infrastructure, the agent system, environment, and data collection scripts. Statistical analysis of the collected data was performed separately and is not included here. The CSV outputs produced by this codebase are sufficient input for any downstream analysis a user wishes to perform.

## Attribution

All code in this repository is my own work. No code was taken directly from COMP4105 module class examples.

## Contact

Morgan Jones - 20750181 - psxmj4@nottingham.ac.uk