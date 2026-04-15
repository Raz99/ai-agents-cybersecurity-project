# AI Agents in Cybersecurity - Part 3

## Project Goal

This project was created for the implementation part of the course assignment in **Advanced Topics in Cybersecurity**.

The implementation option chosen for this project is:

**Compare two learning algorithms on a cybersecurity task**

In this project, we compare:

- a **Random Agent** baseline
- a **DQN (Deep Q-Network) Agent**

Both agents are tested on a small custom cybersecurity environment that simulates a simple cyber kill chain.

## Connection to the Paper

The selected paper is:

**Building adaptative and transparent cyber agents with local language models**

The original paper focuses on autonomous cyber agents based on language models in the NetSecGame environment.

This project does not try to fully reproduce the paper. Instead, it implements a simplified cybersecurity task that is related to the main idea of the paper: an autonomous agent making sequential decisions in a cyber scenario.

## Environment

The environment is implemented in [`cyber_env.py`](/home/razco/Uni/AI-Agent-Project/cyber_env.py).

It includes 5 states:

1. `Start`
2. `Scanned`
3. `Vulns_Found`
4. `Exploited`
5. `Exfiltrated`

The agent has 4 possible actions:

1. `scan_network`
2. `search_vulns`
3. `run_exploit`
4. `steal_data`

The goal is to move through the stages in the correct order and reach the final exfiltration state.

## Agents

### Random Agent

Implemented in [`random_agent_runner.py`](/home/razco/Uni/AI-Agent-Project/random_agent_runner.py).

This agent chooses actions randomly and does not learn.

### DQN Agent

Implemented in [`dqn_agent_runner.py`](/home/razco/Uni/AI-Agent-Project/dqn_agent_runner.py).

This agent learns from experience using reinforcement learning.

The implementation includes:

- a neural network
- replay buffer
- target network
- epsilon-greedy exploration

The code is designed to run on **CPU only**, which is enough for this small project.

## Dataset

The file [`data/netsecdata.csv`](/home/razco/Uni/AI-Agent-Project/data/netsecdata.csv) is used in a simplified way.

It is not used to train the DQN directly.

Instead, the code reads the dataset and uses it to estimate transition probabilities for the environment. This makes the environment less deterministic and connects the project to the data context of the selected paper.

## Main File

The main script is [`compare_agents.py`](/home/razco/Uni/AI-Agent-Project/compare_agents.py).

It does the following:

1. loads the dataset
2. builds the environment
3. runs the Random Agent
4. runs the DQN Agent
5. compares their results
6. saves graphs

## Evaluation

The project compares the agents using:

- average reward
- average steps per episode
- final 100-episode average reward
- wrong actions
- failed episodes

The generated graphs are:

- `reward_comparison.png`
- `steps_comparison.png`

## Files

- [`compare_agents.py`](/home/razco/Uni/AI-Agent-Project/compare_agents.py): main experiment
- [`cyber_env.py`](/home/razco/Uni/AI-Agent-Project/cyber_env.py): environment and dataset handling
- [`random_agent_runner.py`](/home/razco/Uni/AI-Agent-Project/random_agent_runner.py): random baseline agent
- [`dqn_agent_runner.py`](/home/razco/Uni/AI-Agent-Project/dqn_agent_runner.py): DQN agent
- [`requirements.txt`](/home/razco/Uni/AI-Agent-Project/requirements.txt): dependencies
- [`data/netsecdata.csv`](/home/razco/Uni/AI-Agent-Project/data/netsecdata.csv): dataset

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

If needed, PyTorch can also be installed as CPU-only:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## How to Run

Run the project with:

```bash
python compare_agents.py
```

## Expected Result

The expected result is that the DQN agent will perform better than the Random Agent because it learns which actions are better in each state.

This supports the main goal of the implementation task: comparing two learning approaches on a cybersecurity-related problem.

## Authors
- Aliza Lazar
- Raz Cohen