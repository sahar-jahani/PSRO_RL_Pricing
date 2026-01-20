# Empirical Game-Theoretic Analysis of Pricing Games Using Reinforcement Learning

This repository contains the implementation of the experiments studied in **Part II** of the published [PhD thesis](https://etheses.lse.ac.uk/4921/). The experiments were run on [Fabian, the High Performance Computing (HPC)](https://info.lse.ac.uk/staff/divisions/dts/services/fabian/Fabian) cluster at the London School of Economics.

This project studies equilibrium behavior and emergent collusion in a **multi-round asymmetric duopoly pricing game with demand inertia**, with a particular focus on identifying stable outcomes that yield higher returns than the **Subgame Perfect Equilibrium (SPE)**.

The underlying class of pricing games was originally studied by **Reinhard Selten (1965)**, who characterized and computed the SPE. Later, **Claudia Keser (1992)** conducted experimental studies of this game, in which human-submitted strategies (represented as flowcharts) were evaluated in a competitive tournament setting.

In this work, we extend this line of research by using **reinforcement learning (RL)** to model strategic behavior in the pricing game. We train **low-cost** and **high-cost** pricing agents to compete in a **finite-horizon (25-round)** version of the game studied by Keser. The goal is to empirically study the **Nash equilibria** of the evolving game and to identify the conditions under which independently trained learning agents exhibit **collusive behavior**. Understanding these conditions is important because it provides insights into how to prevent sellers from colluding on pricing in the market, thereby protecting consumer rights.

---

## Methodology

To approximate the infinite strategy space of the pricing game, we use the **Policy Space Response Oracles (PSRO)** framework. PSRO constructs an evolving meta-game over a finite set of strategies, providing an empirical approximation to the full game.

The procedure is as follows:

- The meta-game is initialized with a set of pricing strategies, which may be random or predefined deterministic policies.
- An equilibrium of the current meta-game is computed.
- New RL agents are trained as approximate best responses to the equilibrium strategies using policy-gradient methods.
- If a newly trained agent achieves higher expected returns than the current equilibrium payoff, it is added to the meta-game.
- This iterative process continues, expanding the strategy space and refining the meta-game.

Through repeated iterations, the meta-game grows to represent increasingly sophisticated strategic behavior. The equilibria computed in the meta-game serve as **empirical approximations** to equilibria in the original infinite-strategy pricing game.

---

## 🧠 Evolutionary Framework

- We use reinforcement learning algorithms implemented in the [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) library.  
  The framework is fully compatible with all learning algorithms provided by this library, including:
  - **Proximal Policy Optimization (PPO)**
  - **Soft Actor-Critic (SAC)**
- The pricing game is implemented as a custom environment compatible with the **Gymnasium API**, defined in the `ConPricingGame` class.


## ⚙️ Multiprocessing & Data Management

- **Multiprocessing** is used to accelerate training and evaluation.
- The `BimatrixGame` class represents the evolving meta-game.
- All data, including trained agents, meta-games, equilibria, and logs, is stored using custom-designed data management classes and database. 

## 🎯 Equilibrium Enumeration in PSRO

Nash equilibria of the evolving bimatrix meta-game, referred to as **Meta-Strategy Solver** in the **Policy Space Response Oracles (PSRO)** framework, are computed using the **Lemke algorithm**, implemented by Prof. **Bernhard von Stengel**.

📁 The source code for the equilibrium solver is located in the `src/equilibrium_solver/` directory.

## 🛠️ Installation

Before running the training script, make sure you have the required packages installed.

### Install Dependencies

Install [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3):
```bash
pip install stable-baselines3
```

Install [TensorBoard]() for monitoring training progress:
```bash
pip install tensorboard
```
## 🚀 How to Run

To start the training process, set the following parameters in `main_psro_rl.py`, then run the script.

### Parameters to Set:

- **`game_name`**  
  The prefix used for all files and folders generated during the experiment.  
  ⚠️ **Use the same name to resume an existing experiment.**

- **`num_procs`**  
  The number of parallel processes for training, based on your system’s available CPU cores.

### Run the Script

```bash
python main_psro_rl.py
```

### ⚙️ Configuration

All constants and hyperparameters for both the pricing game and training procedure are defined in `globals.py`.

This includes:

- Agent production costs  
- Total demand potential  
- Number of time steps in the pricing game  
- Agents' memory of prices in the observations  
- Number of episodes for training each agent  
- Number of equilibria to follow in each round of the meta-game  
- Hyperparameters for the SAC and PPO algorithms  

The hyperparameters were determined through prior experimentation to ensure efficient and stable learning in this setting.

---
## 📁 Project Structure and Output Files

When you start running `main_psro_rl.py`, the following directories and files will be created in the base folder:

### `games/`
Stores snapshots of meta-games. Each file includes:
- The first line: number of rows and columns.
- The first matrix: payoffs for the **Low-cost** player.
- The second matrix: payoffs for the **High-cost** player.
- Strategy names for both players at the end of the file.

These files serve as a complete record of the evolving meta-game structure across training rounds.

### `models/`
Contains all trained agent models.  
Model files are named using the format: `<game_name>-<timestamp>.zip`

These models can be used for evaluation or to resume training.  
- Models corresponding to strategies in the meta-game are **essential** for continuing training, as new agents may need to train against them if they appear in future equilibria.  
- Models from earlier iterations that were **not added** to the game might still be useful for initializing future agents but can be removed later if needed to save space.

### `logs/`
Contains TensorBoard-compatible training logs, named to match their corresponding models.

To monitor training progress with TensorBoard, run the following command from the base directory:

```bash
tensorboard --logdir logs/
```

### 📄 Base Folder Files

#### `game_<game_name>.txt`
Text representation of the current meta-game used by the equilibrium solver.  
⚠️ **Do not delete** — This file is required for equilibrium computation.

#### `game_<game_name>.pickle`
Serialized `BimatrixGame` object that stores the full meta-game and associated strategies.  
Used to resume training without recomputation.  
⚠️ **Deleting this file will require recomputing all payoffs**, which can take hours for large games.

#### `game_<game_name>.db`
SQLite database storing:
- Trained pricing strategies (pure strategies in the meta-game)
- Detailed results of pricing games between agents
- Meta-game equilibria
- Average probabilities of strategies in these equilibria

⚠️ **Required to resume training**.  
Deleting this file will cause the experiment to start from scratch.

#### `progress_<game_name>.txt`
Logs the training progress in plain text:
- Start time of training
- Meta-game equilibria at each round
- Strategies added during each round based on equilibrium

You can monitor this file during training to check progress.

#### `error.log`
If any error occurs during training, it will be logged here.  
Useful for debugging or identifying failed runs.

---
## 🔄 Resuming Training

To resume a previously stopped experiment:

- Use the **same `game_name`** when rerunning `main_psro_rl.py`.
- Ensure all the following files are present in the base folder:

  - `game_<game_name>.txt`
  - `game_<game_name>.pickle`
  - `game_<game_name>.db`
  - Trained model files in the `models/` directory

These files are required to reload the meta-game, agent strategies, and training logs.  
⚠️ If any of these are missing, the experiment must be restarted from scratch.

---
## 👤 Author

**Sahar Jahani**  
PhD in Mathematics – London School of Economics
