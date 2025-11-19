# Implementation Design Document: GRPO vs PPO for Discrete Control

## 1. Project Overview

This project aims to implement and compare two Reinforcement Learning algorithms—**Proximal Policy Optimization (PPO)** and **Group Relative Policy Optimization (GRPO)**—on discrete OpenAI Gymnasium environments (e.g., CartPole-v1). 

The implementation must bridge the gap between **LLM-centric** definitions (as presented in the DeepSeekMath paper) and **Classical Control** definitions (standard RL). The codebase will use a **3-file architecture** and rely on **Outcome Supervision** for GRPO.

---

## 2. Mathematical Formulation & Notation

To ensure consistency, we map the terminology from the DeepSeekMath paper to standard Gym RL concepts.

### 2.1 Definitions

| Symbol | LLM Paper Definition | Gym Implementation Definition | Description |
| :--- | :--- | :--- | :--- |
| **$q$** | Question / Prompt | **$s_0$** | The initial state of the environment after `reset()`. |
| **$o$** | Output / Completion | **$\tau$** | A full trajectory/episode: $(s_0, a_0, r_0, s_1, \dots, s_T)$. |
| **$G$** | Group Size | **$G$** | Number of parallel episodes sampled from the **exact same** $s_0$. |
| **$\pi_\theta$** | Policy Model | **Actor Network** | Neural network: State $s \to$ Action Logits (Categorical). |
| **$\pi_{old}$** | Old Policy | **Old Actor** | A frozen copy of the policy used for data collection. |
| **$A$** | Advantage | **$A$** | The relative quality of an action/episode compared to a baseline. |
| **$T_i$** | Output Length $|o_i|$ | **Episode Length** | The total number of steps survived in episode $i$. |
| **$R_i$** | Reward | **Episode Return** | The sum of rewards for episode $i$: $\sum_{t=0}^{T_i} r_t$. |

---

### 2.2 Algorithm A: Group Relative Policy Optimization (GRPO)

**Concept:** Actor-Only. No Value Function. The baseline is derived from the mean return of a group of trajectories sampled from the identical starting state.

#### 1. Sampling Strategy (The Group)
For every training step:
1.  Sample a random seed `S`.
2.  Reset the environment $G$ times using `env.reset(seed=S)`. This ensures every episode in the group starts at the exact same $s_0$ (the "Prompt").
3.  Collect $G$ full trajectories $\{o_1, o_2, \dots, o_G\}$.

#### 2. Advantage Calculation (Outcome Supervision)
For a specific group of $G$ episodes, we calculate the mean and standard deviation of their total returns:
$$\text{Mean} = \frac{1}{G} \sum_{i=1}^G R_i, \quad \text{Std} = \sqrt{\frac{1}{G} \sum_{i=1}^G (R_i - \text{Mean})^2}$$

The Advantage $A_i$ for episode $i$ is standardized:
$$A_i = \frac{R_i - \text{Mean}}{\text{Std} + \epsilon_{safe}}$$

**Crucial Implementation Detail:** This scalar Advantage $A_i$ is **broadcast** to every timestep $t$ within episode $i$. Every action in that episode gets equal credit/blame.

#### 3. The Objective Function
We maximize the following objective. Note the inclusion of the $1/T_i$ term to normalize for variable episode lengths.
$$\mathcal{J}_{GRPO}(\theta) = \frac{1}{G} \sum_{i=1}^G \left[ \frac{1}{T_i} \sum_{t=1}^{T_i} \min \left( \frac{\pi_\theta(a_{i,t}|s_{i,t})}{\pi_{old}(a_{i,t}|s_{i,t})} A_{i}, \text{clip}\left(\frac{\pi_\theta}{\pi_{old}}, 1-\epsilon, 1+\epsilon\right) A_{i} \right) \right]$$
*(Note: The KL Divergence term $\beta \mathbb{D}_{KL}$ is omitted for this Gym implementation as we rely on clipping for stability.)*

---

### 2.3 Algorithm B: Proximal Policy Optimization (PPO)

**Concept:** Actor-Critic. Uses a learned Value Function $V_\phi(s)$ as a baseline.

#### 1. Advantage Estimation (GAE)
For every timestep $t$, we compute the delta and Generalized Advantage Estimate:
$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$
$$A_t^{GAE} = \sum_{k=0}^{T-t-1} (\gamma \lambda)^k \delta_{t+k}$$

#### 2. The Objective Function
$$\mathcal{J}_{PPO}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right] - c_1 L_{VF} + c_2 S[\pi_\theta]$$
Where $L_{VF}$ is the Mean Squared Error loss for the Critic $(V_\phi(s_t) - R_t)^2$.

---

## 3. System Architecture

The solution must be implemented in **Python** using **PyTorch** and **Gymnasium**. It must follow a 3-file structure.

### File 1: `utils.py` (Infrastructure)

This file handles data structures, environment wrapping, and visualization.

**1. `MLP` Class (Neural Network)**
* Standard Feed-Forward Network: Input $\to$ Hidden (Tanh) $\to$ Hidden (Tanh) $\to$ Output.
* Used for both Actor (output `num_actions`) and Critic (output `1`).

**2. `SeededGymEnv` Class**
* Wrapper for `gymnasium`.
* **Requirement:** Must implement a `reset(seed=int)` method. This is critical for GRPO to replicate the "same prompt" scenario.

**3. `LivePlotter` Class**
* Uses `matplotlib.pyplot.ion()` for non-blocking real-time visualization.
* Must include `plt.pause(0.001)` to allow UI updates during the training loop.

**4. Buffer Classes**
* **`PPOBuffer`:** Stores flat transitions $(s, a, r, s', \log\pi)$. Implements GAE calculation.
* **`GRPOBuffer`:**
    * Stores data as a **List of Episodes**. Each episode is a dictionary/object containing tensors for states, actions, rewards, etc.
    * **Input:** Raw transitions.
    * **Processing:** On `end_episode()`, converts lists to Tensors and calculates the weight $w = 1/T_i$.
    * **Output:** Returns `List[Dict]` where each Dict is one episode.

---

### File 2: `algorithms.py` (Core Logic)

This file contains the agent classes. **Constraint:** Do NOT flatten all episodes into one giant batch at the start. Process them as a list of tensors to maintain group structure.

**1. `PPOAgent` Class**
* **Components:** `Actor` (MLP), `Critic` (MLP), `Optimizer`.
* **`get_action(state)`:** Returns action, log_prob.
* **`update(buffer)`:**
    * Computes GAE for all data.
    * Flattens data into a standard batch.
    * Performs standard PPO Mini-batch updates.

**2. `GRPOAgent` Class**
* **Components:** `Actor` (MLP), `Optimizer`. (No Critic).
* **`get_action(state)`:** Same as PPO.
* **`update(buffer)`:**
    * **Input:** A list of $N$ episode dictionaries (from `GRPOBuffer`).
    * **Logic:**
        1.  Iterate through the list in **chunks of size $G$** (Group Size).
        2.  **Group Norm:** For each chunk, extract episode returns, compute Mean/Std, and assign a scalar Advantage $A_i$ to each episode.
        3.  **Loss Calculation:** Iterate through all episodes. For each episode:
            * Forward pass current states $\to$ new log_probs.
            * Calculate Ratio $\frac{\pi_{new}}{\pi_{old}}$.
            * Compute element-wise loss: $\min(\text{ratio} \cdot A_i, \text{clip}(\dots) \cdot A_i)$.
            * **Weighting:** Multiply the summed loss by the episode weight $1/T_i$.
        4.  **Backprop:** Sum the losses from all episodes and perform one `optimizer.step()`.

---

### File 3: `main.py` (Controller)

**1. Configuration**
Use a configuration dictionary or constants at the top of the file to easily switch settings.

**2. Training Loops**
* **PPO Loop:**
    * Step-based iteration.
    * Collect $N$ steps.
    * Call `agent.update()`.
* **GRPO Loop:**
    * Iteration based on "Updates".
    * Sample a random seed.
    * Loop `range(G)`: Call `env.reset(seed)` and run full episode.
    * Store episodes in buffer.
    * Call `agent.update()` once buffer is full (e.g., after 4 groups).

**3. Utilities**
* **Model Saving:** Save `state_dict` every $N$ updates and save a `best_model.pt` when avg reward peaks.
* **Play Mode:** A function to load a model and run one episode with `render=True` for visual verification.

---

## 4. Hyperparameters & Tuning

DeepSeekMath uses LLM-specific hyperparameters (very low LR, very large groups). For Gym control tasks, we must use standard RL values.

| Hyperparameter | DeepSeekMath (Reference) | **Our Gym Config (Target)** |
| :--- | :--- | :--- |
| **Learning Rate** | `1e-6` | **`3e-4`** |
| **Group Size ($G$)** | `64` | **`16`** (GRPO only) |
| **Clip Range ($\epsilon$)** | `0.2` | **`0.2`** |
| **KL Coef ($\beta$)** | `0.04` | **`0.0`** (Disabled) |
| **Optimizer** | AdamW | **Adam** |
| **Update Frequency** | - | **2048 Steps** (PPO) / **32 Episodes** (GRPO) |
| **Advantage Norm** | Group Level | **Algorithm Specific** (Batch for PPO, Group for GRPO) |