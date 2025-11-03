# 🎮 AI Tic-Tac-Toe (Heriot Watt University)

**Overview:**  
Developed an AI-powered Tic-Tac-Toe game in Java using **reinforcement learning** techniques (Q-Learning, Policy Iteration, Value Iteration). Trained multiple agents (Aggressive, Defensive, Random) to learn optimal gameplay strategies.

---

## ⚙️ How It Works
- **Environment:** `TTTEnvironment` defines board states, moves, and rewards (win=+1, draw=0, loss=-1).  
- **Q-Learning Agent:** Learns via **epsilon-greedy policy**, updating Q-values after each move.  
- **Policy & Value Iteration Agents:** Compute optimal policies/state-values via dynamic programming.  
- Agents interact with the environment over thousands of episodes to **converge on optimal strategies**.

---

## 🏁 Results
- **Q-Learning:** High win rate, near-zero losses after convergence.  
- **Policy Iteration:** Deterministic optimal policy with consistent results.  
- **Value Iteration:** Perfect decision-making and fast convergence.  
- **Random Agent:** Baseline performance for comparison.  

**Outcome:** All agents achieve **near-optimal play**, often resulting in wins or draws, validating reinforcement learning implementation.


