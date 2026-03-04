# WTRP: Weighted Traveling Repairman Problem for Semantic-aware Frontier Exploration

> For paper writing reference. Contains formal definitions, mathematical formulations, and algorithm pseudocode.

---

## 1. Problem Formulation

### 1.1 Motivation

Traditional frontier-based exploration faces a fundamental trade-off:
- **Greedy Closest**: Minimizes travel distance but ignores semantic relevance
- **Greedy Highest Value**: Prioritizes high-value frontiers but may cause inefficient long-distance traversal

We propose **WTRP (Weighted Traveling Repairman Problem)** to balance exploration efficiency with target discovery probability by minimizing the **weighted arrival time** to semantically valuable frontiers.

### 1.2 Key Insight

Unlike TSP which minimizes total path length, WTRP minimizes the weighted sum of arrival times. This ensures that **high-value frontiers are visited earlier**, reducing the expected time to discover the target object.

---

## 2. Notation

| Symbol | Definition |
|--------|------------|
| $\mathcal{F} = \{f_1, f_2, \ldots, f_n\}$ | Set of frontier candidates |
| $p_0$ | Current agent position |
| $V(f_i)$ | Value function of frontier $f_i$ |
| $c(p_i, p_j)$ | Travel cost from $p_i$ to $p_j$ (A* path length) |
| $\pi: \{1,\ldots,n\} \to \{1,\ldots,n\}$ | Permutation representing visit order |
| $T_\pi(i)$ | Cumulative arrival time at the $i$-th visited node |
| $w_i$ | Weight of frontier $f_i$ (derived from value via softmax) |
| $k$ | Number of top-k candidates |
| $\tau$ | Softmax temperature parameter |
| $\alpha$ | Semantic-information fusion weight |

---

## 3. Value Function: Dual-Value Fusion

The total value of a frontier combines **semantic value** and **information gain**:

$$V(f_i) = \alpha \cdot V_{sem}(f_i) + (1-\alpha) \cdot V_{ig}(f_i)$$

where:
- $V_{sem}(f_i) = \max_{p \in \mathcal{N}(f_i)} \text{HSVM}(p)$: Maximum semantic value in neighborhood
- $V_{ig}(f_i) = \max_{p \in \mathcal{N}(f_i)} \text{IG}(p)$: Maximum information gain in neighborhood
- $\mathcal{N}(f_i)$: Local neighborhood of $f_i$ (5×5 grid)

---

## 4. WTRP Formulation

### 4.1 Objective Function

Given the top-k frontiers $\mathcal{F}_k$ sorted by value, WTRP finds the optimal visit order $\pi^*$ that minimizes the **weighted arrival time**:

$$\pi^* = \arg\min_{\pi \in \Pi_k} \sum_{i=1}^{k} w_{\pi(i)} \cdot T_\pi(i)$$

where the cumulative arrival time is:

$$T_\pi(i) = \sum_{j=1}^{i} c(p_{\pi(j-1)}, p_{\pi(j)}), \quad p_{\pi(0)} = p_0$$

### 4.2 Weight Computation via Softmax

Frontier weights are computed using temperature-scaled softmax:

$$w_i = \frac{\exp\left(\frac{V(f_i) - V_{\max}}{\tau}\right)}{\sum_{j=1}^{k} \exp\left(\frac{V(f_j) - V_{\max}}{\tau}\right)}$$

**Temperature Effect**:
| $\tau$ | Weight Distribution | Behavior |
|--------|---------------------|----------|
| $\tau \to 0$ | One-hot | Only highest-value frontier matters |
| $\tau = 0.5$ | Moderate | Balanced value differentiation |
| $\tau \to \infty$ | Uniform | Degenerates to standard TRP |

### 4.3 Objective Function Expansion

Let $c_i = c(p_{\pi(i-1)}, p_{\pi(i)})$ denote the edge cost. The objective can be rewritten as:

$$\begin{aligned}
J(\pi) &= \sum_{i=1}^{n} w_{\pi(i)} \cdot T_\pi(i) \\
&= w_{\pi(1)} c_1 + w_{\pi(2)}(c_1 + c_2) + w_{\pi(3)}(c_1 + c_2 + c_3) + \cdots \\
&= c_1 \sum_{j=1}^{n} w_{\pi(j)} + c_2 \sum_{j=2}^{n} w_{\pi(j)} + c_3 \sum_{j=3}^{n} w_{\pi(j)} + \cdots \\
&= \sum_{i=1}^{n} c_i \cdot W_i^{tail}
\end{aligned}$$

where $W_i^{tail} = \sum_{j=i}^{n} w_{\pi(j)}$ is the **tail weight sum**.

**Key Insight**: Each edge contributes its length multiplied by the sum of weights of all subsequent nodes. Thus:
- High-weight nodes should be visited early (to reduce tail weight multipliers)
- But the travel cost to reach them must also be considered

---

## 5. Solution Methods

### 5.1 Exact Solution: Brute-Force Enumeration (for $n \leq 10$)

Enumerate all $n!$ permutations and select the one with minimum weighted arrival time.

**Time Complexity**: $O(n! \cdot n)$

### 5.2 Approximate Solution: Greedy Heuristic (for $n > 10$)

At each step, select the unvisited frontier that minimizes the **marginal cost**:

$$j^* = \arg\min_{j \in \mathcal{U}} c(p_{cur}, p_j) \cdot W_{rem}$$

where $W_{rem} = \sum_{j \in \mathcal{U}} w_j$ is the remaining weight sum.

**Intuition**: Choosing node $j$ increases the arrival time of all remaining nodes by $c(p_{cur}, p_j)$. The marginal cost accounts for both $j$'s weighted arrival and the delay imposed on subsequent nodes.

**Time Complexity**: $O(n^2)$

---

## 6. Algorithm Pseudocode

### Algorithm 1: WTRP-based Frontier Selection

```
Input: Position p₀, Frontiers F, Parameters k, τ, α
Output: Next target frontier f*

1:  for each f ∈ F do
2:      V(f) ← α · V_sem(f) + (1-α) · V_ig(f)     // Dual-Value Fusion
3:  end for
4:  F_k ← TopK(F, k, by=V)                         // Top-k selection
5:  F_r ← {f ∈ F_k : A*(p₀, f) ≠ ∅}               // Reachability filter
6:  w ← Softmax(V(F_r) / τ)                        // Compute weights
7:  C ← BuildCostMatrix(p₀, F_r)                   // A* path lengths
8:  if |F_r| ≤ N_bf then
9:      π* ← BruteForceWTRP(C, w)                  // Exact solution
10: else
11:     π* ← GreedyWTRP(C, w)                      // Heuristic solution
12: end if
13: return F_r[π*(1)]                              // First node in optimal order
```

### Algorithm 2: Brute-Force WTRP Solver

```
Input: Cost matrix C ∈ ℝ^{(n+1)×(n+1)}, Weights w ∈ ℝ^n
Output: Optimal order π*

1:  π* ← null, J* ← ∞
2:  for each permutation π of {1, ..., n} do
3:      T ← 0, J ← 0
4:      for i = 1 to n do
5:          T ← T + C[π(i-1), π(i)]                // Cumulative arrival time
6:          J ← J + w[π(i)] · T                    // Weighted arrival cost
7:      end for
8:      if J < J* then
9:          J* ← J, π* ← π
10:     end if
11: end for
12: return π*
```

### Algorithm 3: Greedy WTRP Solver

```
Input: Cost matrix C ∈ ℝ^{(n+1)×(n+1)}, Weights w ∈ ℝ^n
Output: Approximate order π

1:  U ← {1, ..., n}, π ← [], cur ← 0
2:  while U ≠ ∅ do
3:      W_rem ← Σ_{j∈U} w[j]                       // Remaining weight sum
4:      j* ← argmin_{j∈U} C[cur, j] · W_rem        // Minimum marginal cost
5:      π.append(j*)
6:      U ← U \ {j*}, cur ← j*
7:  end while
8:  return π
```

---

## 7. Comparison with TSP-based Methods

| Aspect | TSP (e.g., ApexNav) | WTRP (Ours) |
|--------|---------------------|-------------|
| **Objective** | Minimize total path length | Minimize weighted arrival time |
| **Formula** | $\min \sum_i C_{\pi(i), \pi(i+1)}$ | $\min \sum_i w_{\pi(i)} \cdot T_\pi(i)$ |
| **Value Role** | Filtering only | Filtering + ordering weights |
| **High-value Priority** | Not guaranteed | Guaranteed (via weights) |
| **Solver** | LKH-3 heuristic | Brute-force / Greedy |

---

## 8. Complexity Analysis

| Method | Time Complexity | Space Complexity | Optimality |
|--------|-----------------|------------------|------------|
| Greedy Closest | $O(n)$ | $O(1)$ | Local |
| Greedy Highest Value | $O(n \log n)$ | $O(n)$ | Local |
| TSP (LKH-3) | $O(n^2 \log n)$ | $O(n^2)$ | Near-optimal |
| **WTRP (Brute-force)** | $O(k! \cdot k)$ | $O(k^2)$ | **Optimal** |
| **WTRP (Greedy)** | $O(k^2)$ | $O(k^2)$ | Near-optimal |

where $k$ is the top-k parameter (default: 5).

---

## 9. LaTeX Formulas

```latex
% Dual-Value Fusion
V(f_i) = \alpha \cdot V_{sem}(f_i) + (1-\alpha) \cdot V_{ig}(f_i)

% Softmax Weights
w_i = \frac{\exp\left(\frac{V(f_i) - V_{\max}}{\tau}\right)}{\sum_{j=1}^{k} \exp\left(\frac{V(f_j) - V_{\max}}{\tau}\right)}

% WTRP Objective
\pi^* = \arg\min_{\pi \in \Pi_k} \sum_{i=1}^{k} w_{\pi(i)} \cdot T_\pi(i)

% Cumulative Arrival Time
T_\pi(i) = \sum_{j=1}^{i} c(p_{\pi(j-1)}, p_{\pi(j)})

% Objective Expansion
J(\pi) = \sum_{i=1}^{n} c_i \cdot W_i^{tail}, \quad W_i^{tail} = \sum_{j=i}^{n} w_{\pi(j)}

% Greedy Selection Criterion
j^* = \arg\min_{j \in \mathcal{U}} c(p_{cur}, p_j) \cdot W_{rem}
```

---

## 10. References

1. **Traveling Repairman Problem**: Blum, A., Chalasani, P., Coppersmith, D., Pulleyblank, B., Raghavan, P., & Sudan, M. (1994). The minimum latency problem. *STOC*.

2. **Weighted TRP**: Chakrabarty, D., & Swamy, C. (2011). Facility location with service installation costs. *SODA*.

3. **LKH-3**: Helsgaun, K. (2017). An extension of the Lin-Kernighan-Helsgaun TSP solver for constrained traveling salesman and vehicle routing problems. *Technical Report*.

4. **Frontier-based Exploration**: Yamauchi, B. (1997). A frontier-based approach for autonomous exploration. *CIRA*.

---

*Generated for InfoNav paper writing. Last updated: 2024.*
