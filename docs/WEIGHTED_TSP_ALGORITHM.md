# Semantic-aware TSP / WTRP Algorithm for Frontier Exploration

> **设计初衷**：让智能体向着**近**且**价值高**的 frontier 进行导航，平衡探索效率与目标发现概率。

---

## 1. 问题概述

### 1.1 动机

传统的 frontier-based exploration 存在两种极端策略：
- **Greedy Closest**：总是选择最近的 frontier，忽略语义价值
- **Greedy Highest Value**：总是选择最高价值的 frontier，可能导致长距离迂回

我们提出的 **Semantic-aware TSP** 和 **WTRP** 算法通过以下方式实现平衡：
1. 按语义价值筛选 top-k 个候选 frontier
2. 在候选集合上进行路径优化（TSP 或 WTRP）
3. 导航到优化路径的第一个节点

### 1.2 算法模式

| 模式 | 代码枚举值 | 策略 | 求解器 |
|------|-----------|------|--------|
| `DISTANCE` | 0 | 贪心选择最近 frontier | - |
| `SEMANTIC` | 1 | 贪心选择最高价值 frontier | - |
| `HYBRID` | 2 | **Semantic-TSP**: Top-k 价值 + TSP 路径优化 | LKH-3 |
| `TSP_DIST` | 3 | 对所有 frontier 做 TSP（仅考虑距离） | LKH-3 |
| `WTRP` | 4 | **WTRP**: Top-k 价值 + 加权到达时间最小化 | 暴力枚举/贪心 |

### 1.3 两种算法的核心区别

| 方面 | Semantic-TSP (HYBRID) | WTRP |
|------|----------------------|------|
| **目标函数** | 最小化总路径长度 | 最小化加权到达时间 |
| **求解方法** | LKH-3 启发式 | 暴力枚举（小规模）/ 贪心（大规模） |
| **价值的作用** | 仅用于筛选 top-k | 同时用于筛选和计算访问顺序权重 |
| **优化重点** | 路径效率 | 高价值节点优先到达 |

---

## 2. 符号定义

| 符号 | 定义 |
|------|------|
| $\mathcal{F} = \{f_1, f_2, \ldots, f_n\}$ | Frontier 集合 |
| $p_0$ | 智能体当前位置 |
| $V(f_i)$ | Frontier $f_i$ 的价值函数 |
| $c(p_i, p_j)$ | 从位置 $p_i$ 到 $p_j$ 的路径代价（A* 路径长度） |
| $\pi: \{1,\ldots,n\} \to \{1,\ldots,n\}$ | 访问顺序的排列 |
| $T_{\pi}(i)$ | 在排列 $\pi$ 下到达第 $i$ 个节点的累计时间 |
| $w_i$ | Frontier $f_i$ 的权重（由价值通过 softmax 计算） |
| $k$ | Top-k 选择的数量（默认 5） |
| $\tau$ | Softmax 温度参数（默认 0.5） |
| $\alpha$ | Dual-Value Fusion 中语义价值的权重（默认 0.8） |

---

## 3. 公共模块：价值计算与 Top-k 选择

### 3.1 Dual-Value Fusion（价值计算）

Frontier 的总价值由**语义价值**和**信息增益**融合得到：

$$V_{total}(f_i) = \alpha \cdot V_{sem}(f_i) + (1-\alpha) \cdot V_{ig}(f_i)$$

其中：
- $V_{sem}(f_i) = \max_{p \in \mathcal{N}(f_i)} \text{HSVM}(p)$：邻域内最大语义价值
- $V_{ig}(f_i) = \max_{p \in \mathcal{N}(f_i)} \text{IG}(p)$：邻域内最大信息增益
- $\mathcal{N}(f_i)$：$f_i$ 的 5×5 栅格邻域

**代码位置**：`exploration_manager.cpp:409-438`

### 3.2 Top-k 选择

按价值降序排列，选取前 k 个 frontier：

$$\mathcal{F}_k = \{f_{(1)}, f_{(2)}, \ldots, f_{(k)}\} \quad \text{where} \quad V(f_{(1)}) \geq V(f_{(2)}) \geq \cdots \geq V(f_{(n)})$$

### 3.3 代价矩阵构建

使用 A* 算法计算**实际路径长度**（非欧氏距离）：

$$C_{ij} = \begin{cases}
    \text{PathLength}(\text{A}^*(p_i, p_j)) & \text{if } i \neq j \\
    \infty & \text{if } i = j
\end{cases}$$

其中 $p_0$ 是当前位置，$p_i$ ($i \geq 1$) 对应 frontier $f_{(i)}$。

**代码位置**：`exploration_manager.cpp:543-568`

---

## 4. Semantic-TSP 算法（HYBRID 模式）

### 4.1 目标函数

求解非对称旅行商问题（ATSP），最小化遍历所有 top-k frontier 的**总路径长度**：

$$\pi^* = \arg\min_{\pi \in \Pi_k} \sum_{i=0}^{k-1} C_{\pi(i), \pi(i+1)}$$

其中 $\Pi_k$ 是所有以 $\pi(0) = 0$（当前位置）开始的排列集合。

### 4.2 求解方法：LKH-3

使用 **Lin-Kernighan-Helsgaun** 启发式算法求解 ATSP：
1. 将代价矩阵写入 `.atsp` 文件
2. 调用 LKH-3 ROS 服务
3. 从 `.tour` 文件读取最优路径

**代码位置**：`exploration_manager.cpp:570-658`

### 4.3 算法流程

```
Algorithm: Semantic-TSP (HYBRID Mode)

Input: 当前位置 p₀, Frontier 集合 F, 参数 k
Output: 下一个目标 frontier

1. 计算所有 frontier 的价值 V(f)
2. 选取 top-k 高价值 frontier: F_k
3. 过滤可达 frontier: F_r ⊆ F_k
4. 构建代价矩阵 C (使用 A* 路径长度)
5. 调用 LKH-3 求解 ATSP: π* = LKH3(C)
6. 返回 TSP 路径的第一个节点: f* = F_r[π*(1)]
```

**代码位置**：`exploration_manager.cpp:381-399`

---

## 5. WTRP 算法（加权旅行修理工问题）

### 5.1 问题定义

WTRP (Weighted Traveling Repairman Problem) 最小化**加权到达时间**，而非总路径长度：

$$\pi^* = \arg\min_{\pi \in \Pi_k} \sum_{i=1}^{k} w_{\pi(i)} \cdot T_{\pi}(i)$$

其中累计到达时间为：

$$T_{\pi}(i) = \sum_{j=1}^{i} C_{\pi(j-1), \pi(j)}$$

### 5.2 权重计算：Softmax

$$w_i = \frac{\exp\left(\frac{V(f_i) - V_{\max}}{\tau}\right)}{\sum_{j=1}^{k} \exp\left(\frac{V(f_j) - V_{\max}}{\tau}\right)}$$

**温度参数 $\tau$ 的影响**：

| $\tau$ | 权重分布 | 行为特征 |
|--------|----------|----------|
| $\tau \to 0$ | 趋近 one-hot | 只关注最高价值 frontier |
| $\tau = 0.5$（默认） | 中等分化 | 平衡价值差异 |
| $\tau \to \infty$ | 趋近均匀 | 退化为标准 TRP（无权重） |

**代码位置**：`exploration_manager.cpp:980-1002`

### 5.3 求解方法

**WTRP 不使用 LKH-3**，而是采用以下策略：

#### 5.3.1 小规模（N ≤ 10）：暴力枚举

遍历所有 $N!$ 种排列，计算每种排列的加权到达时间，选择最优解：

```cpp
do {
    double total_cost = 0.0;
    double cumulative_time = 0.0;

    // 计算该排列的加权到达时间
    cumulative_time += cost_matrix(0, perm[0] + 1);
    total_cost += weights[perm[0]] * cumulative_time;

    for (int k = 1; k < N; ++k) {
        cumulative_time += cost_matrix(perm[k-1] + 1, perm[k] + 1);
        total_cost += weights[perm[k]] * cumulative_time;
    }

    if (total_cost < best_cost) {
        best_cost = total_cost;
        best_order = perm;
    }
} while (std::next_permutation(perm.begin(), perm.end()));
```

**代码位置**：`exploration_manager.cpp:1027-1047`

#### 5.3.2 大规模（N > 10）：贪心启发式

每步选择使**边际代价最小**的未访问节点：

$$j^* = \arg\min_{j \in \mathcal{U}} c(p_{cur}, p_j) \cdot W_{remaining}$$

其中 $W_{remaining} = \sum_{k \in \mathcal{U}} w_k$ 是剩余未访问节点的总权重。

**直观解释**：选择节点 $j$ 会使所有剩余节点的到达时间增加 $c(p_{cur}, p_j)$，因此边际代价为行程时间乘以剩余总权重。

**代码位置**：`exploration_manager.cpp:1049-1092`

### 5.4 WTRP 目标函数的数学展开

设访问顺序为 $\pi = (\pi_1, \pi_2, \ldots, \pi_n)$，定义边代价 $c_i = C_{\pi(i-1), \pi(i)}$：

$$\begin{aligned}
J(\pi) &= \sum_{i=1}^{n} w_{\pi_i} \cdot T_{\pi}(i) \\
&= w_{\pi_1} \cdot c_1 + w_{\pi_2} \cdot (c_1 + c_2) + w_{\pi_3} \cdot (c_1 + c_2 + c_3) + \cdots \\
&= c_1 \cdot \sum_{j=1}^{n} w_{\pi_j} + c_2 \cdot \sum_{j=2}^{n} w_{\pi_j} + c_3 \cdot \sum_{j=3}^{n} w_{\pi_j} + \cdots \\
&= \sum_{i=1}^{n} c_i \cdot W_i^{tail}
\end{aligned}$$

其中 $W_i^{tail} = \sum_{j=i}^{n} w_{\pi_j}$ 是从位置 $i$ 开始的**尾部权重和**。

**关键洞察**：每条边的贡献 = 边长度 × 所有后续节点的权重和。因此：
- 高权重节点应尽早访问（减少后续边的乘数）
- 但也要考虑到达该节点的代价

### 5.5 算法流程

```
Algorithm: WTRP for Frontier Exploration

Input: 当前位置 p₀, Frontier 集合 F, 参数 k, τ
Output: 下一个目标 frontier

1. 计算所有 frontier 的价值 V(f)
2. 选取 top-k 高价值 frontier: F_k
3. 过滤可达 frontier: F_r ⊆ F_k
4. 计算 softmax 权重: w_i = softmax(V(f_i) / τ)
5. 构建代价矩阵 C (使用 A* 路径长度)
6. 求解 WTRP:
   - if |F_r| ≤ 10: 暴力枚举所有排列
   - else: 贪心启发式
7. 返回 WTRP 最优路径的第一个节点
```

**代码位置**：`exploration_manager.cpp:1095-1172`

---

## 6. 算法对比

### 6.1 目标函数对比

| 策略 | 目标函数 | 含义 |
|------|----------|------|
| Greedy Closest | $\min_{f \in \mathcal{F}} c(p_0, f)$ | 最近的 frontier |
| Greedy Highest Value | $\max_{f \in \mathcal{F}} V(f)$ | 最高价值的 frontier |
| **Semantic-TSP** | $\min_{\pi \in \Pi_k} \sum_{i} C_{\pi(i), \pi(i+1)}$ | 遍历 top-k 的最短路径 |
| **WTRP** | $\min_{\pi \in \Pi_k} \sum_{i} w_i \cdot T_{\pi}(i)$ | 最小加权到达时间 |

### 6.2 特性对比

| 特性 | Greedy Closest | Greedy Value | Semantic-TSP | WTRP |
|------|----------------|--------------|--------------|------|
| 考虑距离 | ✓（局部） | ✗ | ✓（全局） | ✓（全局） |
| 考虑价值 | ✗ | ✓（局部） | ✓（筛选） | ✓（筛选+加权） |
| 全局路径规划 | ✗ | ✗ | ✓ | ✓ |
| 价值影响访问顺序 | ✗ | ✗ | ✗ | ✓ |
| 求解器 | - | - | LKH-3 | 暴力/贪心 |

### 6.3 复杂度对比

| 算法 | 时间复杂度 | 空间复杂度 | 最优性 |
|------|------------|------------|--------|
| Greedy Closest | $O(n \cdot A^*)$ | $O(1)$ | 局部最优 |
| Greedy Highest Value | $O(n \log n + n \cdot A^*)$ | $O(n)$ | 局部最优 |
| **Semantic-TSP** | $O(n \log n + k^2 \cdot A^* + \text{LKH}(k))$ | $O(k^2)$ | 近似最优 |
| **WTRP (暴力)** | $O(n \log n + k^2 \cdot A^* + k! \cdot k)$ | $O(k^2)$ | 全局最优 |
| **WTRP (贪心)** | $O(n \log n + k^2 \cdot A^* + k^2)$ | $O(k^2)$ | 近似最优 |

其中 $A^*$ 表示单次 A* 路径规划的复杂度，$k$ 为 top-k 参数（默认 5）。

### 6.4 Semantic-TSP vs WTRP 的选择

| 场景 | 推荐算法 | 原因 |
|------|----------|------|
| 价值差异不大 | Semantic-TSP | TSP 路径更短，效率更高 |
| 存在明显高价值目标 | WTRP | 高价值节点会被优先访问 |
| frontier 数量多 (k > 10) | Semantic-TSP | LKH-3 比贪心 WTRP 更稳定 |
| frontier 数量少 (k ≤ 10) | WTRP | 暴力枚举得到全局最优 |

---

## 7. 示例分析

### 场景设置

```
        f1 (V=0.9)
        |
        | d=2
        |
p0 ---- f2 (V=0.3) ---- f3 (V=0.6)
   d=1        d=1
```

### 各算法行为

**Greedy Closest**：
- 选择：$f_2$（距离最近）
- 路径：$p_0 \to f_2$
- 问题：忽略了 $f_1$ 的高价值

**Greedy Highest Value**：
- 选择：$f_1$（价值最高）
- 路径：$p_0 \to f_1$
- 问题：距离较远，可能不是全局最优

**Semantic-TSP (k=3)**：
- TSP 最优路径：$p_0 \to f_2 \to f_1 \to f_3$（总长度 = 1+2+√5 ≈ 5.24）
- 或：$p_0 \to f_2 \to f_3 \to f_1$（总长度 = 1+1+√5 ≈ 4.24）
- 选择：$f_2$（TSP 路径的第一个节点）

**WTRP (k=3, τ=0.5)**：
- 权重计算（假设）：$w_1 \approx 0.59, w_2 \approx 0.09, w_3 \approx 0.32$
- 评估各顺序的加权到达时间：
  - $\pi = (f_1, f_2, f_3)$: $J = 0.59 \times 2 + 0.09 \times 4 + 0.32 \times 5 = 3.14$
  - $\pi = (f_2, f_1, f_3)$: $J = 0.09 \times 1 + 0.59 \times 3 + 0.32 \times 5.24 = 3.54$
  - $\pi = (f_2, f_3, f_1)$: $J = 0.09 \times 1 + 0.32 \times 2 + 0.59 \times 4.24 = 3.23$
- 最优：$\pi = (f_1, f_2, f_3)$
- 选择：$f_1$（WTRP 优先访问高价值节点）

**结论**：WTRP 会优先选择高价值的 $f_1$，而 Semantic-TSP 会选择路径更短的起点 $f_2$。

---

## 8. 参数配置

### 8.1 默认参数

| 参数 | 默认值 | 配置位置 | 说明 |
|------|--------|----------|------|
| `policy` | 4 (WTRP) | `algorithm.xml:59` | 探索策略模式 |
| `top_k_value` | 5 | `algorithm.xml:60` | Top-k 选择数量 |
| `fusion_alpha` | 0.8 | `algorithm.xml:63` | 语义价值权重 |
| `use_ig_fusion` | true | `algorithm.xml:64` | 是否融合信息增益 |
| `wtrp_temperature` | 0.5 | `algorithm.xml:66` | Softmax 温度 |
| `wtrp_max_brute_force` | 10 | `algorithm.xml:67` | 暴力枚举阈值 |

### 8.2 参数调优建议

| 参数 | 调小效果 | 调大效果 |
|------|----------|----------|
| `top_k_value` | 更关注高价值，计算更快 | 考虑更多候选，但可能引入低价值 frontier |
| `fusion_alpha` | 更依赖信息增益（探索未知区域） | 更依赖语义价值（导向目标） |
| `wtrp_temperature` | 权重更集中于高价值（激进） | 权重更均匀（保守） |
| `wtrp_max_brute_force` | 更快但可能非最优 | 更慢但保证小规模最优 |

---

## 9. 代码结构

### 9.1 核心文件

```
InfoNav/src/planner/exploration_manager/
├── src/
│   └── exploration_manager.cpp    # 主要算法实现
├── include/exploration_manager/
│   ├── exploration_manager.h      # 类定义
│   └── exploration_data.h         # 数据结构定义
└── launch/
    └── algorithm.xml              # 参数配置

InfoNav/src/planner/utils/lkh_mtsp_solver/
├── src2/
│   ├── tsp_node.cpp               # TSP ROS 服务节点
│   └── lkh3_interface.cpp         # LKH-3 求解器接口
└── src/
    └── ...                        # LKH-3 算法实现
```

### 9.2 关键函数索引

| 函数 | 代码行 | 功能 |
|------|--------|------|
| `chooseExplorationPolicy` | L347-379 | 策略选择入口 |
| `hybridExplorePolicy` | L381-399 | Semantic-TSP 实现 |
| `wtrpExplorePolicy` | L1095-1172 | WTRP 实现 |
| `getSortedSemanticFrontiers` | L869-935 | Dual-Value Fusion + 排序 |
| `computeSoftmaxWeights` | L980-1002 | Softmax 权重计算 |
| `solveWTRP` | L1004-1093 | WTRP 求解（暴力/贪心） |
| `computeATSPTour` | L570-658 | ATSP 求解（调用 LKH-3） |
| `computeATSPCostMatrix` | L543-568 | 代价矩阵构建 |
| `findTSPTourPolicy` | L509-533 | TSP 路径策略 |

---

## 10. 算法伪代码（论文用）

### Algorithm 1: Semantic-aware TSP

```
Input: Position p₀, Frontiers F, Value function V(·), Parameter k
Output: Next frontier f*

1:  for each f ∈ F do
2:      V(f) ← α·V_sem(f) + (1-α)·V_ig(f)    // Dual-Value Fusion
3:  end for
4:  F_k ← TopK(F, k, by=V)                    // Top-k selection
5:  F_r ← {f ∈ F_k : A*(p₀,f) ≠ ∅}           // Reachability filter
6:  C ← BuildCostMatrix(p₀, F_r)              // A* path lengths
7:  π* ← LKH3(C)                              // Solve ATSP
8:  return F_r[π*(1)]
```

### Algorithm 2: WTRP

```
Input: Position p₀, Frontiers F, Value function V(·), Parameters k, τ
Output: Next frontier f*

1:  for each f ∈ F do
2:      V(f) ← α·V_sem(f) + (1-α)·V_ig(f)    // Dual-Value Fusion
3:  end for
4:  F_k ← TopK(F, k, by=V)                    // Top-k selection
5:  F_r ← {f ∈ F_k : A*(p₀,f) ≠ ∅}           // Reachability filter
6:  w ← Softmax(V(F_r) / τ)                   // Compute weights
7:  C ← BuildCostMatrix(p₀, F_r)              // A* path lengths
8:  if |F_r| ≤ N_bf then
9:      π* ← BruteForceWTRP(C, w)             // Exact solution
10: else
11:     π* ← GreedyWTRP(C, w)                 // Heuristic solution
12: end if
13: return F_r[π*(1)]
```

### Algorithm 3: WTRP Solver (Brute-force)

```
Input: Cost matrix C, Weights w
Output: Optimal order π*

1:  π* ← null, J* ← ∞
2:  for each permutation π of {1,...,n} do
3:      T ← 0, J ← 0
4:      for i = 1 to n do
5:          T ← T + C[π(i-1), π(i)]           // Cumulative time
6:          J ← J + w[π(i)] · T               // Weighted arrival
7:      end for
8:      if J < J* then J* ← J, π* ← π
9:  end for
10: return π*
```

### Algorithm 4: WTRP Solver (Greedy)

```
Input: Cost matrix C, Weights w
Output: Approximate order π

1:  U ← {1,...,n}, π ← [], cur ← 0
2:  while U ≠ ∅ do
3:      W_rem ← Σ_{j∈U} w[j]                  // Remaining weight
4:      j* ← argmin_{j∈U} C[cur,j] · W_rem    // Min marginal cost
5:      π.append(j*), U ← U\{j*}, cur ← j*
6:  end while
7:  return π
```

---

## 11. LaTeX 公式汇总

```latex
% Dual-Value Fusion
V_{total}(f_i) = \alpha \cdot V_{sem}(f_i) + (1-\alpha) \cdot V_{ig}(f_i)

% Softmax Weights
w_i = \frac{\exp\left(\frac{V(f_i) - V_{\max}}{\tau}\right)}{\sum_{j=1}^{k} \exp\left(\frac{V(f_j) - V_{\max}}{\tau}\right)}

% Semantic-TSP Objective (uses LKH-3)
\pi^* = \arg\min_{\pi \in \Pi_k} \sum_{i=0}^{k-1} C_{\pi(i), \pi(i+1)}

% WTRP Objective (uses brute-force or greedy)
\pi^* = \arg\min_{\pi \in \Pi_k} \sum_{i=1}^{k} w_{\pi(i)} \cdot T_{\pi}(i)

% Cumulative Arrival Time
T_{\pi}(i) = \sum_{j=1}^{i} C_{\pi(j-1), \pi(j)}

% WTRP Objective Expansion
J(\pi) = \sum_{i=1}^{n} c_i \cdot W_i^{tail}, \quad W_i^{tail} = \sum_{j=i}^{n} w_{\pi_j}

% Greedy WTRP Marginal Cost
j^* = \arg\min_{j \in \mathcal{U}} c(p_{cur}, p_j) \cdot W_{remaining}
```

---

## 12. 参考文献

1. **LKH-3**: Helsgaun, K. (2017). An extension of the Lin-Kernighan-Helsgaun TSP solver for constrained traveling salesman and vehicle routing problems.

2. **Traveling Repairman Problem**: Blum, A., et al. (1994). Minimum latency problem.

3. **Weighted TRP**: Chakrabarty, D., & Swamy, C. (2011). Facility location with service installation costs.

4. **Frontier-based Exploration**: Yamauchi, B. (1997). A frontier-based approach for autonomous exploration.

---

*Document generated for InfoNav project. Last updated: 2024.*
