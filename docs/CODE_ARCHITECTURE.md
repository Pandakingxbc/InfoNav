# InfoNav 代码架构文档

> 本文档梳理 InfoNav 项目的代码逻辑，方便后续代码修改和维护。

---

## 一、整体架构概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                    habitat_evaluation.py (Python)                    │
│                     Habitat 仿真环境 + 评估主循环                      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ ROS 消息通信
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 exploration_manager (C++ ROS Node)                   │
│              ExplorationFSM + ExplorationManager                     │
│                   导航决策 + 前沿探索 + 路径规划                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 核心组件

| 组件 | 语言 | 职责 |
|------|------|------|
| `habitat_evaluation.py` | Python | Habitat 仿真、VLM 推理、评估指标 |
| `ExplorationFSM` | C++ | 状态机管理、动作规划 |
| `ExplorationManager` | C++ | 前沿探索、路径搜索、策略选择 |
| `FrontierMap2D` | C++ | 前沿点检测与管理 |
| `ObjectMap2D` | C++ | 检测物体的位置管理 |
| `SDFMap2D` | C++ | 障碍物栅格地图 (SDF) |
| `ValueMap2D` | C++ | 语义价值地图 |

---

## 二、habitat_evaluation.py 详解

### 2.1 文件位置
```
/home/yangz/Nav/InfoNav/habitat_evaluation.py (821 行)
```

### 2.2 初始化阶段 (L214-354)

```python
main()
  ├── 加载配置 (Hydra: hm3dv1/hm3dv2/mp3d)
  ├── 创建 Habitat 环境: env = habitat.Env(cfg)
  ├── 初始化 ROS 节点和发布/订阅器
  └── 从上次中断处继续 (read_record)
```

#### ROS 话题定义

| 话题名称 | 类型 | 方向 | 说明 |
|---------|------|------|------|
| `/habitat/object_point_cloud` | PointCloud2 | 发布 | 检测到的物体点云 |
| `/habitat/state` | Int32 | 发布 | Habitat 状态码 |
| `/habitat/semantic_scores` | SemanticScores | 发布 | 多源语义得分 |
| `/habitat/episode_reset` | Empty | 发布 | Episode 重置信号 |
| `/blip2/cosine_score` | Float64 | 发布 | 融合后的 ITM 余弦相似度 |
| `/detector/clouds_with_scores` | MultipleMasksWithConfidence | 发布 | 检测点云+置信度 |
| `/habitat/plan_action` | Int32 | 订阅 | 接收动作指令 |
| `/ros/state` | Int32 | 订阅 | 接收 ROS FSM 状态 |
| `/ros/expl_result` | Int32 | 订阅 | 接收探索结果 |

### 2.3 Episode 循环主流程 (L355-801)

```python
for epi in episodes:
    # === 1. Episode 初始化 ===
    env.reset()                          # 重置 Habitat 环境
    episode_reset_pub.publish(Empty())   # 通知 ROS 清空地图

    # === 2. 获取任务信息 ===
    label = env.current_episode.object_category  # 目标物体类别
    llm_answer, room, fusion_threshold = read_answer(...)  # LLM 答案
    current_hypotheses = load_llm_hypotheses(...)  # 加载语义假说

    # === 3. 等待 ROS 就绪 ===
    while ros_state == ROS_STATE.INIT or ros_state == ROS_STATE.WAIT_TRIGGER:
        rate.sleep()

    # === 4. 步骤循环 ===
    while not episode_over:
        # 4.1 解析动作
        if global_action == ACTION.MOVE_FORWARD:
            action = HabitatSimActions.move_forward
        elif global_action == ACTION.TURN_LEFT:
            action = HabitatSimActions.turn_left
        # ... 其他动作映射

        # 4.2 执行动作
        publish_int32(state_pub, HABITAT_STATE.ACTION_EXEC)
        observations = env.step(action)

        # 4.3 VLM 推理 - 计算多源语义得分
        hypotheses_data = get_multi_source_cosine(
            observations["rgb"], label, room, current_hypotheses
        )
        semantic_scores_pub.publish(semantic_scores_msg)

        # 4.4 物体检测
        rgb, score_list, object_masks_list, label_list = get_object(
            label, observations["rgb"], detector_cfg, llm_answer
        )

        # 4.5 生成并发布点云
        obj_point_cloud_list = get_object_point_cloud(cfg, observations, object_masks_list)
        cld_with_score_pub.publish(cld_with_score_msg)

        # 4.6 通知动作完成
        publish_int32(state_pub, HABITAT_STATE.ACTION_FINISH)
        rate.sleep()

    # === 5. Episode 结束 ===
    publish_int32(state_pub, HABITAT_STATE.EPISODE_FINISH)
    # 记录评估指标 (Success, SPL, Soft SPL, Distance to Goal)
```

### 2.4 关键函数说明

| 函数 | 位置 | 功能 |
|------|------|------|
| `load_llm_hypotheses()` | L123-151 | 加载 LLM 语义假说 JSON 文件 |
| `publish_observations()` | L166-174 | 定时发布 Habitat 观测数据 |
| `ros_action_callback()` | L177-179 | 接收 ROS 发布的动作 |
| `ros_state_callback()` | L182-184 | 接收 ROS FSM 状态 |

---

## 三、状态机通信协议

### 3.1 状态码定义 (params.py)

```python
class HABITAT_STATE:
    READY = 0           # 准备就绪
    ACTION_EXEC = 1     # 动作执行中
    ACTION_FINISH = 2   # 动作完成
    EPISODE_FINISH = 3  # Episode 结束

class ROS_STATE:
    INIT = 0            # 初始化
    WAIT_TRIGGER = 1    # 等待触发
    PLAN_ACTION = 2     # 规划动作
    WAIT_ACTION_FINISH = 3  # 等待动作完成
    PUB_ACTION = 4      # 发布动作
    FINISH = 5          # 完成

class ACTION:
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    TURN_DOWN = 4
    TURN_UP = 5
```

### 3.2 探索结果类型

```python
class EXPL_RESULT:
    EXPLORATION = 0              # 正常探索
    SEARCH_BEST_OBJECT = 1       # 找到高置信度物体
    SEARCH_OVER_DEPTH_OBJECT = 2 # 搜索过深物体
    SEARCH_SUSPICIOUS_OBJECT = 3 # 调查可疑物体
    NO_PASSABLE_FRONTIER = 4     # 无可通过前沿
    NO_COVERABLE_FRONTIER = 5    # 无可覆盖前沿
    SEARCH_EXTREME = 6           # 极端搜索模式
```

---

## 四、exploration_manager 详解

### 4.1 目录结构

```
src/planner/exploration_manager/
├── include/exploration_manager/
│   ├── exploration_manager.h    # 规划管理器头文件
│   ├── exploration_fsm.h        # FSM 状态机头文件
│   ├── exploration_data.h       # 数据结构定义
│   └── backward.hpp
├── src/
│   ├── exploration_node.cpp     # ROS 节点入口 (414 行)
│   ├── exploration_fsm.cpp      # FSM 实现 (33KB)
│   └── exploration_manager.cpp  # 规划器实现 (23KB)
├── config/
│   └── algorithm.xml            # 算法配置
└── launch/
    └── exploration.launch       # 启动文件
```

### 4.2 类结构关系

```
exploration_node.cpp (入口)
       │
       ▼
┌─────────────────────────────────────────────────────┐
│              ExplorationFSM (状态机)                 │
│  ├── FSMCallback()        10ms 定时器              │
│  ├── frontierCallback()   1s 定时器 (前沿更新)      │
│  ├── odometryCallback()   接收 Habitat 里程计       │
│  └── habitatStateCallback()  接收 Habitat 状态      │
└────────────────────┬────────────────────────────────┘
                     │ 调用
                     ▼
┌─────────────────────────────────────────────────────┐
│            ExplorationManager (规划器)               │
│  ├── planNextBestPoint()      主规划入口            │
│  ├── chooseExplorationPolicy()  策略选择            │
│  └── searchObjectPath() / searchFrontierPath()     │
└────────────────────┬────────────────────────────────┘
                     │ 依赖
    ┌────────────────┼────────────────┐
    ▼                ▼                ▼
FrontierMap2D   ObjectMap2D     SDFMap2D + ValueMap2D
```

### 4.3 FSM 状态转移图

```
        ┌──────────────────────────────────────────────┐
        │                                              │
        ▼                                              │
     ┌──────┐     收到触发      ┌─────────────┐        │
     │ INIT │ ────────────────→ │ WAIT_TRIGGER│        │
     └──────┘                   └──────┬──────┘        │
        ▲                              │               │
        │                         收到里程计            │
        │                              ▼               │
        │                      ┌─────────────┐         │
        │ Episode 重置         │ PLAN_ACTION │←────────┤
        │                      └──────┬──────┘         │
        │                             │                │
        │                      规划动作                │
        │                             ▼                │
        │                      ┌─────────────┐         │
        │                      │ PUB_ACTION  │         │
        │                      └──────┬──────┘         │
        │                             │                │
        │                      发布 action             │
        │                             ▼                │
        │                  ┌────────────────────┐      │
        │                  │ WAIT_ACTION_FINISH │      │
        │                  └─────────┬──────────┘      │
        │                            │                 │
        │              Habitat 返回 ACTION_FINISH      │
        │                            │                 │
        └────────────────────────────┴─────────────────┘
```

### 4.4 核心常量 (FSMConstants)

```cpp
namespace FSMConstants {
    // 定时器
    constexpr double EXEC_TIMER_DURATION = 0.01;      // 10ms
    constexpr double FRONTIER_TIMER_DURATION = 1.0;   // 1s

    // 动作参数
    constexpr double ACTION_DISTANCE = 0.25;          // 前进距离
    constexpr double ACTION_ANGLE = M_PI / 6.0;       // 转向角度 (30°)

    // 距离阈值
    constexpr double STUCKING_DISTANCE = 0.05;        // 卡住判定距离
    constexpr double REACH_DISTANCE = 0.20;           // 到达物体距离
    constexpr double LOCAL_DISTANCE = 0.80;           // 局部目标距离
    constexpr double MIN_SAFE_DISTANCE = 0.15;        // 最小安全距离

    // 计数器阈值
    constexpr int MAX_STUCKING_COUNT = 25;            // 最大卡住次数
    constexpr int MAX_STUCKING_NEXT_POS_COUNT = 14;   // 同目标卡住上限

    // 无进展检测阈值 (用于检测无法到达的 Frontier)
    // 智能体可能在移动但由于深度缺陷或离散动作无法接近 Frontier
    constexpr int NO_PROGRESS_MAX_STEPS = 50;         // 向同一 Frontier 的最大步数
    constexpr double NO_PROGRESS_MIN_IMPROVEMENT = 0.3;  // 要求的最小距离改善 (米)
    constexpr double NO_PROGRESS_CHECK_INTERVAL = 15;    // 每 N 步检查一次进度
    constexpr double FRONTIER_CHANGE_THRESHOLD = 0.5;    // Frontier 变化判定阈值 (米)
}
```

### 4.5 核心规划算法

#### planNextBestPoint() 流程

```cpp
int ExplorationManager::planNextBestPoint(pos, yaw)
{
    // 1. 获取前沿点
    frontiers = frontier_map2d_->getFrontiers();

    // 2. 检查高置信度物体
    if (有置信度 > threshold 的物体) {
        searchObjectPath(pos, object_cloud);
        return SEARCH_BEST_OBJECT;
    }

    // 3. 检查可疑物体
    if (有可疑物体) {
        searchObjectPath();
        return SEARCH_SUSPICIOUS_OBJECT;
    }

    // 4. 前沿探索
    if (frontiers.empty()) {
        return NO_PASSABLE_FRONTIER;
    }

    // 5. 策略选择
    chooseExplorationPolicy(cur_pos, frontiers, next_pos, path);

    return EXPLORATION;
}
```

#### 探索策略 (POLICY_MODE)

| 策略 | 枚举值 | 函数 | 说明 |
|------|--------|------|------|
| DISTANCE | 0 | `findClosestFrontierPolicy()` | 选择路径最短的前沿 |
| SEMANTIC | 1 | `findHighestSemanticsFrontierPolicy()` | 选择语义价值最高的前沿 |
| HYBRID | 2 | `hybridExplorePolicy()` | 根据语义方差动态切换 |
| TSP_DIST | 3 | `findTSPTourPolicy()` | TSP 优化访问顺序 |

#### 动作规划 (callActionPlanner)

```cpp
int ExplorationFSM::callActionPlanner()
{
    // 1. 获取规划路径
    path = ed_->next_best_path_;

    // 2. 选择局部目标点
    local_target = selectLocalTarget(current_pos, path, LOCAL_DISTANCE);

    // 3. 计算目标角度
    target_yaw = atan2(target.y - pos.y, target.x - pos.x);

    // 4. 决定动作
    if (角度差 > 阈值)
        return TURN_LEFT / TURN_RIGHT;
    else if (距离 > 阈值)
        return MOVE_FORWARD;
    else
        return STOP;
}
```

### 4.6 数据结构

#### FSMData (状态机数据)

```cpp
struct FSMData {
    // 状态标志
    bool trigger_, have_odom_, have_confidence_, have_finished_;

    // 里程计
    Vector3d odom_pos_;
    Quaterniond odom_orient_;
    double odom_yaw_;

    // 卡住检测
    int stucking_action_count_;
    int stucking_next_pos_count_;
    vector<Vector3d> stucking_points_;

    // Patience-Aware Navigation
    int step_count_;
    std::string target_category_;

    // 无进展检测 (Frontier 探索)
    Vector2d tracked_frontier_pos_;      // 当前追踪的 Frontier 位置
    double initial_dist_to_frontier_;    // 开始追踪时的距离
    int steps_toward_frontier_;          // 向当前 Frontier 的步数
    double best_dist_to_frontier_;       // 达到过的最小距离
};
```

#### ExplorationData (探索数据)

```cpp
struct ExplorationData {
    vector<vector<Vector2d>> frontiers_;          // 前沿簇
    vector<Vector2d> frontier_averages_;          // 前沿中心点
    vector<vector<Vector2d>> dormant_frontiers_;  // 休眠前沿

    vector<vector<Vector2d>> objects_;            // 检测到的物体
    vector<Vector2d> object_averages_;            // 物体中心点
    vector<int> object_labels_;                   // 物体标签

    Vector2d next_pos_;                           // 下一个目标点
    vector<Vector2d> next_best_path_;             // 最优路径
    vector<Vector2d> tsp_tour_;                   // TSP 路径
};
```

#### SemanticFrontier (语义前沿)

```cpp
struct SemanticFrontier {
    Vector2d position;      // 2D 位置
    double semantic_value;  // 语义价值
    double path_length;     // 路径长度
    vector<Vector2d> path;  // 完整路径

    // 排序: 语义价值降序，相同则路径长度升序
    bool operator<(const SemanticFrontier& other) const;
};
```

---

## 五、数据流总结

```
┌─────────────────────────────────────────────────────────────────────┐
│                         数据流向图                                   │
└─────────────────────────────────────────────────────────────────────┘

Habitat Simulator
       │
       │ RGB + Depth + Odometry
       ▼
┌──────────────────┐     RGB      ┌──────────────────┐
│ habitat_evalua-  │ ──────────→  │ VLM 模块         │
│ tion.py          │              │ - get_object()   │
│                  │              │ - get_multi_     │
│                  │ ◄────────── │   source_cosine()│
│                  │ 检测结果     └──────────────────┘
└────────┬─────────┘
         │
         │ ROS Topics
         ▼
┌──────────────────────────────────────────────────────┐
│            exploration_manager (C++)                  │
│                                                       │
│  SDFMap2D ← Depth → 障碍物栅格                        │
│  ValueMap2D ← Semantic Scores → 语义价值地图          │
│  FrontierMap2D → 前沿点检测                           │
│  ObjectMap2D ← Point Clouds → 物体位置                │
│                                                       │
│  ExplorationManager.planNextBestPoint()               │
│           ↓                                           │
│  ExplorationFSM.callActionPlanner()                   │
│           ↓                                           │
│       action (0-5)                                    │
└──────────────────────┬───────────────────────────────┘
                       │
                       │ /habitat/plan_action
                       ▼
              habitat_evaluation.py
                       │
                       │ env.step(action)
                       ▼
                 Habitat Simulator
```

---

## 六、关键文件索引

### Python 模块

| 功能 | 文件路径 |
|------|---------|
| 评估主程序 | `habitat_evaluation.py` |
| 状态/动作定义 | `params.py` |
| VLM 语义融合 | `vlm/utils/get_itm_message.py` |
| VLM 物体检测 | `vlm/utils/get_object_utils.py` |
| LLM 答案读取 | `llm/answer_reader/answer_reader.py` |
| LLM 假说加载 | `llm/prompt/value_map_hypothesis.py` |
| 点云生成 | `basic_utils/object_point_cloud_utils/object_point_cloud.py` |
| 失败检测 | `basic_utils/failure_check/failure_check.py` |
| Habitat→ROS 发布 | `habitat2ros/habitat_publisher.py` |

### C++ 模块

| 功能 | 文件路径 |
|------|---------|
| FSM 头文件 | `src/planner/exploration_manager/include/exploration_manager/exploration_fsm.h` |
| FSM 实现 | `src/planner/exploration_manager/src/exploration_fsm.cpp` |
| 规划器头文件 | `src/planner/exploration_manager/include/exploration_manager/exploration_manager.h` |
| 规划器实现 | `src/planner/exploration_manager/src/exploration_manager.cpp` |
| 数据结构 | `src/planner/exploration_manager/include/exploration_manager/exploration_data.h` |
| A* 路径搜索 | `src/planner/path_searching/include/path_searching/astar2d.h` |
| 前沿地图 | `src/planner/plan_env/include/plan_env/frontier_map2d.h` |
| 物体地图 | `src/planner/plan_env/include/plan_env/object_map2d.h` |
| SDF 地图 | `src/planner/plan_env/include/plan_env/sdf_map2d.h` |
| 价值地图 | `src/planner/plan_env/include/plan_env/value_map2d.h` |

### 配置文件

| 功能 | 文件路径 |
|------|---------|
| HM3D-v1 配置 | `config/habitat_eval_hm3dv1.yaml` |
| HM3D-v2 配置 | `config/habitat_eval_hm3dv2.yaml` |
| MP3D 配置 | `config/habitat_eval_mp3d.yaml` |
| 日志配置 | `config/logging_config.yaml` |
| 算法配置 | `src/planner/exploration_manager/config/algorithm.xml` |
| 启动文件 | `src/planner/exploration_manager/launch/exploration.launch` |

---

## 七、修改指南

### 7.1 添加新的 VLM 检测器

1. 在 `vlm/detector/` 下创建新检测器文件
2. 修改 `vlm/utils/get_object_utils.py` 中的 `get_object()` 函数
3. 在配置文件中添加检测器配置项

### 7.2 修改探索策略

1. 在 `ExplorationParam::POLICY_MODE` 枚举中添加新策略
2. 在 `exploration_manager.cpp` 中实现新策略函数
3. 在 `chooseExplorationPolicy()` 中添加策略分支

### 7.3 添加新的 ROS 话题

1. **Python 端**: 在 `habitat_evaluation.py` 中添加 Publisher/Subscriber
2. **C++ 端**: 在 `exploration_fsm.h` 中声明，在 `exploration_fsm.cpp` 的 `init()` 中初始化
3. 如果是自定义消息类型，在 `plan_env/msg/` 下定义

### 7.4 调整动作参数

修改 `exploration_fsm.h` 中的 `FSMConstants` 命名空间:
- `ACTION_DISTANCE`: 前进步长
- `ACTION_ANGLE`: 转向角度
- `STUCKING_DISTANCE`: 卡住判定阈值
- `MAX_STUCKING_COUNT`: 最大卡住次数

### 7.5 调试技巧

```bash
# 查看 ROS 话题
rostopic list
rostopic echo /habitat/plan_action
rostopic echo /ros/state

# 查看节点信息
rosnode info /exploration_node

# 启动 RViz 可视化
roslaunch exploration_manager rviz.launch
```

---

## 八、项目统计

| 类别 | 数量 |
|------|------|
| Python 顶级脚本 | 5 |
| VLM 模块文件 | 20 |
| LLM 模块文件 | 14 |
| exploration_manager 头文件 | 27 |
| exploration_manager 源文件 | 166 |
| 配置文件 | 4 |

---

## 九、问题诊断：目标导航中途切换到 Frontier 探索

### 9.1 问题描述

**现象**: 智能体在地图上已经显示了目标物体（红色标记，高置信度），并且正在向目标导航，但是在途中会突然切换到 Frontier 探索模式，尽管目标物体仍然显示为红色（高置信度）。

### 9.2 日志证据分析

从运行日志中可以清晰看到问题的根本原因：

```
[WARN] [Navigation Mode] Get object_cloud num = 1        ← 检测到1个高置信度物体
[ERROR] Failed to find object path.                       ← 路径规划失败！
[WARN] [Exploration Mode] TSP with top-5 value frontiers  ← 降级到Frontier探索
[WARN] To the next point (-7.55m -1.57m), distance = 5.33 m
```

关键观察：
1. `Get object_cloud num = 1` - 系统确实检测到了高置信度物体
2. `Failed to find object path` - **A\*路径搜索失败**，无法找到到达物体的路径
3. 系统自动降级到 Frontier 探索模式

### 9.3 根本原因：路径规划失败 (非置信度问题)

问题**不是**置信度下降导致的，而是 **A\*路径搜索失败**。

#### 决策流程 (`exploration_manager.cpp` 第86-138行)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    planNextBestPoint() 决策流程                       │
└─────────────────────────────────────────────────────────────────────┘

Step 1: getTopConfidenceObjectCloud(object_clouds)
        ↓
        object_clouds.size() == 1  ✓ 有高置信度物体
        ↓
Step 2: searchObjectPath(pos, object_cloud)
        ↓
        A* 路径搜索
        ↓
        ┌─────────────────┐
        │ 路径搜索结果?    │
        └────────┬────────┘
                 │
     ┌───────────┴───────────┐
     ▼                       ▼
  成功 (path非空)         失败 (path为空)
     │                       │
     ▼                       ▼
  return                 继续检查其他选项
  SEARCH_BEST_OBJECT     (Over Depth → Suspicious → Frontier)
```

#### 代码逻辑 (`exploration_manager.cpp`)

```cpp
// 第86-103行: 高置信度物体处理
sdf_map_->object_map2d_->getTopConfidenceObjectCloud(object_clouds);
ROS_WARN("[Navigation Mode] Get object_cloud num = %ld", object_clouds.size());

if (!object_clouds.empty()) {
    // 尝试搜索到物体的路径
    searchObjectPath(fd_->start_pt_, object_clouds[0], next_best_path);

    if (!next_best_path.empty()) {
        // 路径搜索成功 → 导航到物体
        return SEARCH_BEST_OBJECT;
    }
    // 路径搜索失败 → 继续尝试其他选项...
}

// 第138-207行: 降级到Frontier探索
if (路径仍然为空) {
    // 尝试 Over Depth 物体
    // 尝试 Suspicious 物体
    // 最终降级到 Frontier 探索
    searchFrontierPath(...);
    return EXPLORATION;
}
```

### 9.4 路径规划失败的可能原因

#### 原因1: 物体位置在障碍物内或靠近障碍物

```
┌──────────────────────────────┐
│          SDF Map             │
│                              │
│    ████████████████          │
│    █  目标物体  █            │
│    ████████████████          │
│         ▲                    │
│         │ A* 无法到达        │
│    智能体位置                │
└──────────────────────────────┘
```

物体点云可能落在了 SDF 地图的障碍物区域或安全边界内。

#### 原因2: 物体距离超出搜索范围

日志中出现的：
```
[ERROR] Have all over depth object cloud!!!!
```
表明所有物体都超出了正常搜索深度，系统尝试使用 `Over Depth` 模式但仍然失败。

#### 原因3: 路径被障碍物完全阻挡

```
┌──────────────────────────────┐
│                              │
│   目标物体 ●                 │
│                              │
│   ██████████████████████     │  ← 障碍物墙壁
│                              │
│            ○ 智能体          │
│                              │
└──────────────────────────────┘
```

### 9.5 日志中的完整切换模式

```
时间线:
┌────────────────────────────────────────────────────────────────────────┐
│ Step 136-138: 路径规划失败，切换到Frontier探索                          │
│   [Navigation Mode] Get object_cloud num = 1                           │
│   [ERROR] Failed to find object path.                                  │
│   [Exploration Mode] TSP with top-5 value frontiers                    │
│   To the next point (-4.95m -2.92m)                                    │
├────────────────────────────────────────────────────────────────────────┤
│ Step 138-145: 路径规划成功，导航到物体                                   │
│   [Navigation Mode] Get object_cloud num = 1                           │
│   I'm going to the object! dist = 0.70m!                               │
│   To the next point (1.83m 2.42m), distance = 5.68 m                   │
├────────────────────────────────────────────────────────────────────────┤
│ Step 145-154: 路径规划再次失败，切换到Frontier探索                       │
│   [Navigation Mode] Get object_cloud num = 1                           │
│   [ERROR] Failed to find object path.                                  │
│   [Navigation Mode (Over Depth)] Get over depth object cloud           │
│   [ERROR] Failed to find object path.                                  │
│   [Exploration Mode] TSP with top-5 value frontiers                    │
│   To the next point (-7.55m -1.57m)                                    │
├────────────────────────────────────────────────────────────────────────┤
│ Step 154-157: 路径规划成功，又导航回物体                                 │
│   [Navigation Mode] Get object_cloud num = 1                           │
│   I'm going to the object! dist = 0.70m!                               │
│   To the next point (1.83m 2.42m)                                      │
└────────────────────────────────────────────────────────────────────────┘
```

**关键发现**: 系统在物体导航和Frontier探索之间**振荡**，这是因为：
1. 智能体位置变化导致 A* 路径有时可达、有时不可达
2. SDF 地图更新可能导致障碍物状态变化

### 9.6 相关代码位置

| 功能 | 文件 | 行号 |
|------|------|------|
| 目标选择决策 | `exploration_manager.cpp` | 76-223 |
| A* 路径搜索 | `exploration_manager.cpp` | `searchObjectPath()` |
| 高置信度物体获取 | `object_map2d.cpp` | 607-703 |
| 置信度阈值计算 | `object_map2d.cpp` | 799-809 |
| FSM 状态转换 | `exploration_fsm.cpp` | 283-312 |

### 9.7 路径搜索机制深入分析

#### 当前路径搜索流程

```
searchObjectPath() [exploration_manager.cpp:592-619]
        │
        ▼
findNearestObjectPoint()  ← 找到物体点云中距离智能体最近的点
        │
        ▼
trySearchObjectPathWithDistance() ← 依次尝试 0.5m, 0.7m, 0.85m 距离
        │
        ├─► astarSearch(start, object_pose, distance=0.5m) ← 第一次尝试
        │       │
        │       ▼ 失败
        ├─► astarSearch(start, object_pose, distance=0.7m) ← 第二次尝试
        │       │
        │       ▼ 失败
        └─► astarSearch(start, object_pose, distance=0.85m) ← 第三次尝试
                │
                ▼ 失败
        return false → "Failed to find object path."
```

#### A* 安全检查机制 (`astar2d.cpp:252-273`)

```cpp
bool Astar2D::checkPointSafety(const Eigen::Vector2d& pos, int safety_mode)
{
  // 1. 检查是否在地图内
  if (!sdf_map_->isInMap(pos))
    return false;

  // 2. EXTREME模式：跳过所有障碍物检查
  if (safety_mode == SAFETY_MODE::EXTREME)
    return true;

  // 3. 检查膨胀后的占用状态 ← 关键！
  if (sdf_map_->getInflateOccupancy(pos) == 1 || occ == SDFMap2D::OCCUPIED)
    return false;  // 不安全

  // 4. NORMAL模式下，UNKNOWN也被视为不安全
  if (occ == SDFMap2D::UNKNOWN && safety_mode == SAFETY_MODE::NORMAL)
    return false;

  return true;
}
```

#### 障碍物膨胀参数

当前配置 (`algorithm.xml:26`):
```xml
<param name="sdf_map/obstacles_inflation" value="0.18" />
```

**膨胀计算**:
- 分辨率: `resolution = 0.05m`
- 膨胀半径: `obstacles_inflation = 0.18m`
- 膨胀步数: `inf_step = ceil(0.18 / 0.05) = 4` 格

这意味着**每个障碍物会向外扩展 4 格 (约 0.18m)**，导致：
- 狭窄通道可能被完全封死
- 物体如果靠近墙壁，其周围可达区域被大幅压缩

### 9.8 解决方案

#### 方案1: 减小障碍物膨胀半径 (最直接)

修改 `algorithm.xml`:
```xml
<!-- 原值: 0.18 → 建议减小到 0.12-0.15 -->
<param name="sdf_map/obstacles_inflation" value="0.12" />
```

**影响评估**:
- 优点: 增加可通行区域，物体更容易到达
- 缺点: 智能体可能更接近障碍物，碰撞风险略增
- 建议值: `0.12m` (Habitat agent_radius 通常是 0.1m)

#### 方案2: 增加更大的停靠距离选项

修改 `exploration_manager.cpp:605`:
```cpp
// 原始配置
const std::vector<double> distances = { 0.5, 0.70, 0.85 };

// 增加更大距离选项
const std::vector<double> distances = { 0.5, 0.70, 0.85, 1.0, 1.2 };
```

**原理**: 如果物体周围 0.85m 内都被膨胀区域覆盖，尝试停在更远处。

#### 方案3: 为物体导航启用 OPTIMISTIC 模式

修改 `trySearchObjectPathWithDistance()`:
```cpp
// 当前调用 (默认 NORMAL 模式)
path_finder_->astarSearch(start2d, object_pose, distance, max_search_time)

// 修改为 OPTIMISTIC 模式 (允许穿越 UNKNOWN 区域)
path_finder_->astarSearch(start2d, object_pose, distance, max_search_time,
                          Astar2D::SAFETY_MODE::OPTIMISTIC)
```

**原理**: UNKNOWN 区域通常是未探索区域，OPTIMISTIC 模式允许路径穿过这些区域。

#### 方案4: 添加多方向候选目标点

在 `searchObjectPath()` 中添加:
```cpp
bool ExplorationManager::searchObjectPath(...)
{
  // ... 原有代码 ...

  // 如果所有距离都失败，尝试物体周围的多个方向
  if (path.empty()) {
    const std::vector<double> angles = {0, M_PI/4, M_PI/2, 3*M_PI/4,
                                        M_PI, -3*M_PI/4, -M_PI/2, -M_PI/4};
    const double search_radius = 1.0;  // 在物体1m范围内寻找可达点

    for (double angle : angles) {
      Vector2d candidate = object_pose + search_radius * Vector2d(cos(angle), sin(angle));
      if (sdf_map_->getInflateOccupancy(candidate) != 1 &&
          trySearchObjectPathWithDistance(start2d, candidate, 0.2, max_search_time,
                                          refined_pos, refined_path, "")) {
        ROS_WARN("Found alternative approach point at angle %.1f rad", angle);
        return true;
      }
    }
  }

  ROS_ERROR("Failed to find object path.");
  return false;
}
```

#### 方案5: 添加目标锁定机制防止振荡 (已部分实现)

系统已有 `suspicious_target_locked_` 机制，可以扩展到高置信度目标:
```cpp
// 在 planNextBestPoint() 中添加
if (object_clouds.size() > 0 && !searchObjectPath(...)) {
  // 路径失败但有高置信度物体时，不立即切换到 Frontier
  // 而是记录失败次数，超过阈值后才切换
  object_path_fail_count_++;
  if (object_path_fail_count_ < MAX_OBJECT_PATH_RETRY) {
    // 继续尝试导航到物体，可能下一帧地图更新后路径可达
    return SEARCH_BEST_OBJECT_RETRY;
  }
}
```

### 9.9 推荐修复顺序

| 优先级 | 方案 | 修改文件 | 风险 |
|--------|------|----------|------|
| 1 | 减小膨胀半径 (0.18→0.12) | `algorithm.xml` | 低 |
| 2 | 增加停靠距离选项 | `exploration_manager.cpp` | 低 |
| 3 | 启用 OPTIMISTIC 模式 | `exploration_manager.cpp` | 中 |
| 4 | 多方向候选点搜索 | `exploration_manager.cpp` | 低 |
| 5 | 目标锁定防振荡 | `exploration_manager.cpp` | 低 |

### 9.10 调试建议

在 `trySearchObjectPathWithDistance()` 中添加详细日志:
```cpp
bool ExplorationManager::trySearchObjectPathWithDistance(...)
{
  // 添加调试信息
  ROS_DEBUG("Trying path: start=(%.2f,%.2f) -> object=(%.2f,%.2f), dist=%.2f",
            start2d.x(), start2d.y(), object_pose.x(), object_pose.y(), distance);
  ROS_DEBUG("Object inflated: %d, Start inflated: %d",
            sdf_map_->getInflateOccupancy(object_pose),
            sdf_map_->getInflateOccupancy(start2d));

  // ... 原有代码 ...
}
```

### 9.11 总结

| 项目 | 说明 |
|------|------|
| **问题根因** | A* 路径搜索失败，目标点落在膨胀障碍物区域内 |
| **核心参数** | `obstacles_inflation = 0.18m` 导致物体周围被大面积标记为不可达 |
| **表现** | 系统检测到高置信度物体，但无法规划路径，降级到Frontier探索 |
| **日志特征** | `[Navigation Mode] Get object_cloud num = 1` 后紧跟 `[ERROR] Failed to find object path` |
| **最快修复** | 将 `algorithm.xml` 中 `obstacles_inflation` 从 `0.18` 减小到 `0.12` |
| **完整修复** | 结合减小膨胀 + 增加停靠距离 + 多方向搜索 |

---

*文档生成时间: 2026-01-19*
*更新时间: 2026-01-22 - 添加目标切换问题诊断*
*更新时间: 2026-01-29 - 添加无进展检测机制*
*更新时间: 2026-02-07 - 添加接近检测失败后卡住问题诊断*
*基于 InfoNav 代码仓库分析*

---

## 十、接近阶段检测校验机制 (Approach Detection Validation)

### 10.1 功能概述

实现了一个接近阶段检测校验机制，确保智能体在 Object Navigation 模式下，只有在接近目标物体时检测到过目标，才能确认导航成功。如果到达目标位置但从未在接近阶段检测到目标（可能是因为导航到了墙的另一边），系统会：

1. 标记当前物体为 "invalid"
2. 重新规划到其他候选点
3. 如果没有其他目标，降级到 Frontier 探索

### 10.2 关键流程

```
Object Navigation 模式 (SEARCH_OBJECT)
        │
        ▼
┌───────────────────────────────────┐
│ 每步调用 updateApproachDetection │
│ Check() 更新滑动窗口              │
└─────────────┬─────────────────────┘
              │
              ▼
┌───────────────────────────────────┐
│ 检测器当前帧是否看到目标？        │
│ (last_frame_has_target_ &&        │
│  elapsed < 0.5s)                  │
└─────────────┬─────────────────────┘
              │
              ▼
┌───────────────────────────────────┐
│ 添加到滑动窗口 (最多保留 6 帧)     │
└─────────────┬─────────────────────┘
              │
              ▼
┌───────────────────────────────────┐
│ 距离目标 < 0.2m (到达) ?          │
└─────────────┬─────────────────────┘
              │ YES
              ▼
┌───────────────────────────────────┐
│ isApproachDetectionValid() ?      │
│ (滑动窗口中至少有1次检测到目标)    │
└─────────────┬─────────────────────┘
       ┌──────┴──────┐
       ▼             ▼
     TRUE          FALSE
       │             │
       ▼             ▼
    成功！      ┌──────────────────────┐
   REACH_      │ markObjectAsInvalid  │
   OBJECT      │ 设置 replan_flag     │
               │ 返回 SEARCH_OBJECT   │
               └──────────────────────┘
```

### 10.3 关键参数 (exploration_fsm.h - FSMConstants)

```cpp
constexpr double APPROACH_CHECK_NEAR_DISTANCE = 0.6;  // 触发最终验证的距离 (米)
constexpr int APPROACH_CHECK_STEP_WINDOW = 6;         // 滑动窗口大小 (步数)
constexpr double APPROACH_CONFIDENCE_RATIO = 0.4;     // 最小置信度比例
```

### 10.4 相关代码位置

| 功能 | 文件 | 行号 |
|------|------|------|
| 更新滑动窗口 | `exploration_fsm.cpp` | 309-311, 1954-1980 |
| 验证函数 | `exploration_fsm.cpp` | 1982-1991 |
| 验证失败处理 | `exploration_fsm.cpp` | 441-485 |
| 标记物体无效 | `object_map2d.cpp` | 145-149 |

---

## 十一、无进展检测机制 (No-Progress Detection)

### 11.1 问题背景

在 Frontier 探索过程中，智能体可能遇到以下情况：
- 由于深度传感器缺陷，某些 Frontier 的实际位置被错误估计
- 离散动作空间（30°转向、0.25m前进）限制了智能体的精确导航
- 智能体虽然在**移动**，但由于上述原因**无法有效接近目标 Frontier**

现有的卡住检测机制（`stucking_action_count_`）只能检测智能体完全静止的情况，无法处理"移动但无进展"的场景。

### 11.2 解决方案

实现了**无进展检测机制**，追踪智能体向同一 Frontier 的进展情况：

```
┌─────────────────────────────────────────────────────────────────────┐
│                      无进展检测流程                                  │
└─────────────────────────────────────────────────────────────────────┘

每步 callActionPlanner() 中：
        │
        ▼
┌───────────────────────────────────────┐
│ Frontier 位置变化 > 0.5m ?            │
└─────────────────┬─────────────────────┘
         ┌────────┴────────┐
         ▼                 ▼
       YES                NO
         │                 │
         ▼                 ▼
┌─────────────────┐  ┌─────────────────────────────┐
│ 新 Frontier      │  │ 同一 Frontier               │
│ 重置追踪状态     │  │ steps_toward_frontier_++    │
│ 记录初始距离     │  │ 更新最佳距离                │
└─────────────────┘  └──────────────┬──────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │ steps >= 50 (阈值) ?        │
                     └──────────────┬──────────────┘
                            ┌───────┴───────┐
                            ▼               ▼
                          YES              NO
                            │               │
                            ▼               └─► 继续导航
                     ┌─────────────────────────────┐
                     │ 距离改善 < 0.3m ?            │
                     └──────────────┬──────────────┘
                            ┌───────┴───────┐
                            ▼               ▼
                          YES              NO
                            │               │
                            ▼               ▼
                 ┌──────────────────┐  ┌──────────────────┐
                 │ 无进展！          │  │ 有进展            │
                 │ 标记 DORMANT     │  │ 重置步数计数器    │
                 │ 选择其他 Frontier │  │ 继续追踪         │
                 └──────────────────┘  └──────────────────┘
```

### 11.3 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `NO_PROGRESS_MAX_STEPS` | 50 | 向同一 Frontier 的最大步数 |
| `NO_PROGRESS_MIN_IMPROVEMENT` | 0.3m | 要求的最小距离改善 |
| `NO_PROGRESS_CHECK_INTERVAL` | 15 | 每 N 步输出进度日志 |
| `FRONTIER_CHANGE_THRESHOLD` | 0.5m | 判定 Frontier 是否变化的阈值 |

### 11.4 追踪变量 (FSMData)

```cpp
// No-progress detection for frontier exploration
Vector2d tracked_frontier_pos_;      ///< 当前追踪的 Frontier 位置
double initial_dist_to_frontier_;    ///< 开始追踪时的距离
int steps_toward_frontier_;          ///< 向当前 Frontier 的步数
double best_dist_to_frontier_;       ///< 达到过的最小距离
```

### 11.5 日志输出示例

```
[No-Progress] New frontier target at (5.20, 3.10), initial dist = 2.50 m
[No-Progress] Step 15 toward frontier, initial=2.50, best=2.30, current=2.35, improvement=0.20
[No-Progress] Step 30 toward frontier, initial=2.50, best=2.20, current=2.40, improvement=0.30
[No-Progress] Step 45 toward frontier, initial=2.50, best=2.15, current=2.50, improvement=0.35
[No-Progress] Abandoning frontier after 50 steps with only 0.25m improvement (threshold: 0.30m)
[No-Progress] Frontier at (5.20, 3.10), initial dist=2.50, best dist=2.25, current dist=2.45
```

### 11.6 代码位置

| 功能 | 文件 | 位置 |
|------|------|------|
| 常量定义 | `exploration_fsm.h` | `FSMConstants` 命名空间 |
| 数据结构 | `exploration_data.h` | `FSMData` 结构体 |
| 核心逻辑 | `exploration_fsm.cpp` | `callActionPlanner()` 函数 |

### 11.7 与现有卡住检测的关系

| 机制 | 检测场景 | 触发条件 |
|------|----------|----------|
| `stucking_action_count_` | 完全静止 | 连续 25 步移动距离 < 0.05m |
| `stucking_next_pos_count_` | 原地卡住 | 同目标 + 移动距离 < 0.05m，连续 14 次 |
| **无进展检测** | 移动但无进展 | 50 步内距离改善 < 0.3m |

三种机制互补，覆盖了不同类型的导航困境。

---

## 十二、软到达检测机制 (Soft Arrival Detection)

### 12.1 问题背景

在 Object Navigation 模式下，智能体在接近目标物体时可能出现以下问题：
- **振荡行为**：智能体在尝试更接近目标时，由于障碍物或离散动作限制，出现"前进-后退"振荡
- **无法精确到达**：即使智能体已经非常接近目标（如 0.4m），也无法突破硬到达距离（0.2m）
- **卡住检测失效**：由于智能体确实在移动（每次移动 > 0.05m），现有的卡住检测无法触发

### 12.2 解决方案

实现了**软到达检测机制**，追踪智能体在接近阶段的距离改善情况：

```
┌─────────────────────────────────────────────────────────────────────┐
│                      软到达检测流程                                  │
└─────────────────────────────────────────────────────────────────────┘

每步 callActionPlanner() 中（Object Navigation 模式）：
        │
        ▼
┌───────────────────────────────────────┐
│ dist_to_target < 0.60m ?              │  ← SOFT_ARRIVAL_DISTANCE
└─────────────────┬─────────────────────┘
         ┌────────┴────────┐
         ▼                 ▼
       YES                NO
         │                 │
         ▼                 ▼
┌─────────────────┐  ┌─────────────────────────────┐
│ 进入软到达范围   │  │ 重置追踪状态               │
│ 追踪距离改善     │  └─────────────────────────────┘
└────────┬────────┘
         │
         ▼
┌───────────────────────────────────────┐
│ dist < approach_best_dist - 0.03m ?   │  ← 有效改善阈值
└─────────────────┬─────────────────────┘
         ┌────────┴────────┐
         ▼                 ▼
       YES                NO
         │                 │
         ▼                 ▼
┌─────────────────┐  ┌─────────────────────────────┐
│ 有进展！         │  │ 无进展                       │
│ 更新 best_dist  │  │ no_improvement_count++      │
│ 重置无改善计数   │  └──────────────┬──────────────┘
└─────────────────┘                 │
                                    ▼
                     ┌─────────────────────────────┐
                     │ no_improvement >= 5 ?       │  ← MAX_NO_IMPROVE
                     └──────────────┬──────────────┘
                            ┌───────┴───────┐
                            ▼               ▼
                          YES              NO
                            │               │
                            ▼               └─► 继续导航
                 ┌──────────────────────────────┐
                 │ 软到达触发！                   │
                 │ - 检查 Approach Detection    │
                 │ - 执行 VLM 验证（如启用）    │
                 │ - 确认到达或重新规划         │
                 └──────────────────────────────┘
```

### 12.3 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `SOFT_ARRIVAL_DISTANCE` | 0.60m | 进入软到达检测范围的距离 |
| `SOFT_ARRIVAL_IMPROVEMENT_THRESH` | 0.03m | 有效改善的最小阈值 |
| `SOFT_ARRIVAL_MAX_NO_IMPROVE` | 5 | 无改善最大步数，超过则触发软到达 |

### 12.4 追踪变量 (FSMData)

```cpp
// Soft arrival detection for object navigation
double approach_best_dist_;          ///< 当前接近过程中达到的最佳距离
int approach_attempt_count_;         ///< 当前接近尝试的总步数
int approach_no_improvement_count_;  ///< 连续无改善的步数
```

### 12.5 日志输出示例

```
[Soft Arrival] Progress! dist=0.52m (best=0.52m), attempts=1
[Soft Arrival] Progress! dist=0.45m (best=0.45m), attempts=3
[Soft Arrival] No progress: dist=0.48m, best=0.45m, no_improve=1/5
[Soft Arrival] No progress: dist=0.47m, best=0.45m, no_improve=2/5
[Soft Arrival] No progress: dist=0.50m, best=0.45m, no_improve=3/5
[Soft Arrival] No progress: dist=0.46m, best=0.45m, no_improve=4/5
[Soft Arrival] No progress: dist=0.49m, best=0.45m, no_improve=5/5
[Soft Arrival] Triggered! Agent oscillating at dist=0.49m (best=0.45m) after 8 attempts
[Soft Arrival] SUCCESS! Reached object at dist=0.49m (oscillation detected)
```

### 12.6 与其他机制的关系

| 机制 | 触发条件 | 检测范围 |
|------|----------|----------|
| **硬到达** (`REACH_DISTANCE`) | dist < 0.20m | 精确到达 |
| **卡住到达** (`SOFT_REACH_DISTANCE`) | dist < 0.45m + 完全静止 | 卡住时软化 |
| **软到达** (新机制) | dist < 0.60m + 连续5步无改善 | 振荡时软化 |

三种机制形成递进关系：
1. 首先尝试硬到达（精确）
2. 如果完全静止，检查卡住到达
3. 如果在移动但振荡，检查软到达

### 12.7 代码位置

| 功能 | 文件 | 位置 |
|------|------|------|
| 常量定义 | `exploration_fsm.h` | `FSMConstants` 命名空间 |
| 数据结构 | `exploration_data.h` | `FSMData` 结构体 |
| 核心逻辑 | `exploration_fsm.cpp` | `callActionPlanner()` 函数 |
| 重置函数 | `exploration_fsm.cpp` | `resetSoftArrivalTracking()` |

---

## 十三、置信度融合机制优化

### 13.1 问题背景

在目标导航过程中，智能体可能在前往目标途中突然放弃目标，尽管目标物体仍显示为高置信度（红色标记）。

### 13.2 根本原因分析

通过对比 ApexNav 和 InfoNav 代码发现，InfoNav 的负样本反馈机制存在问题：

| 方面 | **ApexNav** | **InfoNav (修改前)** |
|------|-------------|----------------------|
| 调用参数 | `max(0.0, itm_score_)` | `0.0` (硬编码) |
| label=0 时的置信度 | `confidence_now = itm_score` (正的ITM分数) | `confidence_now = 0.0` (固定为0) |

**问题**: InfoNav 将 ITM score 移除后，直接传入 `0.0`，导致**每次观察都是纯负样本**，持续拉低目标对象的置信度。

### 13.3 解决方案

**禁用负样本反馈机制**，只保留有正向检测结果时的置信度更新。

### 13.4 代码修改

**文件**: `src/planner/plan_env/src/map_ros.cpp` (第520-522行)

```cpp
// 修改前
map_->object_map2d_->inputObservationObjectsCloud(observation_clouds, 0.0);

// 修改后 (已注释)
// DISABLED: Negative sample feedback was causing confidence to drop during navigation,
// leading to premature target abandonment when object not detected in some frames.
// map_->object_map2d_->inputObservationObjectsCloud(observation_clouds, 0.0);
```

### 13.5 影响

- 智能体在前往目标途中，即使某些帧没有检测到目标，置信度也不会被拉低
- 目标物体的置信度更加稳定，减少了导航过程中的目标切换

---

## 十四、检测方向优先路径选择 (Detection-Direction Prioritized Path Selection)

### 14.1 问题背景

智能体检测到目标后，可能选择空间上更近但实际上在墙另一边的路径，而不是从检测到目标的方向前往。

**问题场景**:
```
┌──────────────────────────────────────┐
│                                      │
│   智能体检测位置 ○ ──检测方向──→ ● 目标物体  │
│                                      │
│   ████████████ 墙壁 █████████████    │
│                                      │
│              ○ 智能体当前位置         │
│                                      │
└──────────────────────────────────────┘

原始行为: 智能体可能选择从墙下方绕过去（空间距离更近）
期望行为: 优先从检测方向（墙上方）接近目标
```

### 14.2 根本原因

1. **候选点纯距离选择**: `findCandidateObjectPoints()` 使用 KdTree 找最近的点
2. **第一条可达路径即返回**: `searchObjectPath()` 不考虑路径质量
3. **缺乏检测方向信息**: `ObjectCluster` 结构中未存储检测时的机器人位置

### 14.3 解决方案

记录首次检测方向，并在选择候选点时优先考虑该方向。

### 14.4 数据结构修改

**文件**: `src/planner/plan_env/include/plan_env/object_map2d.h`

在 `ObjectCluster` 结构中添加检测方向字段:

```cpp
struct ObjectCluster {
  // ... 原有字段 ...

  /******* Detection Direction Information *******/
  Vector2d first_detection_pos_;      ///< Robot position when object was first detected
  bool has_detection_pos_;            ///< Whether detection position is recorded

  ObjectCluster(int size = 5)
    : // ... 原有初始化 ...
    , first_detection_pos_(Vector2d::Zero())
    , has_detection_pos_(false)
  {
  }
};
```

添加相关方法:

```cpp
class ObjectMap2D {
public:
  // ... 原有方法 ...

  /**
   * @brief Set current robot position (called before object detection)
   */
  void setCurrentRobotPosition(const Vector2d& pos);

  /**
   * @brief Get first detection position for an object
   */
  bool getFirstDetectionPosition(int object_id, Vector2d& detection_pos) const;

private:
  Vector2d current_robot_pos_;  ///< Current robot position for recording detection direction
};
```

### 14.5 检测位置记录

**文件**: `src/planner/plan_env/src/object_map2d.cpp`

在 `createNewObjectCluster()` 中记录首次检测位置:

```cpp
void ObjectMap2D::createNewObjectCluster(...)
{
  // ... 原有代码 ...

  // Record first detection position for direction-aware path planning
  obj.first_detection_pos_ = current_robot_pos_;
  obj.has_detection_pos_ = true;

  ROS_INFO("[New Object] id=%d, first_detection_pos=(%.2f, %.2f)",
           obj.id_, obj.first_detection_pos_.x(), obj.first_detection_pos_.y());

  // ... 原有代码 ...
}
```

**文件**: `src/planner/plan_env/src/map_ros.cpp`

在检测前设置当前机器人位置:

```cpp
void MapROS::detectedObjectCloudCallback(...)
{
  // ... 原有代码 ...

  // Set current robot position before object detection
  Eigen::Vector2d robot_pos_2d(camera_pos_(0), camera_pos_(1));
  map_->object_map2d_->setCurrentRobotPosition(robot_pos_2d);

  // 然后进行物体检测
  map_->inputObjectCloud2D(detected_objects, detected_object_cluster_ids);
}
```

### 14.6 路径选择算法修改

**文件**: `src/planner/exploration_manager/src/exploration_manager.cpp`

修改 `searchObjectPath()` 函数，按检测方向优先级排序候选点:

```cpp
bool ExplorationManager::searchObjectPath(...)
{
  // Get more candidate points (10 instead of 5)
  std::vector<Vector2d> candidate_points = findCandidateObjectPoints(start, object_cloud, 10);

  // Compute object center
  Vector2d object_center(0, 0);
  for (const auto& pt : object_cloud->points) {
    object_center += Vector2d(pt.x, pt.y);
  }
  object_center /= object_cloud->points.size();

  // Get first detection position
  Vector2d detection_pos;
  bool has_detection_dir = false;
  Vector2d detection_dir;
  if (object_id >= 0 && object_map2d_->getFirstDetectionPosition(object_id, detection_pos)) {
    detection_dir = (object_center - detection_pos).normalized();
    has_detection_dir = true;
  }

  // Sort candidates by direction preference
  if (has_detection_dir) {
    std::vector<std::pair<double, size_t>> scored_candidates;
    for (size_t i = 0; i < candidate_points.size(); ++i) {
      const Vector2d& cand = candidate_points[i];
      Vector2d cand_to_obj = (object_center - cand).normalized();

      // Direction score: dot product with detection direction
      double direction_score = cand_to_obj.dot(detection_dir);

      // Distance penalty
      double dist_to_robot = (cand - start2d).norm();

      // Combined score: 70% direction + 30% distance penalty
      double score = 0.7 * direction_score - 0.3 * (dist_to_robot / 10.0);
      scored_candidates.emplace_back(score, i);
    }

    // Sort by score (highest first)
    std::sort(scored_candidates.begin(), scored_candidates.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Reorder candidate_points by score
    std::vector<Vector2d> sorted_candidates;
    for (const auto& sc : scored_candidates) {
      sorted_candidates.push_back(candidate_points[sc.second]);
    }
    candidate_points = std::move(sorted_candidates);
  }

  // Try candidates in priority order (detection direction first)
  for (const auto& object_pose : candidate_points) {
    // ... 原有路径规划逻辑 ...
  }
}
```

### 14.7 评分公式

```
score = 0.7 × direction_score - 0.3 × (dist_to_robot / 10.0)
```

其中:
- `direction_score`: 候选点到物体方向与检测方向的点积 (范围 [-1, 1])
  - 值为 1 表示候选点正好在检测方向
  - 值为 -1 表示候选点在相反方向
- `dist_to_robot`: 候选点到当前机器人的距离
- 权重 0.7/0.3: 方向优先，但也考虑距离因素

### 14.8 数据流总结

```
物体检测回调
    │
    ├─► setCurrentRobotPosition(camera_pos)  ← 记录当前位置
    │
    └─► inputObjectCloud2D()
            │
            └─► createNewObjectCluster()
                    │
                    └─► obj.first_detection_pos_ = current_robot_pos_  ← 存储检测位置
                        obj.has_detection_pos_ = true

路径规划时
    │
    └─► searchObjectPath()
            │
            ├─► getFirstDetectionPosition(object_id)  ← 获取检测位置
            │
            ├─► 计算检测方向: detection_dir = (center - detection_pos).normalized()
            │
            ├─► 对候选点评分并排序
            │
            └─► 按优先级尝试路径规划
```

### 14.9 修改文件汇总

| 文件 | 修改内容 |
|------|----------|
| `object_map2d.h` | 添加 `first_detection_pos_`、`has_detection_pos_` 字段和相关方法 |
| `object_map2d.cpp` | 实现 `getFirstDetectionPosition()`，在创建时记录检测位置 |
| `map_ros.cpp` | 在检测前调用 `setCurrentRobotPosition()` |
| `exploration_manager.cpp` | 修改 `searchObjectPath()` 按检测方向排序候选点 |

### 14.10 日志输出示例

```
[New Object] id=3, first_detection_pos=(2.50, 1.80)
[searchObjectPath] Object 3: detection_pos=(2.50,1.80), center=(4.20,3.10), dir=(0.74,0.67)
[searchObjectPath] Found path using candidate 0 (detection-direction prioritized)
```

---

## 十五、问题诊断：接近检测失败后智能体卡住 (Stuck After Approach Validation Failure)

### 15.1 问题描述

**现象**: 智能体在 Object Navigation 模式下到达目标物体附近 (< 0.2m)，但由于最后 6 帧没有检测到目标物体，没有执行 STOP 完成 episode。同时，智能体也没有进入其他状态（如前往新的 object 或 frontier），而是处于一种"中间状态"。

### 15.2 问题分析

#### 检测验证失败后的处理流程

```
距离 < 0.2m 触发到达检查
        │
        ▼
┌───────────────────────────────────┐
│ isApproachDetectionValid() ?      │
│ (检查滑动窗口中是否有检测)         │
└─────────────┬─────────────────────┘
              │ FALSE (6帧都没检测到)
              ▼
┌───────────────────────────────────┐
│ markObjectAsInvalid(obj_id)       │  ← exploration_fsm.cpp:457
│ 标记物体为无效                     │
└─────────────┬─────────────────────┘
              │
              ▼
┌───────────────────────────────────┐
│ fd_->replan_flag_ = true          │  ← exploration_fsm.cpp:482
│ return fd_->final_result_         │  ← 仍返回 SEARCH_OBJECT
└─────────────┬─────────────────────┘
              │
              ▼
        下一个 FSM 循环
              │
              ▼
┌───────────────────────────────────┐
│ planNextBestPoint()               │
│ getTopConfidenceObjectCloud()     │
│   → 跳过 invalid 物体             │
└─────────────┬─────────────────────┘
              │
        ┌─────┴─────┐
        ▼           ▼
    有其他物体    无其他物体
        │           │
        ▼           ▼
  正常导航     检查 Frontier
                    │
              ┌─────┴─────┐
              ▼           ▼
          有 Frontier  无 Frontier
              │           │
              ▼           ▼
          EXPLORE     NO_FRONTIER
                      → STOP
```

#### 可能导致卡住的原因

**原因1: 物体ID获取失败导致无法标记**

```cpp
// exploration_fsm.cpp:447-453
int current_obj_id = expl_manager_->object_map2d_->getCurrentTargetObjectId();

if (current_obj_id < 0) {
  current_obj_id = expl_manager_->object_map2d_->findNearestObjectId(expl_manager_->ed_->next_pos_);
  // 如果两者都返回 -1，物体不会被标记为 invalid
}

if (current_obj_id >= 0) {
  expl_manager_->object_map2d_->markObjectAsInvalid(current_obj_id);  // 只有 >= 0 才执行
} else {
  ROS_ERROR("[Final Approach] Could not find any object to mark as invalid!");
  // 问题：物体没有被标记，下次规划仍会选择同一个物体
}
```

**日志特征**:
```
[Final Approach] getCurrentTargetObjectId returned -1, found nearest object id=-1
[Final Approach] Could not find any object to mark as invalid!
```

**原因2: 路径为空但状态仍是 SEARCH_OBJECT**

```cpp
// exploration_fsm.cpp:699-702
if (final_res == FINAL_RESULT::NO_FRONTIER || expl_manager_->ed_->next_best_path_.empty()) {
  ROS_WARN("No (passable) frontier");
  return final_res;  // 如果 final_res 仍是 SEARCH_OBJECT，可能出现问题
}
```

**原因3: Approach Lock 没有正确释放**

```cpp
// exploration_fsm.cpp:474-476
if (expl_manager_->isObjectApproachLocked()) {
  expl_manager_->setObjectApproachLock(false);
}
// 如果这里没有执行到，下次规划可能仍锁定同一个点
```

### 15.3 日志特征

问题发生时的典型日志序列：

```
[Approach Check] Target DETECTED at dist=0.45m (window: 5/6)
[Approach Check] (后续帧没有检测日志)
[Final Approach] Validation FAILED! Target not detected in sliding window.
[Final Approach] Robot may have approached from wrong side (e.g., behind a wall).
[Final Approach] Object 3 marked as INVALID (validation failed).
[Final Approach] Replanning to find other targets or frontiers...
(之后没有明确的导航目标日志，智能体可能陷入循环)
```

### 15.4 相关代码位置

| 功能 | 文件 | 行号 |
|------|------|------|
| 验证失败处理 | `exploration_fsm.cpp` | 441-485 |
| 物体ID获取 | `exploration_fsm.cpp` | 447-453 |
| 重规划触发 | `exploration_fsm.cpp` | 482 |
| 路径空检查 | `exploration_fsm.cpp` | 699-702 |
| 物体标记无效 | `object_map2d.cpp` | 145-149 |
| 跳过无效物体 | `object_map2d.cpp` | 879-883 |

### 15.5 可能的解决方案

#### 方案1: 增强物体ID获取逻辑

当 `getCurrentTargetObjectId()` 和 `findNearestObjectId()` 都返回 -1 时，使用目标位置直接查找：

```cpp
if (current_obj_id < 0) {
  // 尝试根据 next_pos_ 附近的物体点云直接查找
  std::vector<int> nearby_ids;
  expl_manager_->object_map2d_->findObjectsNearPosition(
      expl_manager_->ed_->next_pos_, 0.5, nearby_ids);
  if (!nearby_ids.empty()) {
    current_obj_id = nearby_ids[0];
  }
}
```

#### 方案2: 强制进入 Frontier 探索模式

当验证失败且无法标记物体时，强制切换到探索模式：

```cpp
if (current_obj_id < 0) {
  ROS_ERROR("[Final Approach] Forcing exploration mode due to object ID failure");
  final_res = FINAL_RESULT::EXPLORE;  // 强制切换
  return final_res;
}
```

#### 方案3: 添加重规划失败计数器

防止无限循环重规划：

```cpp
static int replan_fail_count = 0;
if (fd_->replan_flag_ && expl_manager_->ed_->next_best_path_.empty()) {
  replan_fail_count++;
  if (replan_fail_count > 5) {
    ROS_ERROR("[FSM] Replan failed %d times, switching to NO_FRONTIER", replan_fail_count);
    replan_fail_count = 0;
    return FINAL_RESULT::NO_FRONTIER;
  }
}
```

### 15.6 调试建议

1. **添加详细日志**：在验证失败后的每个关键点添加日志
2. **检查物体ID状态**：验证 `getCurrentTargetObjectId()` 的返回值
3. **监控 replan_flag**：跟踪重规划标志的设置和消费
4. **检查 next_best_path_**：确认重规划后路径是否为空

```cpp
// 在 callActionPlanner() 开始处添加
ROS_DEBUG("[FSM Debug] replan_flag=%d, final_result=%d, path_size=%zu",
          fd_->replan_flag_, fd_->final_result_,
          expl_manager_->ed_->next_best_path_.size());
```

### 15.7 总结

| 项目 | 说明 |
|------|------|
| **问题根因** | 接近检测验证失败后，物体可能无法正确标记为 invalid，或重规划没有找到新目标 |
| **核心参数** | `APPROACH_CHECK_STEP_WINDOW = 6` (滑动窗口大小) |
| **表现** | 智能体距离目标 < 0.2m，但没有 STOP，也没有前往新目标 |
| **日志特征** | `[Final Approach] Validation FAILED!` 后缺少明确的导航目标 |
| **可能触发条件** | 1) 物体ID为 -1  2) 无其他高置信度物体  3) 无可达 Frontier |

---

*文档更新时间: 2026-02-07*
*新增: 无进展检测机制 (No-Progress Detection)*
*新增: 软到达检测机制 (Soft Arrival Detection)*
*新增: 接近检测失败后卡住问题诊断*
*新增: 禁用负样本反馈机制*
*新增: 检测方向优先路径选择 (Detection-Direction Prioritized Path Selection)*
