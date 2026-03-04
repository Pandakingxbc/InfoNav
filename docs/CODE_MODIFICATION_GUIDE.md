# InfoNav 代码机制详解与优化指南

本文档详细解析了 InfoNav 系统中三个关键机制的工作原理，并基于测试数据提供优化建议。

---

## 目录

1. [Overdepth 机制：物体在深度传感器范围外的导航](#1-overdepth-机制物体在深度传感器范围外的导航)
2. [智能体朝向约束：检测方向优先的路径规划](#2-智能体朝向约束检测方向优先的路径规划)
3. [最后几步检测约束：接近目标时的可见性验证](#3-最后几步检测约束接近目标时的可见性验证)
4. [**[新增] Final Approach Validation：终点扫视验证机制**](#4-新增-final-approach-validation终点扫视验证机制)
5. [测试数据分析与优化建议](#5-测试数据分析与优化建议)

---

## 1. Overdepth 机制：物体在深度传感器范围外的导航

### 1.1 原理概述

当目标物体被检测到但其深度超出传感器的有效范围时，系统会将这些点存储在专门的 `over_depth_object_cloud_` 中，并引导智能体向该方向移动，直到目标进入有效深度范围。

### 1.2 关键阈值

| 参数 | 值 | 位置 |
|------|-----|------|
| 深度过滤阈值 | `depth_filter_maxdist_ - 0.10` 米 | [map_ros.cpp:267](src/planner/plan_env/src/map_ros.cpp#L267) |
| 一致性保持帧数 | 5 帧 | [map_ros.cpp:311](src/planner/plan_env/src/map_ros.cpp#L311) |

### 1.3 核心代码

**深度过滤与存储** ([map_ros.cpp:265-283](src/planner/plan_env/src/map_ros.cpp#L265-L283)):

```cpp
for (auto object_pt : single_object_cloud->points) {
  Eigen::Vector3d object_pt3d = Eigen::Vector3d(object_pt.x, object_pt.y, object_pt.z);
  if ((object_pt3d - camera_pos_).norm() > depth_filter_maxdist_ - 0.10) {
    // 仅为目标物体 (label == 0) 存储超深度点
    if (label == 0)
      over_depth_object_cloud->points.push_back(object_pt);
    continue;
  }
  tmp_object_cloud->points.push_back(object_pt);
}
```

**一致性追踪** ([map_ros.cpp:307-317](src/planner/plan_env/src/map_ros.cpp#L307-L317)):

```cpp
// 保持超深度物体追踪的一致性（防止抖动）
if (continue_over_depth_count_ == -1 &&
    !map_->object_map2d_->over_depth_object_cloud_->points.empty())
  continue_over_depth_count_ = 0;
else if (continue_over_depth_count_ <= 4 && continue_over_depth_count_ >= 0) {
  continue_over_depth_count_++;
  *map_->object_map2d_->over_depth_object_cloud_ = *last_over_depth_cloud;
}
else {
  continue_over_depth_count_ = -1;
}
```

**导航优先级** ([exploration_manager.cpp:192-204](src/planner/exploration_manager/src/exploration_manager.cpp#L192-L204)):

```cpp
// ==================== 导航模式：超深度物体 ====================
if (!object_map2d_->over_depth_object_cloud_->points.empty()) {
  ROS_WARN("[Navigation Mode (Over Depth)] Get over depth object cloud");

  // 发现超深度物体时解锁可疑目标
  if (suspicious_target_locked_) {
    setSuspiciousTargetLock(false);
  }

  if (searchObjectPath(
          pos, object_map2d_->over_depth_object_cloud_, ed_->next_pos_, ed_->next_best_path_))
    return SEARCH_OVER_DEPTH_OBJECT;
}
```

### 1.4 导航优先级顺序

1. **确认的目标物体** (SEARCH_BEST_OBJECT)
2. **超深度物体** (SEARCH_OVER_DEPTH_OBJECT) ← 你的修改涉及此处
3. **锁定的可疑目标** (SEARCH_SUSPICIOUS_OBJECT)
4. **前沿探索** (EXPLORE)

### 1.5 可能的修改点

- **阈值调整**: `depth_filter_maxdist_ - 0.10` 中的 0.10 米可以根据传感器精度调整
- **一致性帧数**: 当前为 5 帧（0-4），可根据需要增减
- **标签过滤**: 目前只对 `label == 0` 的目标物体生效

---

## 2. 智能体朝向约束：检测方向优先的路径规划

### 2.1 原理概述

系统记录首次检测到物体时的机器人位置，并在后续路径规划中优先选择从该方向接近物体的候选点。这确保智能体从"看得见"的方向接近目标，而不是从墙后或盲区接近。

### 2.2 关键数据结构

**ObjectCluster 中的检测方向信息** ([object_map2d.h:96-98](src/planner/plan_env/include/plan_env/object_map2d.h#L96-L98)):

```cpp
/******* Detection Direction Information *******/
Vector2d first_detection_pos_;      ///< 首次检测时的机器人位置
bool has_detection_pos_;            ///< 是否已记录检测位置
```

### 2.3 核心代码

**记录首次检测位置** ([object_map2d.cpp:546-548](src/planner/plan_env/src/object_map2d.cpp#L546-L548)):

```cpp
// 为方向感知路径规划记录首次检测位置
obj.first_detection_pos_ = current_robot_pos_;
obj.has_detection_pos_ = true;
```

**方向优先排序** ([exploration_manager.cpp:790-829](src/planner/exploration_manager/src/exploration_manager.cpp#L790-L829)):

```cpp
// 获取首次检测位置并计算检测方向
Vector2d detection_pos;
bool has_detection_dir = false;
Vector2d detection_dir;
if (object_id >= 0 && object_map2d_->getFirstDetectionPosition(object_id, detection_pos)) {
  detection_dir = (object_center - detection_pos).normalized();
  has_detection_dir = true;
}

// 如果有检测方向，按方向偏好排序候选点
if (has_detection_dir) {
  std::vector<std::pair<double, size_t>> scored_candidates;
  for (size_t i = 0; i < candidate_points.size(); ++i) {
    const Vector2d& cand = candidate_points[i];
    // 从候选点到物体中心的方向
    Vector2d cand_to_obj = (object_center - cand).normalized();
    // 点积：分数越高表示候选点在检测方向同侧
    // (即从检测方向接近)
    double direction_score = cand_to_obj.dot(detection_dir);
    // 同时考虑到机器人当前位置的距离作为次要因素
    double dist_to_robot = (cand - start2d).norm();
    // 综合分数：方向偏好（权重0.7）+ 距离惩罚（权重0.3）
    double score = 0.7 * direction_score - 0.3 * (dist_to_robot / 10.0);
    scored_candidates.emplace_back(score, i);
  }
  // 按分数排序（最高优先）
  std::sort(scored_candidates.begin(), scored_candidates.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
}
```

### 2.4 权重配置

| 因素 | 权重 | 说明 |
|------|------|------|
| 方向偏好 | 0.7 | 候选点是否在检测方向同侧 |
| 距离惩罚 | 0.3 | 候选点到当前位置的距离 |

### 2.5 可能的修改点

- **权重调整**: 修改 `0.7` 和 `0.3` 的比例
- **距离归一化**: 修改 `/10.0` 中的归一化因子
- **方向计算逻辑**: 可考虑使用更复杂的方向评估（如视野锥）

---

## 3. 最后几步检测约束：接近目标时的可见性验证

### 3.1 原理概述

在智能体接近目标物体的最后几步（0.6米范围内），系统通过滑动窗口机制检查目标是否在最近几步中被检测到至少一次。如果没有检测到，表明智能体可能从错误方向（如墙后）接近，系统会标记该接近点为失败并重新规划路径。

### 3.2 关键参数

**常量定义** ([exploration_fsm.h:104-109](src/planner/exploration_manager/include/exploration_manager/exploration_fsm.h#L104-L109)):

```cpp
// ==================== 接近检测检查（物体导航）====================
// 在到达目标前的最后几步验证目标可见性
// 必须在距离 < NEAR_DISTANCE 前的最后 N 步中至少检测到目标一次
constexpr double APPROACH_CHECK_NEAR_DISTANCE = 0.6;  ///< 触发验证的距离阈值（米）
constexpr int APPROACH_CHECK_STEP_WINDOW = 4;         ///< 检查检测的最近步数
constexpr double APPROACH_CONFIDENCE_RATIO = 2.0 / 3.0;  ///< 接近检测的最小置信度比例
```

| 参数 | 值 | 说明 |
|------|-----|------|
| APPROACH_CHECK_NEAR_DISTANCE | 0.6 米 | 触发验证的距离阈值 |
| APPROACH_CHECK_STEP_WINDOW | **4 步** | 滑动窗口大小 |
| 检测超时 | 0.5 秒 | 检测时间戳的有效期 |

### 3.3 核心代码

**状态更新（每步调用）** ([exploration_fsm.cpp:2031-2057](src/planner/exploration_manager/src/exploration_fsm.cpp#L2031-L2057)):

```cpp
void ExplorationFSM::updateApproachDetectionCheck(double dist_to_target)
{
  const int window_size = FSMConstants::APPROACH_CHECK_STEP_WINDOW;  // 4 步

  // 检查检测器当前是否看到目标
  bool current_detection = false;
  {
    boost::mutex::scoped_lock lock(detector_mutex_);
    // 如果检测发生在最近 0.5 秒内则认为有效
    double elapsed = (ros::Time::now() - last_detection_time_).toSec();
    current_detection = last_frame_has_target_ && elapsed < 0.5;
  }

  // 将当前检测结果添加到滑动窗口
  approach_detection_history_.push_back(current_detection);

  // 只保留最近 N 步
  while (static_cast<int>(approach_detection_history_.size()) > window_size) {
    approach_detection_history_.pop_front();
  }
}
```

**有效性检查** ([exploration_fsm.cpp:2059-2068](src/planner/exploration_manager/src/exploration_fsm.cpp#L2059-L2068)):

```cpp
bool ExplorationFSM::isApproachDetectionValid() const
{
  // 检查滑动窗口中是否至少检测到目标一次
  for (bool detected : approach_detection_history_) {
    if (detected) {
      return true;  // 只需要在最近 4 步中有一次检测
    }
  }
  return false;
}
```

**验证触发与失败处理** ([exploration_fsm.cpp:414-449](src/planner/exploration_manager/src/exploration_fsm.cpp#L414-L449)):

```cpp
// 到达物体 - 检查是否足够接近目标物体
if (!vlm_collection_locked_ &&
    fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT &&
    dist_to_target < reach_distance) {

  // ========== 接近检测验证 ==========
  // 检查在接近阶段（0.6m - 3.0m）是否至少检测到目标一次
  if (!isApproachDetectionValid()) {
    ROS_WARN("[Approach Check] FAILED! Target was NOT detected during approach.");
    ROS_WARN("[Approach Check] Robot may have approached from wrong side.");

    // 为该物体标记当前接近点为失败
    int current_obj_id = expl_manager_->object_map2d_->getCurrentTargetObjectId();
    if (current_obj_id >= 0) {
      expl_manager_->object_map2d_->addFailedApproachPoint(
          current_obj_id, expl_manager_->ed_->next_pos_);
    }

    // 解锁相关状态
    if (expl_manager_->isSuspiciousTargetLocked()) {
      expl_manager_->setSuspiciousTargetLock(false);
    }
    if (expl_manager_->isObjectApproachLocked()) {
      expl_manager_->setObjectApproachLock(false);
    }

    // 重置并强制重新规划
    resetApproachDetectionCheck();
    fd_->replan_flag_ = true;
    // 继续正常规划（会跳过失败的接近点）
  }
}
```

### 3.4 调用时机

更新函数在每一步都被调用 ([exploration_fsm.cpp:302-304](src/planner/exploration_manager/src/exploration_fsm.cpp#L302-L304)):

```cpp
double dist_to_target = (current_pos - expl_manager_->ed_->next_pos_).norm();
if (fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT) {
  updateApproachDetectionCheck(dist_to_target);
}
```

### 3.5 可能的修改点

- **窗口大小**: `APPROACH_CHECK_STEP_WINDOW = 4` 可增大以提供更长的检测窗口
- **触发距离**: `APPROACH_CHECK_NEAR_DISTANCE = 0.6` 可调整触发验证的时机
- **检测超时**: `0.5` 秒可根据检测器延迟调整
- **验证逻辑**: 可从"至少一次"改为"超过一定比例"

---

## 4. [新增] Final Approach Validation：终点扫视验证机制

### 4.1 功能概述

**已实现的新功能**：当智能体到达目标位置时，执行以下验证流程：

1. **提前返回检查** - 如果滑动窗口内已检测到目标，直接返回成功，跳过扫视
2. **旋转对准物体中心** - 使用**物体实际中心**（非规划目标点）计算朝向并旋转对齐
3. **下上扫视** - 先 LOOK_DOWN（物体通常在地面），再 LOOK_UP（两次，经过中间到上方），最后回中间
4. **左右微调** - 如果上下扫视未检测到目标，进行 ±15° 的左右微调
5. **OR 逻辑验证** - 扫视检测 OR 滑动窗口检测，任一满足即认为成功

### 4.2 状态机流程

```
到达规划终点 (dist < reach_distance)
         │
         ▼
┌─────────────────────────────────────┐
│ 滑动窗口已检测到目标？               │
│ (isApproachDetectionValid())        │
└───────────┬─────────────────────────┘
            │
    ┌───────┴───────┐
    │               │
    ▼               ▼
   是              否
    │               │
    ▼               ▼
 直接成功!     继续扫视验证
                    │
                    ▼
┌─────────────────────────┐
│ ROTATING_TO_OBJ         │ ─→ 旋转对准**物体中心**（非规划点）
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ LOOK_DOWN → WAIT        │ ─→ 先低头扫视（物体通常在地面）
└───────────┬─────────────┘    pitch: 0 → -1
            │
            ▼
┌─────────────────────────┐
│ LOOK_UP → WAIT (x2)     │ ─→ 抬头两次：下→中→上
└───────────┬─────────────┘    pitch: -1 → 0 → +1
            │
            ▼
┌─────────────────────────┐
│ LOOK_CENTER → WAIT      │ ─→ 低头回中间
└───────────┬─────────────┘    pitch: +1 → 0
            │
    ┌───────┴───────┐
    │               │
    ▼               ▼
 检测到?         未检测到
    │               │
    ▼               ▼
 成功!        ┌─────────────────┐
              │ ADJUST_LEFT     │
              │ ADJUST_RIGHT    │ ─→ 左右微调
              │ ADJUST_CENTER   │
              └────────┬────────┘
                       │
               ┌───────┴───────┐
               │               │
               ▼               ▼
            检测到?         未检测到
               │               │
               ▼               ▼
            成功!           失败→重规划
```

### 4.3 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `APPROACH_CHECK_STEP_WINDOW` | **6** (原4) | 滑动窗口大小，已增加 |
| `FINAL_APPROACH_YAW_THRESHOLD` | π/12 (~15°) | 对准阈值 |
| `FINAL_APPROACH_SCAN_WAIT` | 0.3秒 | 每个扫视动作的检测等待时间 |
| `FINAL_APPROACH_YAW_ADJUST` | π/12 (~15°) | 左右微调角度 |

### 4.4 核心代码位置

| 功能 | 文件 | 行号 |
|------|------|------|
| 状态机定义 | [exploration_fsm.h](src/planner/exploration_manager/include/exploration_manager/exploration_fsm.h) | 124-138 |
| 常量定义 | [exploration_fsm.h](src/planner/exploration_manager/include/exploration_manager/exploration_fsm.h) | 112-116 |
| 状态机实现 | [exploration_fsm.cpp](src/planner/exploration_manager/src/exploration_fsm.cpp) | 2091-2270 |
| 主流程集成 | [exploration_fsm.cpp](src/planner/exploration_manager/src/exploration_fsm.cpp) | 414-478 |

### 4.5 验证逻辑

**提前返回（滑动窗口已检测）**:
```cpp
// 在进入扫视前先检查滑动窗口
if (final_approach_state_ == FinalApproachState::IDLE && isApproachDetectionValid()) {
  ROS_INFO("[Final Approach] EARLY SUCCESS! Target detected in sliding window (6-step), skipping scan.");
  return FINAL_RESULT::REACH_OBJECT;  // 直接成功，跳过扫视
}
```

**扫视过程中的 OR 逻辑**:
```cpp
// 成功条件：扫视检测 OR 滑动窗口检测
if (final_approach_scan_detected_ || isApproachDetectionValid()) {
  return FINAL_RESULT::REACH_OBJECT;  // 成功
}
```

**面向物体中心（而非规划目标点）**:
```cpp
// 使用物体实际中心计算朝向，而不是 ed_->next_pos_（规划目标点）
int current_obj_id = expl_manager_->object_map2d_->getCurrentTargetObjectId();
Vector2d object_center;
if (expl_manager_->object_map2d_->getObjectCenter(current_obj_id, object_center)) {
  final_approach_target_yaw_ = std::atan2(object_center.y() - current_pos.y(),
                                           object_center.x() - current_pos.x());
}
```

### 4.6 优点

1. **提前返回优化** - 如果滑动窗口已检测到目标，直接成功，节省扫视时间
2. **正确的朝向计算** - 使用物体实际中心而非规划目标点，确保面向物体
3. **先下后上扫视** - 物体通常在地面（如椅子），先低头更容易找到
4. **pitch 状态跟踪** - 使用 `pitch_offset_` 跟踪相机俯仰，确保正确回到中间位置
5. **与滑动窗口互补** - OR 关系让系统更鲁棒

---

## 5. 测试数据分析与优化建议

### 5.1 测试结果概览

基于 `test_hm3dv2_val/record.txt` 的数据分析：

| 指标 | 值 |
|------|-----|
| 平均成功率 | ~72.5% |
| 平均 SPL | ~34.5% |
| 平均到目标距离 | ~1.58m |

### 5.2 失败模式统计

根据 grep 搜索结果，主要失败模式包括：

| 失败类型 | 说明 | 优化方向 |
|----------|------|----------|
| **false positive** | 检测器误报，认错目标 | 提高检测置信度阈值，增强VLM验证 |
| **false negative** | 目标存在但未检测到 | 优化检测器，增加多视角验证 |
| **stepout feasible** | 超时但实际上可以找到 | 增加步数限制或优化路径规划效率 |
| **[stepout] false negative** | 超时且错过了目标 | 优化探索策略，增加对高置信区域的关注 |
| **no frontier** | 没有可用的探索前沿 | 优化前沿生成或使用备用策略 |
| **stucking** | 智能体卡住 | 优化避障和脱困逻辑 |

### 5.3 针对机制的优化建议

#### 5.3.1 Overdepth 机制优化

**问题**: 超深度物体可能在几帧后消失，导致智能体无目标可追。

**建议修改**:

```cpp
// 位置: map_ros.cpp:311
// 当前: continue_over_depth_count_ <= 4 (5帧)
// 建议: 增加到 6-8 帧以提高稳定性
else if (continue_over_depth_count_ <= 6 && continue_over_depth_count_ >= 0) {
```

或者增加距离衰减：

```cpp
// 当智能体靠近时自动清除超深度状态
if (distance_moved_towards_overdepth > 2.0) {
  continue_over_depth_count_ = -1;  // 已经移动了足够距离，重置
}
```

#### 5.3.2 方向约束优化

**问题**: 权重固定可能不适合所有场景。

**建议修改**:

```cpp
// 位置: exploration_manager.cpp:816
// 当前: 固定权重 0.7 和 0.3
// 建议: 根据距离动态调整权重

double dist_ratio = dist_to_object / max_detection_distance;
double direction_weight = 0.5 + 0.3 * (1.0 - dist_ratio);  // 距离远时更重视方向
double distance_weight = 1.0 - direction_weight;
double score = direction_weight * direction_score - distance_weight * (dist_to_robot / 10.0);
```

#### 5.3.3 接近检测约束优化（已通过 Final Approach Validation 实现）

**问题**: 当前要求"至少一帧"可能太宽松或太严格。

**建议修改 A** - 更灵活的窗口：

```cpp
// 位置: exploration_fsm.h:108
// 当前: APPROACH_CHECK_STEP_WINDOW = 4
// 建议: 增加到 6 步以提供更多检测机会
constexpr int APPROACH_CHECK_STEP_WINDOW = 6;
```

**建议修改 B** - 要求一定比例的检测：

```cpp
// 位置: exploration_fsm.cpp:2059-2068
// 当前: 至少检测到一次
// 建议: 要求至少检测到 2 次（如果窗口是 6 步）
bool ExplorationFSM::isApproachDetectionValid() const
{
  int detection_count = 0;
  for (bool detected : approach_detection_history_) {
    if (detected) detection_count++;
  }
  return detection_count >= 2;  // 至少 2 次检测
}
```

**建议修改 C** - 早期预警：

```cpp
// 在更远距离（如 1.5m）开始追踪，但只在 0.6m 处验证
constexpr double APPROACH_TRACK_START_DISTANCE = 1.5;  // 开始追踪的距离
constexpr double APPROACH_CHECK_NEAR_DISTANCE = 0.6;   // 验证的距离
```

### 5.4 综合优化建议

1. **检测器置信度动态调整**
   - 在接近目标时降低置信度阈值（因为物体更大更清晰）
   - 在远距离时使用更严格的阈值

2. **多策略融合**
   - 当方向约束和超深度机制冲突时，使用置信度加权决策
   - 避免单一机制主导导航

3. **失败恢复机制增强**
   - 记录失败原因和位置
   - 实现回溯机制，当当前策略失败时尝试之前成功的路径

4. **日志增强**
   - 在关键决策点增加详细日志
   - 记录每次接近检测失败的具体情况以便后续分析

---

## 附录：关键文件路径索引

| 文件 | 主要内容 |
|------|----------|
| [exploration_fsm.h](src/planner/exploration_manager/include/exploration_manager/exploration_fsm.h) | FSM常量定义、接近检测声明 |
| [exploration_fsm.cpp](src/planner/exploration_manager/src/exploration_fsm.cpp) | 接近检测实现、旋转动作计算 |
| [exploration_manager.cpp](src/planner/exploration_manager/src/exploration_manager.cpp) | 超深度导航、方向优先路径规划 |
| [map_ros.cpp](src/planner/plan_env/src/map_ros.cpp) | 深度过滤、超深度点云管理 |
| [object_map2d.h](src/planner/plan_env/include/plan_env/object_map2d.h) | ObjectCluster定义、检测方向存储 |
| [object_map2d.cpp](src/planner/plan_env/src/object_map2d.cpp) | 首次检测位置记录 |

---

*文档生成时间: 2026-02-03*
*适用于 InfoNav 代码库*
