# InfoNav 检测系统重构计划

## 概述

本文档描述了 ApexNav 项目中检测器和 ObjectMap 的重构计划，目标是：
1. **保留 ITM/BLIP2**：用于 Multi-Source Semantic Value Map 构建
2. 实现论文中的 **Patience-Aware Navigation** 机制（时间动态阈值）
3. **简化 ObjectMap2D**：只存储物体及其置信度，查询时动态判断是否为确信目标

---

## 当前系统架构

```
                          ┌─────────────────────────────────────────────────┐
                          │              语义价值图构建路径                   │
RGB Image ───────────────►│  BLIP2 ITM → SemanticScores → MultiSourceValueMap│
                          │       (用于探索引导，保留不删除)                   │
                          └─────────────────────────────────────────────────┘

                          ┌─────────────────────────────────────────────────┐
                          │              目标检测路径                        │
RGB + Depth ─────────────►│  D-FINE/GroundingDINO → ObjectMap2D             │
                          │       (需要实现 Patience-Aware 机制)              │
                          └─────────────────────────────────────────────────┘
```

**关键区分**：
- **ITM/BLIP2**：用于语义价值图更新，指导探索方向 → **保留**
- **ObjectMap2D**：用于目标物体检测和导航 → **需要实现论文机制**

---

## 论文核心机制 (Section 3.5)

### 核心思想：Patience-Driven Threshold Adaptation

论文中的检测机制基于一个核心观察：**探索早期应该保守（避免假阳性），探索后期应该激进（提高召回率）**。

#### 1. 时间动态阈值

检测置信度阈值 τ(t) 随探索步数 t **线性下降**：

```
τ(t) = τ_high - (t / T_max) × (τ_high - τ_low)
```

| 参数 | 值 | 说明 |
|------|-----|------|
| τ_high | 0.4 | 初始严格阈值（探索开始时） |
| τ_low | 0.2 | 最终宽松阈值（探索结束时） |
| T_max | 500 | 最大探索步数 |

**行为示例**：
- t=0 时：τ(0) = 0.4（严格，减少假阳性）
- t=250 时：τ(250) = 0.3（中等）
- t=500 时：τ(500) = 0.2（宽松，提高召回率）

#### 2. 物体分类（两类物体）

根据当前阈值 τ(t)，Object Map 中的物体分为两类：

| 类型 | 条件 | 行为 |
|------|------|------|
| **确信目标 (Confirmed)** | confidence ≥ τ(t) | 可以直接触发导航 |
| **怀疑目标 (Suspicious)** | confidence < τ(t) | 继续探索，等待阈值下降后自动升级 |

**关键设计**：
- Object Map **只存储物体及其置信度**
- **不存储** `is_confirmed_` / `is_suspicious_` 状态
- 每次查询时，用当前 τ(t) **动态判断**物体是否为确信目标

#### 3. 可疑目标的自动升级机制

**核心思想**：可疑目标**不需要主动回访**，随着时间推移**自动升级**为确信目标。

**关键机制**：
- 物体的置信度是固定的（检测时确定，通过时序融合更新）
- 但阈值 τ(t) 随时间下降
- 因此，**原本的可疑目标会在后续时刻自动变成确信目标**

**时间轴示例**：
```
t=100: 检测到物体 A, confidence=0.35
       τ(100) = 0.36 → 0.35 < 0.36 → A 是可疑目标，继续探索

t=200: 阈值下降，τ(200) = 0.32
       查询 Object Map：0.35 ≥ 0.32 → A 现在是确信目标！
       导航过去
```

**行为逻辑**：
1. 机器人正常进行 frontier 探索
2. 每一步查询 Object Map 时，用当前 τ(t) 判断是否有确信目标
3. 一旦发现确信目标 → 立即导航过去
4. **不需要存储分类状态，不需要主动回访**

#### 4. 针对不同目标物体的阈值

小物体检测置信度通常较低，需要针对性调整：

| 物体类型 | 检测器 | 建议阈值 |
|---------|--------|---------|
| 大物体（床、沙发等） | D-FINE | 0.4 |
| 中等物体（椅子、马桶等） | D-FINE/GroundingDINO | 0.35 |
| 小物体（植物、花瓶、书等） | GroundingDINO | 0.25 |
| 细粒度物体 | GroundingDINO | 0.2 |

---

## 第一部分：ObjectMap2D 重构 (核心修改)

### 1.1 当前问题分析

| 问题 | 现状 | 论文要求 |
|------|------|----------|
| 阈值机制 | 固定 `min_confidence_` | 时间动态阈值 τ(t) |
| 置信度融合 | 简单加权平均 | 保留，用于时序融合 |
| 目标特定阈值 | 无 | 按物体类别设置不同阈值 |

### 1.2 需要添加的数据结构

**文件**: `src/planner/plan_env/include/plan_env/object_map2d.h`

#### 1.2.1 修改 ObjectCluster 结构体 (约第69行)

```cpp
struct ObjectCluster {
  // ... 现有成员保持不变 ...

  /******* Patience-Aware Navigation (论文 Section 3.5) *******/
  double fused_confidence_;        ///< 时序融合后的置信度（用于与 τ(t) 比较）

  // 修改构造函数
  ObjectCluster(int size = 5)
    : clouds_(size)
    , confidence_scores_(size, 0.0)
    , observation_nums_(size, 0)
    , observation_cloud_sums_(size, 0)
    , fused_confidence_(0.0)       // 新增
  {
  }
};
```

**注意**：不需要存储 `is_confirmed_` / `is_suspicious_` 状态，查询时动态计算。

### 1.3 需要添加的参数

**文件**: `src/planner/plan_env/include/plan_env/object_map2d.h`

在 `ObjectMap2D` 类的 private 成员区域添加 (约第175行后):

```cpp
private:
  // ... 现有成员 ...

  // ====== Patience-Aware Navigation 参数 (论文 Section 3.5) ======
  double tau_high_;                ///< 初始严格阈值 (默认 0.4)
  double tau_low_;                 ///< 最终宽松阈值 (默认 0.2)
  int T_max_;                      ///< 最大探索步数 (默认 500)
  int current_step_;               ///< 当前探索步数

  // ====== 目标特定阈值 ======
  std::unordered_map<std::string, double> target_specific_thresholds_;
  std::string current_target_category_;  ///< 当前搜索的目标类别
```

### 1.4 需要添加的方法声明

**文件**: `src/planner/plan_env/include/plan_env/object_map2d.h`

在 public 区域添加:

```cpp
public:
  // ... 现有方法 ...

  // ====== Patience-Aware Navigation 方法 (论文 Section 3.5) ======

  /**
   * @brief 计算当前时间步的动态阈值 τ(t)
   * @return 当前阈值 τ(t) = τ_high - (t/T_max) × (τ_high - τ_low)
   */
  double getCurrentConfidenceThreshold() const;

  /**
   * @brief 更新当前探索步数（每帧调用）
   * @param step 当前步数
   */
  void setCurrentStep(int step);

  /**
   * @brief 设置当前搜索的目标类别
   * @param category 目标类别名称（如 "bed", "toilet", "plant"）
   */
  void setTargetCategory(const std::string& category);

  /**
   * @brief 检查物体是否为确信目标（动态判断）
   * @param obj_idx 物体索引
   * @return confidence ≥ τ(t)
   */
  bool isConfirmedTarget(int obj_idx) const;

  /**
   * @brief 检查是否存在确信目标（动态判断）
   * @return 是否有至少一个确信目标
   */
  bool hasConfirmedTarget() const;

  /**
   * @brief 获取所有确信目标的点云（动态判断）
   * @param confirmed_clouds 输出：确信目标的点云列表
   */
  void getConfirmedTargetClouds(
      std::vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>>& confirmed_clouds);

  /**
   * @brief 获取针对当前目标类别的检测阈值
   * @return 目标特定阈值（若未配置则返回默认值）
   */
  double getTargetSpecificThreshold() const;

private:
  /**
   * @brief 初始化目标特定阈值映射表
   */
  void initTargetSpecificThresholds();
```

### 1.5 实现细节

**文件**: `src/planner/plan_env/src/object_map2d.cpp`

#### 1.5.1 构造函数中加载参数 (约第30行后添加)

```cpp
ObjectMap2D::ObjectMap2D(SDFMap2D* sdf_map, ros::NodeHandle& nh)
{
  // ... 现有初始化代码 ...

  // ====== Patience-Aware Navigation 参数 (论文 Section 3.5) ======
  nh.param("object/tau_high", tau_high_, 0.4);
  nh.param("object/tau_low", tau_low_, 0.2);
  nh.param("object/T_max", T_max_, 500);
  current_step_ = 0;
  current_target_category_ = "";

  // 初始化目标特定阈值
  initTargetSpecificThresholds();

  ROS_INFO("[ObjectMap2D] Patience-Aware Navigation enabled:");
  ROS_INFO("  - tau_high: %.2f", tau_high_);
  ROS_INFO("  - tau_low: %.2f", tau_low_);
  ROS_INFO("  - T_max: %d steps", T_max_);
}
```

#### 1.5.2 核心方法实现 (在文件末尾添加)

```cpp
// ==================== Patience-Aware Navigation 实现 ====================

void ObjectMap2D::initTargetSpecificThresholds()
{
  // 大物体：正常阈值
  target_specific_thresholds_["bed"] = 0.4;
  target_specific_thresholds_["couch"] = 0.4;
  target_specific_thresholds_["sofa"] = 0.4;
  target_specific_thresholds_["refrigerator"] = 0.4;
  target_specific_thresholds_["tv"] = 0.35;

  // 中等物体：稍低阈值
  target_specific_thresholds_["chair"] = 0.35;
  target_specific_thresholds_["toilet"] = 0.35;
  target_specific_thresholds_["sink"] = 0.35;
  target_specific_thresholds_["table"] = 0.35;

  // 小物体：更低阈值（检测置信度通常较低）
  target_specific_thresholds_["plant"] = 0.25;
  target_specific_thresholds_["vase"] = 0.25;
  target_specific_thresholds_["book"] = 0.2;
  target_specific_thresholds_["cup"] = 0.2;
  target_specific_thresholds_["bottle"] = 0.2;
  target_specific_thresholds_["remote"] = 0.2;

  ROS_INFO("[ObjectMap2D] Initialized %zu target-specific thresholds",
           target_specific_thresholds_.size());
}

double ObjectMap2D::getCurrentConfidenceThreshold() const
{
  // 论文 Equation 8: τ(t) = τ_high - (t/T_max) × (τ_high - τ_low)
  double t = std::min(current_step_, T_max_);
  double tau_t = tau_high_ - (t / static_cast<double>(T_max_)) * (tau_high_ - tau_low_);

  // 应用目标特定阈值修正（小物体使用更低阈值）
  double target_th = getTargetSpecificThreshold();
  if (target_th < tau_t) {
    tau_t = target_th;
  }

  return tau_t;
}

double ObjectMap2D::getTargetSpecificThreshold() const
{
  if (current_target_category_.empty())
    return tau_high_;  // 默认使用严格阈值

  auto it = target_specific_thresholds_.find(current_target_category_);
  if (it != target_specific_thresholds_.end()) {
    return it->second;
  }
  return tau_high_;  // 未知类别使用默认阈值
}

void ObjectMap2D::setCurrentStep(int step)
{
  current_step_ = step;
}

void ObjectMap2D::setTargetCategory(const std::string& category)
{
  // 转换为小写以便匹配
  std::string lower_category = category;
  std::transform(lower_category.begin(), lower_category.end(),
                 lower_category.begin(), ::tolower);
  current_target_category_ = lower_category;

  double specific_th = getTargetSpecificThreshold();
  ROS_INFO("[ObjectMap2D] Target category set to '%s', specific threshold: %.2f",
           category.c_str(), specific_th);
}

bool ObjectMap2D::isConfirmedTarget(int obj_idx) const
{
  if (obj_idx < 0 || obj_idx >= static_cast<int>(objects_.size()))
    return false;

  double tau_t = getCurrentConfidenceThreshold();
  return objects_[obj_idx].fused_confidence_ >= tau_t;
}

bool ObjectMap2D::hasConfirmedTarget() const
{
  double tau_t = getCurrentConfidenceThreshold();

  for (const auto& obj : objects_) {
    if (obj.fused_confidence_ >= tau_t)
      return true;
  }
  return false;
}

void ObjectMap2D::getConfirmedTargetClouds(
    std::vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>>& confirmed_clouds)
{
  confirmed_clouds.clear();
  double tau_t = getCurrentConfidenceThreshold();

  for (const auto& obj : objects_) {
    // 动态判断：只返回确信目标
    if (obj.fused_confidence_ < tau_t)
      continue;

    if (obj.best_label_ < 0 || obj.best_label_ >= static_cast<int>(obj.clouds_.size()))
      continue;

    if (obj.clouds_[obj.best_label_] && !obj.clouds_[obj.best_label_]->empty()) {
      confirmed_clouds.push_back(obj.clouds_[obj.best_label_]);
    }
  }
}
```

#### 1.5.3 修改 `searchSingleObjectCluster()` (约第165行后)

在成功聚类后更新融合置信度:

```cpp
int ObjectMap2D::searchSingleObjectCluster(const DetectedObject& detected_object)
{
  // ... 现有代码保持不变，直到 obj_idx 确定后 ...

  // ====== 更新融合置信度 (在 updateObjectBestLabel 之后添加) ======
  if (obj_idx != -1) {
    auto& obj = objects_[obj_idx];

    // 更新融合置信度（使用 best_label 的置信度）
    if (obj.best_label_ >= 0) {
      obj.fused_confidence_ = obj.confidence_scores_[obj.best_label_];
    }
  }

  // ... 现有代码继续 ...
}
```

#### 1.5.4 修改 `getTopConfidenceObjectCloud()` (约第571行)

使用动态阈值判断:

```cpp
void ObjectMap2D::getTopConfidenceObjectCloud(
    vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>>& top_object_clouds,
    bool limited_confidence, bool extreme)
{
  top_object_clouds.clear();
  vector<ObjectCluster> top_objects;
  double tau_t = getCurrentConfidenceThreshold();  // 获取当前动态阈值

  if (limited_confidence) {
    for (const auto& object : objects_) {
      // ====== 使用动态阈值判断 ======
      if (object.fused_confidence_ < tau_t)
        continue;

      // ... 现有过滤逻辑继续 ...
      int max_func_score = 0, best_label = -1;
      for (int label = 0; label < (int)object.clouds_.size(); label++) {
        auto obs_sum = object.observation_cloud_sums_[label];
        auto score = object.confidence_scores_[label];
        int func_score = obs_sum * score;
        if (func_score > max_func_score) {
          max_func_score = func_score;
          best_label = label;
        }
      }

      if (best_label == 0 && isConfidenceObject(object))
        top_objects.push_back(object);
    }
  }
  // ... 其余代码保持不变 ...
}
```

#### 1.5.5 修改 `clearObjects()` (约第55行)

重置探索步数:

```cpp
void ObjectMap2D::clearObjects()
{
  // 记录清理的物体数量
  size_t num_objects = objects_.size();

  // 清理所有物体
  objects_.clear();

  // 重置探索步数
  current_step_ = 0;

  // ... 现有清理代码 ...

  ROS_INFO("[ObjectMap2D] Cleared %zu objects for new episode", num_objects);
}
```

---

## 第二部分：ExplorationFSM 集成

### 2.1 在 FSM 回调中更新探索步数

**文件**: `src/planner/exploration_manager/src/exploration_fsm.cpp`

修改 `updateFrontierAndObject()` (约第589行):

```cpp
bool ExplorationFSM::updateFrontierAndObject()
{
  // ... 现有代码 ...

  t0 = ros::Time::now();

  // ====== 新增：更新探索步数 ======
  obj_map->setCurrentStep(fd_->step_count_);  // 假设 fd_ 中有 step_count_

  // ====== 检查是否有确信目标 ======
  if (obj_map->hasConfirmedTarget()) {
    ROS_INFO("[FSM] Found confirmed target at step %d (tau=%.3f)",
             fd_->step_count_, obj_map->getCurrentConfidenceThreshold());
    // 切换到目标导航状态
  }

  // 获取物体 (现在会使用动态阈值判断)
  obj_map->getObjects(ed->objects_, ed->object_averages_, ed->object_labels_);
  double t_get_objects = (ros::Time::now() - t0).toSec();

  // ... 现有代码继续 ...
}
```

### 2.2 设置当前目标类别

在 episode 开始时设置目标类别：

```cpp
void ExplorationFSM::startNewEpisode(const std::string& target_category)
{
  // ... 现有代码 ...

  // 设置目标类别以应用特定阈值
  obj_map->setTargetCategory(target_category);
}
```

---

## 第三部分：配置文件修改

### 3.1 添加新参数到 launch 文件

**文件**: `src/planner/exploration_manager/launch/exploration.launch` 或 `algorithm.xml`

```xml
<!-- ====== Patience-Aware Navigation Parameters (论文 Section 3.5) ====== -->
<param name="object/tau_high" value="0.4"/>
<param name="object/tau_low" value="0.2"/>
<param name="object/T_max" value="500"/>
```

### 3.2 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `tau_high` | 0.4 | 初始严格阈值（探索开始时） |
| `tau_low` | 0.2 | 最终宽松阈值（探索结束时） |
| `T_max` | 500 | 最大探索步数（与 episode 长度对应） |

---

## 第四部分：文件修改总结

| 操作 | 文件 | 修改内容 |
|------|------|----------|
| 修改 | `plan_env/include/plan_env/object_map2d.h` | 添加 `fused_confidence_` 成员、动态阈值相关方法 |
| 修改 | `plan_env/src/object_map2d.cpp` | 实现时间动态阈值、目标特定阈值 |
| 修改 | `exploration_manager/src/exploration_fsm.cpp` | 集成探索步数更新、确信目标检查 |
| 修改 | `exploration_manager/launch/algorithm.xml` | 添加新参数 |

---

## 第五部分：不需要修改的文件

以下文件 **保留不变**：

| 文件 | 原因 |
|------|------|
| `vlm/itm/blip2itm.py` | Multi-Source Value Map 需要使用 |
| `vlm/itm/blip2itm_client.py` | Multi-Source Value Map 需要使用 |
| `vlm/utils/get_itm_message.py` | 可能被 Habitat 侧使用 |
| `plan_env/scripts/multi_source_itm_node.py` | 语义分数桥接节点 |
| `plan_env/src/multi_source_value_map.cpp` | 语义价值图构建 |
| `plan_env/src/map_ros.cpp` 中的 ITM 相关代码 | 可能被 Value Map 使用 |

---

## 第六部分：测试检查清单

- [ ] 编译通过，无新增警告
- [ ] `ObjectCluster` 新成员 `fused_confidence_` 正确初始化
- [ ] `getCurrentConfidenceThreshold()` 随步数正确变化
  - [ ] step=0 时返回 0.4
  - [ ] step=250 时返回 0.3
  - [ ] step=500 时返回 0.2
- [ ] 动态判断逻辑正确
  - [ ] `isConfirmedTarget()` 使用当前 τ(t) 判断
  - [ ] `hasConfirmedTarget()` 使用当前 τ(t) 判断
  - [ ] `getConfirmedTargetClouds()` 使用当前 τ(t) 过滤
- [ ] 目标特定阈值生效
  - [ ] 小物体使用更低阈值
- [ ] 整体导航流程正常：探索 → 检测 → 动态判断 → 导航

---

## 第七部分：预期行为流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Patience-Aware Navigation 流程                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  【检测阶段】                                                         │
│  检测到物体，时序融合后置信度 = c                                      │
│  将物体及其置信度存入 Object Map                                       │
│                                                                     │
│  【查询阶段】（每帧执行）                                               │
│       │                                                             │
│       ▼                                                             │
│  计算当前阈值 τ(t) = τ_high - (t/T_max) × (τ_high - τ_low)           │
│       │                                                             │
│       ▼                                                             │
│  遍历 Object Map 中所有物体                                           │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  物体置信度 c ≥ τ(t) ?                                   │        │
│  │     │                                                    │        │
│  │  Yes ▼                                                   │        │
│  │  该物体是确信目标 ────► 导航过去                          │        │
│  │                                                          │        │
│  │  No ▼                                                    │        │
│  │  该物体是可疑目标 ────► 继续探索，等待 τ(t) 下降           │        │
│  └─────────────────────────────────────────────────────────┘        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      可疑目标自动升级示例                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  t=100: 检测到物体 A, confidence=0.35                               │
│         τ(100) = 0.36                                               │
│         0.35 < 0.36 → A 是可疑目标，继续探索                         │
│                                                                     │
│  t=150: 继续探索...                                                  │
│         τ(150) = 0.34                                               │
│         0.35 ≥ 0.34 → A 现在是确信目标！                             │
│         导航到 A                                                     │
│                                                                     │
│  【关键】：A 的置信度没变，但 τ(t) 下降了，所以 A 自动"升级"          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      阈值随时间变化示意                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  τ(t)                                                               │
│    ▲                                                                │
│ 0.4│ ▬▬▬▬▬▬▬▬●                                                      │
│    │           ╲                                                    │
│ 0.3│            ╲                                                   │
│    │             ╲                                                  │
│ 0.2│              ╲▬▬▬▬▬▬●                                          │
│    │                                                                │
│    └─────────────────────────────────────────────►  t (探索步数)     │
│         0       250       500                                       │
│                                                                     │
│    早期：高阈值 → 减少假阳性                                          │
│    后期：低阈值 → 提高召回率                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 设计优点

1. **简洁**：Object Map 只存储置信度，不存储分类状态
2. **动态**：每次查询时用当前 τ(t) 判断，无需维护状态
3. **自动升级**：可疑目标随时间自动变成确信目标，无需主动回访
4. **符合论文**：准确实现论文 Section 3.5 的 Patience-Aware 机制

---

*文档生成时间: 2026-01-19*
*项目: ApexNav/InfoNav*
*对应论文: Section 3.5 Patience-Aware Navigation Mechanism*
