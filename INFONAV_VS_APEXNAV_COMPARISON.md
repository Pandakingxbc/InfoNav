# InfoNav vs ApexNav 对比分析报告

> 生成时间: 2024
> 目的: 分析两个系统的架构差异，找出 InfoNav 假阳性率高、成功率低的潜在原因

---

## 一、架构概览

| 特性 | InfoNav | ApexNav |
|------|---------|---------|
| **定位** | 研究版本（复杂验证机制） | 简化版本（核心导航功能） |
| **VLM 验证** | 可选功能（默认关闭） | 无 |
| **置信度模型** | 动态 Patience-Aware τ(t) | 静态 min_confidence_ |
| **假阳性处理** | markObjectAsInvalid + 重规划 | 无专门处理 |
| **目标锁定** | VLM锁定 + 可疑目标锁定 | 无 |

### 1.1 VLM 验证是可选功能

**重要澄清**: VLM 验证在 InfoNav 中是**可配置的可选功能**，默认关闭：

```xml
<!-- launch/algorithm.xml -->
<arg name="vlm_validation_enabled_" default="false"/>
<param name="vlm_validation/enabled" value="$(arg vlm_validation_enabled_)" type="bool"/>
```

```cpp
// exploration_fsm.cpp:57
nh.param("vlm_validation/enabled", vlm_validation_enabled_, false);  // 默认 false
```
**如果 VLM 禁用**，则：
- 不会执行 VLM 滑动窗口验证
- 不会执行多视角验证
- 不会调用 `markObjectAsInvalid()`
- 行为更接近 ApexNav

---

## 二、状态机对比

### 2.1 主状态机（两者相同）

```
INIT → WAIT_TRIGGER → PLAN_ACTION → PUB_ACTION → WAIT_ACTION_FINISH → FINISH
```

### 2.2 InfoNav VLM 验证状态机（仅 VLM 启用时生效）

```cpp
enum class VLMValidationState {
  IDLE,              // 未在验证模式
  NAVIGATING,        // 导航到观察视点
  ROTATING,          // 旋转面向物体
  CAPTURING,         // 捕获图像用于 VLM
  WAITING_RESPONSE,  // 等待 VLM 响应
  COMPLETED          // 验证完成
};
```

---

## 三、Patience-Aware 动态阈值机制详解

### 3.1 设计意图

**你的理解是正确的**：动态阈值的设计目的是：
1. **不是拒绝检测**，而是**延迟导航决策**
2. 物体仍然被记录在 object map 中
3. 只是阈值未达到时不会作为导航目标
4. 让智能体先遍历周围环境，收集更多信息后再做综合决策
5. 随着时间推移阈值降低，之前被"暂缓"的物体可以成为目标

### 3.2 工作流程

```
检测到物体 (confidence = 0.35)
    │
    ▼
┌─────────────────────────────────────────┐
│  Step 1: 记录到 Object Map              │
│  - 创建/合并 ObjectCluster              │
│  - 存储 confidence_scores_[label]       │
│  - 更新 fused_confidence_               │
│  - 物体信息被保留！                      │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Step 2: 导航目标选择时检查阈值          │
│  τ(t) = τ_high - (t/T_max) × (τ_high - τ_low) │
│                                         │
│  t=0:   τ(0) = 0.45, conf=0.35 < τ      │
│         → 不作为导航目标（但仍在map中）   │
│                                         │
│  t=200: τ(200) = 0.35, conf=0.35 >= τ   │
│         → 可以作为导航目标               │
└─────────────────────────────────────────┘
```

### 3.3 关键代码证据

```cpp
// object_map2d.cpp:761-800 - getTopConfidenceObjectCloud
double tau_t = getCurrentConfidenceThreshold();

for (auto object : objects_) {  // 遍历所有已记录的物体
  // ...
  // Tier 1: best_label == 0 且置信度达标
  if (best_label == 0 && isConfidenceObject(object)) {
    shouldInclude = true;  // 作为导航目标
  }
  // Tier 2: best_label != 0 但 conf[0] 仍然很高
  else if (object.confidence_scores_[0] >= 1.2 * tau_t) {
    shouldInclude = true;  // 仍可作为导航目标
  }
}
```

---

## 四、🚨 关键问题：best_label 覆盖问题

### 4.1 问题描述

**你的担忧是正确的！** 这是一个潜在的严重问题：

当一个 object map location 维护多个 object 类别时，`best_label_` 会被设置为**得分最高的类别**，而不是目标类别 (label=0)。

### 4.2 问题代码

```cpp
// object_map2d.cpp:376-393
void ObjectMap2D::updateObjectBestLabel(int obj_idx)
{
  double max_func_score = 0.1;
  int best_label = -1;

  // 遍历所有类别，选择得分最高的
  for (int label = 0; label < (int)objects_[obj_idx].clouds_.size(); label++) {
    auto obs_sum = objects_[obj_idx].observation_cloud_sums_[label];
    auto score = objects_[obj_idx].confidence_scores_[label];
    int func_score = obs_sum * score;  // 功能得分 = 观测次数 × 置信度

    if (func_score > max_func_score) {
      max_func_score = func_score;
      best_label = label;  // 最高分的类别成为 best_label
    }
  }
  objects_[obj_idx].best_label_ = best_label;  // 可能不是目标类别!
}
```

### 4.3 问题场景

```
场景: 目标是 "chair" (label=0)，但检测器也检测到 "table" (label=1)

时间 t1: 检测到 chair, conf[0]=0.4, obs[0]=5
         检测到 table, conf[1]=0.6, obs[1]=3

         func_score[0] = 5 × 0.4 = 2.0
         func_score[1] = 3 × 0.6 = 1.8

         best_label = 0 (chair) ✓ 正确

时间 t2: 更多 table 检测
         conf[0]=0.4, obs[0]=5
         conf[1]=0.7, obs[1]=10

         func_score[0] = 5 × 0.4 = 2.0
         func_score[1] = 10 × 0.7 = 7.0

         best_label = 1 (table) ✗ 目标被覆盖!
```

### 4.4 后果分析

```cpp
// object_map2d.cpp:783-788 - Tier 1 检查
if (best_label == 0 && isConfidenceObject(object)) {
  shouldInclude = true;  // 只有 best_label==0 才进入 Tier 1
}
```

**如果 `best_label` 被其他类别覆盖**：
1. ❌ Tier 1 条件 `best_label == 0` 失败
2. ⚠️ 只能依赖 Tier 2：`conf[0] >= 1.2 × τ(t)`
3. 如果 `conf[0]` 不够高（< 1.2 × τ），该物体**永远不会成为导航目标**
4. 即使后续阈值 τ(t) 降低，Tier 1 仍然会失败

### 4.5 这就是假阳性/低成功率的重要原因！

```
实际是目标物体 (chair)
    │
    ▼
被检测器同时标记为 chair 和 table
    │
    ▼
table 的 func_score 更高
    │
    ▼
best_label = table (不是目标)
    │
    ▼
Tier 1 失败 (best_label != 0)
    │
    ▼
Tier 2 检查: conf[0] >= 1.2 × τ(t) ?
    │
    ├─ 是 → 仍可作为目标（但优先级降低）
    │
    └─ 否 → 永远不会成为目标！← 这就是问题
```

---

## 五、假阳性处理机制对比

### 5.1 InfoNav 的多层假阳性处理

```
检测物体
    │
    ▼
┌─────────────────────────────────────────┐
│  Layer 1: 记录到 Object Map             │
│  - 所有检测都被记录                      │
│  - 不会因为阈值而丢弃                    │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Layer 2: Patience-Aware 动态阈值 τ(t)   │
│  - 决定是否作为导航目标                  │
│  - 不满足阈值 = 暂不导航（不是删除）      │
└─────────────────────────────────────────┘
    │ 通过阈值
    ▼
┌─────────────────────────────────────────┐
│  Layer 3: 可疑目标锁定（无前沿时）        │
│  - 防止目标间振荡                        │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Layer 4: VLM 验证（如果启用）           │
│  - 默认关闭                             │
│  - 启用时进行多视角验证                  │
└─────────────────────────────────────────┘
```

### 5.2 ApexNav 的处理

```
检测物体
    │
    ▼
┌─────────────────────────────────────────┐
│  仅: 静态阈值过滤                        │
│  if (confidence >= min_confidence_)     │
│      → 接受并导航                        │
└─────────────────────────────────────────┘
```

### 5.3 关键差异表

| 机制 | InfoNav | ApexNav |
|------|---------|---------|
| **VLM 验证** | 可选（默认关闭） | 无 |
| **动态阈值 τ(t)** | ✅ 有 | ❌ 无 |
| **物体暂存** | ✅ 先记录后决策 | ❌ 即时决策 |
| **best_label 覆盖** | ⚠️ 存在问题 | ⚠️ 可能也有 |
| **Tier 2 回退** | ✅ 有 | ❌ 无 |

---

## 六、卡住处理对比

### 6.1 共同参数

| 参数 | 值 | 作用 |
|------|-----|------|
| `STUCKING_DISTANCE` | 0.05m | 判定卡住的位移阈值 |
| `MAX_STUCKING_COUNT` | 25 | 全局卡住次数 → 终止 episode |
| `MAX_STUCKING_NEXT_POS_COUNT` | 14 | 同目标卡住次数 → 休眠 frontier |
| `FORCE_DORMANT_DISTANCE` | 0.35m | 强制休眠距离 |
| `FRONTIER_TIMER_DURATION` | 0.25s | 前沿更新频率（已统一） |

---

## 七、问题总结与建议

### 7.1 已确认的问题

| 问题 | 严重性 | 描述 |
|------|--------|------|
| **best_label 覆盖** | 🔴 高 | 目标物体可能被其他类别覆盖，导致永远不被选为导航目标 |
| **τ_low 可能过高** | 🟡 中 | 即使阈值降到最低，Tier 2 的 1.2 倍要求可能仍然过高 |
| **Tier 2 条件严格** | 🟡 中 | 需要 `conf[0] >= 1.2 × τ(t)`，可能错过真阳性 |

### 7.2 建议修改

#### 建议 1: 降低 τ_low（你提到的）

```cpp
// 当前配置 (algorithm.xml)
<param name="object/tau_high" value="0.45" type="double"/>
<param name="object/tau_low" value="0.25" type="double"/>

// 建议修改
<param name="object/tau_high" value="0.40" type="double"/>
<param name="object/tau_low" value="0.15" type="double"/>  // 降低最低阈值
```

#### 建议 2: 修复 best_label 覆盖问题

**方案 A: 在目标选择时直接检查 conf[0]，忽略 best_label**

```cpp
// 修改 getTopConfidenceObjectCloud 的 Tier 1 逻辑
// 原代码:
if (best_label == 0 && isConfidenceObject(object)) {
  shouldInclude = true;
}

// 建议修改为:
// Tier 1: 检查目标类别置信度，不依赖 best_label
if (object.confidence_scores_[0] >= tau_t &&
    object.observation_nums_[0] >= min_observation_num_) {
  shouldInclude = true;
  ROS_WARN("[TopConfidence] id=%d TIER1-FIXED: conf[0]=%.3f >= tau=%.3f (best_label=%d)",
           object.id_, object.confidence_scores_[0], tau_t, best_label);
}
```

**方案 B: 降低 Tier 2 的倍数要求**

```cpp
// 原代码: 1.2 倍
else if (object.confidence_scores_[0] >= 1.2 * tau_t) {

// 建议修改: 1.0 倍（与 Tier 1 相同）
else if (object.confidence_scores_[0] >= tau_t) {
```

#### 建议 3: 添加专门的目标类别检查

```cpp
// 新增函数：直接检查目标类别置信度
bool ObjectMap2D::isTargetConfident(const ObjectCluster& obj) {
  double tau_t = getCurrentConfidenceThreshold();
  // 直接检查 label=0 的置信度，不管 best_label 是什么
  return obj.confidence_scores_[0] >= tau_t &&
         obj.observation_nums_[0] >= min_observation_num_;
}
```

### 7.3 验证方法

添加调试日志来确认问题：

```cpp
// 在 getTopConfidenceObjectCloud 中添加
for (auto object : objects_) {
  ROS_INFO("[DEBUG] Object id=%d: best_label=%d, conf[0]=%.3f, conf[best]=%.3f, tau=%.3f",
           object.id_, best_label,
           object.confidence_scores_[0],
           object.confidence_scores_[best_label],
           tau_t);

  // 检查是否存在 best_label 覆盖问题
  if (best_label != 0 && object.confidence_scores_[0] > 0.1) {
    ROS_WARN("[POTENTIAL ISSUE] Object id=%d: target conf=%.3f but best_label=%d",
             object.id_, object.confidence_scores_[0], best_label);
  }
}
```

---

## 八、结论

### 8.1 你的理解是正确的

1. ✅ VLM 是可选功能，默认关闭
2. ✅ 动态阈值 τ(t) 不是拒绝检测，而是延迟导航决策
3. ✅ 物体被记录在 object map 中，等待阈值满足
4. ✅ 设计意图是让智能体先探索环境再做决策

### 8.2 发现的关键问题

🔴 **best_label 覆盖问题是真实存在的**：
- 当同一位置检测到多个类别时
- 如果其他类别的 `func_score`（观测次数 × 置信度）更高
- `best_label` 会被设置为非目标类别
- 导致 Tier 1 条件失败，只能依赖 Tier 2
- 如果 Tier 2 也失败，该目标**永远不会被选中**

### 8.3 推荐的修改优先级

1. **高优先级**: 修复 Tier 1 逻辑，直接检查 `conf[0]` 而非依赖 `best_label`
2. **中优先级**: 降低 `τ_low` 到 0.15
3. **中优先级**: 降低 Tier 2 的 1.2 倍系数到 1.0
4. **低优先级**: 添加调试日志确认问题频率

---

## 九、关键代码位置索引

| 功能 | 文件 | 行号 |
|------|------|------|
| VLM 启用开关 | exploration_fsm.cpp | 57 |
| VLM 默认值 | algorithm.xml | 15 |
| best_label 更新 | object_map2d.cpp | 376-393 |
| Tier 1/2 过滤 | object_map2d.cpp | 783-796 |
| 动态阈值计算 | object_map2d.cpp | 928-933 |
| tau_high/tau_low 配置 | algorithm.xml | 82-83 |
| fused_confidence 更新 | object_map2d.cpp | 355-362 |
