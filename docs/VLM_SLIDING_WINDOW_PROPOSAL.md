# VLM 滑动窗口验证方案

> 作者: Claude Code Assistant
> 日期: 2026-01-24
> 版本: v1.1 (已实现)
> 状态: ✅ 已实现并编译通过

## 1. 问题分析

### 1.1 当前系统参数

| 参数 | 值 | 位置 |
|------|-----|------|
| `REACH_DISTANCE` | 0.20m | `exploration_fsm.h:55` |
| `SOFT_REACH_DISTANCE` | 0.45m | `exploration_fsm.h:56` |

**当前验证触发逻辑**: 距离目标 < 0.20m 时触发VLM验证

**建议的VLM触发距离**: 0.40m ~ 0.50m (REACH_DISTANCE × 2)

### 1.2 核心问题

在前往目标物体的路途中：
1. 检测器可能检测到**多个同类物体**（例如多把椅子、多个植物）
2. 这些物体可能**不是我们要去的目标**，但也会被检测出来
3. 如果仅基于"检测到目标"就触发VLM，可能会**误验证途中的物体**
4. 需要一种机制来**区分途中物体和目标物体**

### 1.3 解决思路：滑动窗口 + 距离感知

```
路径示意图:

Robot ----[途中物体A]----[途中物体B]----[目标物体]
  |           |              |              |
  |     检测到但不验证   检测到但不验证    触发VLM验证
  |           |              |              |
  └───────────┴──────────────┴──────────────┘
              滑动窗口收集高置信度图片
```

---

## 2. 滑动窗口设计方案

### 2.1 核心概念

```cpp
struct VLMCandidateFrame {
    cv::Mat image;                    // RGB图像
    ros::Time timestamp;              // 时间戳
    double detection_confidence;      // 检测器置信度
    double distance_to_target;        // 到目标object的距离
    Eigen::Vector2d robot_position;   // 机器人位置
    double robot_yaw;                 // 机器人朝向
    int detected_object_id;           // 检测到的物体ID（来自object_map）
};

class VLMSlidingWindow {
    std::deque<VLMCandidateFrame> frames_;
    size_t max_frames_ = 3;           // 最多保存3帧
    double min_confidence_ = 0.3;     // 最低检测置信度
    double max_distance_ = 2.0;       // 最大收集距离
    double min_distance_ = 0.4;       // 触发验证距离
};
```

### 2.2 滑动窗口工作流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SEARCH_OBJECT 状态主循环                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  每一步 (step):                                                             │
│                                                                             │
│  1. 计算当前位置到目标object的距离 dist                                       │
│                                                                             │
│  2. 如果 dist > max_distance (2.0m):                                        │
│     └─> 正常导航，不收集图片                                                  │
│                                                                             │
│  3. 如果 max_distance >= dist > min_distance (2.0m ~ 0.4m):                 │
│     └─> 收集阶段                                                            │
│         ├─ 检查检测器是否检测到目标 (label=0)                                 │
│         ├─ 检查检测到的物体是否是当前目标object (通过位置匹配)                  │
│         ├─ 如果匹配且置信度 > min_confidence:                                │
│         │   └─ 将当前帧加入滑动窗口                                          │
│         │       ├─ 如果窗口已满，移除最旧/最低置信度的帧                       │
│         │       └─ 保留置信度最高的 N 帧                                     │
│         └─ 继续导航                                                         │
│                                                                             │
│  4. 如果 dist <= min_distance (0.4m):                                       │
│     └─> 验证阶段                                                            │
│         ├─ 停止移动                                                         │
│         ├─ 从滑动窗口中选择最佳 1-2 帧                                       │
│         ├─ 调用 VLM 服务验证                                                │
│         ├─ 如果验证通过:                                                    │
│         │   └─ 锁定目标，返回 REACH_OBJECT                                  │
│         └─ 如果验证失败:                                                    │
│             └─ 加入黑名单，清空窗口，重新规划                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 关键判断：区分途中物体 vs 目标物体

**问题**: 检测器检测到目标类别的物体，但如何判断它是不是我们要去的那个？

**解决方案**: 利用 Object Map 中的空间信息

```cpp
bool isDetectionFromTargetObject(
    const Vector2d& detection_position,    // 检测到的物体位置（从点云计算）
    const Vector2d& target_object_center,  // 目标object的中心位置
    double tolerance = 0.5                 // 容差（米）
) {
    // 如果检测到的物体位置与目标object位置接近，认为是同一个物体
    return (detection_position - target_object_center).norm() < tolerance;
}
```

**更精确的方案**: 使用 object_map 的 ID 匹配

```cpp
bool isDetectionFromTargetObject(
    int detected_object_id,       // 从检测点云查询到的 object_map ID
    int target_object_id          // 当前导航目标的 object_map ID
) {
    return detected_object_id == target_object_id;
}
```

---

## 3. 代码修改清单

### 3.1 新增文件

| 文件 | 描述 |
|------|------|
| `exploration_fsm.h` | 添加滑动窗口相关成员变量 |
| `exploration_fsm.cpp` | 添加滑动窗口逻辑 |

### 3.2 修改 `exploration_fsm.h`

**位置**: `exploration_fsm.h:114-140` 附近（VLM Validation 区域）

```cpp
// ==================== VLM Sliding Window Validation ====================

/// VLM验证触发距离（建议 REACH_DISTANCE × 2 = 0.4m）
static constexpr double VLM_TRIGGER_DISTANCE = 0.40;
/// 开始收集候选帧的距离
static constexpr double VLM_COLLECT_DISTANCE = 3.0;
/// 滑动窗口最大帧数
static constexpr int VLM_WINDOW_MAX_FRAMES = 3;
/// 最低检测置信度阈值
static constexpr double VLM_MIN_DETECTION_CONFIDENCE = 0.3;

/// 候选帧结构
struct VLMCandidateFrame {
    ros::Time timestamp;
    double detection_confidence;
    double distance_to_target;
    Eigen::Vector2d robot_position;
    double robot_yaw;
    int detected_object_id;
    // 注意：图像存储在Python端，这里只存元信息
    // 通过timestamp可以从Python端检索对应图像
};

/// 滑动窗口
std::deque<VLMCandidateFrame> vlm_candidate_frames_;

/// VLM目标锁定状态
bool vlm_target_locked_;
Eigen::Vector2d vlm_locked_target_pos_;
int vlm_locked_object_id_;

/// 滑动窗口管理函数
void collectVLMCandidateFrame();
void clearVLMCandidateFrames();
bool selectBestFramesForValidation(std::vector<VLMCandidateFrame>& selected);
bool isDetectionFromTargetObject();
```

### 3.3 修改 `exploration_fsm.cpp` - 核心逻辑

**位置**: `callActionPlanner()` 函数中，在处理 `SEARCH_OBJECT` 状态的部分

#### 3.3.1 在 `init()` 中初始化

```cpp
// VLM Sliding Window initialization
vlm_target_locked_ = false;
vlm_locked_object_id_ = -1;
vlm_candidate_frames_.clear();
```

#### 3.3.2 在 `episodeResetCallback()` 中重置

```cpp
// Reset VLM sliding window state
vlm_target_locked_ = false;
vlm_locked_object_id_ = -1;
vlm_candidate_frames_.clear();
```

#### 3.3.3 修改 `callActionPlanner()` 中的 SEARCH_OBJECT 处理

```cpp
int ExplorationFSM::callActionPlanner()
{
    // ... 现有代码 ...

    // ==================== VLM Sliding Window Logic ====================
    // 只在 SEARCH_OBJECT 状态下执行
    if (fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT ||
        vlm_target_locked_) {  // 或者已经被VLM锁定

        double dist_to_target = (current_pos - expl_manager_->ed_->next_pos_).norm();

        // Case 1: 目标已被VLM锁定，直接前往
        if (vlm_target_locked_) {
            if (dist_to_target < FSMConstants::REACH_DISTANCE) {
                ROS_INFO("[VLM Lock] Reached VLM-locked target!");
                vlm_target_locked_ = false;
                vlm_candidate_frames_.clear();
                return FINAL_RESULT::REACH_OBJECT;
            }
            // 强制使用锁定的目标位置
            expl_manager_->ed_->next_pos_ = vlm_locked_target_pos_;
            // 继续正常导航逻辑...
            return FINAL_RESULT::SEARCH_OBJECT;
        }

        // Case 2: 在收集距离内，收集候选帧
        if (dist_to_target <= VLM_COLLECT_DISTANCE &&
            dist_to_target > VLM_TRIGGER_DISTANCE) {
            collectVLMCandidateFrame();
        }

        // Case 3: 达到触发距离，执行VLM验证
        if (dist_to_target <= VLM_TRIGGER_DISTANCE) {
            ROS_INFO("[VLM Trigger] Distance=%.2fm, triggering validation with %zu candidate frames",
                     dist_to_target, vlm_candidate_frames_.size());

            // 选择最佳帧进行验证
            std::vector<VLMCandidateFrame> selected_frames;
            bool has_frames = selectBestFramesForValidation(selected_frames);

            bool vlm_result = false;
            if (has_frames) {
                // 使用收集的帧进行验证
                vlm_result = performSlidingWindowVLMValidation(selected_frames);
            } else {
                // 没有收集到帧，使用当前帧
                ROS_WARN("[VLM] No candidate frames collected, using current frame");
                vlm_result = callVLMValidation();
            }

            if (vlm_result) {
                // VLM验证通过 -> 锁定目标
                vlm_target_locked_ = true;
                vlm_locked_target_pos_ = expl_manager_->ed_->next_pos_;
                vlm_locked_object_id_ = expl_manager_->object_map2d_->getCurrentTargetObjectId();
                ROS_INFO("[VLM PASS] Target LOCKED at (%.2f, %.2f), id=%d",
                         vlm_locked_target_pos_.x(), vlm_locked_target_pos_.y(),
                         vlm_locked_object_id_);
                vlm_candidate_frames_.clear();
            } else {
                // VLM验证失败 -> 加入黑名单
                int obj_id = expl_manager_->object_map2d_->getCurrentTargetObjectId();
                if (obj_id >= 0) {
                    expl_manager_->object_map2d_->markObjectAsInvalid(obj_id);
                    ROS_WARN("[VLM FAIL] Object id=%d added to blacklist", obj_id);
                }
                vlm_candidate_frames_.clear();
                fd_->replan_flag_ = true;
            }
        }
    }

    // ... 原有的代码继续 ...
}
```

#### 3.3.4 新增滑动窗口管理函数

```cpp
void ExplorationFSM::collectVLMCandidateFrame()
{
    // 检查检测器是否检测到目标
    boost::mutex::scoped_lock lock(detector_mutex_);
    if (!last_frame_has_target_) {
        return;
    }

    // 检查检测时间是否足够新
    double time_since_detection = (ros::Time::now() - last_detection_time_).toSec();
    if (time_since_detection > 0.5) {
        return;
    }

    // 检查检测到的是否是目标object（通过位置匹配）
    if (!isDetectionFromTargetObject()) {
        ROS_DEBUG("[VLM Collect] Detection not from target object, skipping");
        return;
    }

    // 创建候选帧
    VLMCandidateFrame frame;
    frame.timestamp = last_detection_time_;
    frame.detection_confidence = 0.5;  // TODO: 从检测消息获取实际置信度
    frame.distance_to_target = (fd_->start_pt_.head<2>() - expl_manager_->ed_->next_pos_).norm();
    frame.robot_position = fd_->start_pt_.head<2>();
    frame.robot_yaw = fd_->start_yaw_(0);
    frame.detected_object_id = expl_manager_->object_map2d_->getCurrentTargetObjectId();

    // 添加到窗口
    vlm_candidate_frames_.push_back(frame);

    // 如果超过最大帧数，移除最旧的
    while (vlm_candidate_frames_.size() > VLM_WINDOW_MAX_FRAMES) {
        vlm_candidate_frames_.pop_front();
    }

    ROS_INFO("[VLM Collect] Added candidate frame, window size=%zu, dist=%.2fm, conf=%.2f",
             vlm_candidate_frames_.size(), frame.distance_to_target, frame.detection_confidence);
}

bool ExplorationFSM::isDetectionFromTargetObject()
{
    // 方案1: 简单的距离判断
    // 如果当前位置到目标的距离在合理范围内，认为检测的是目标
    double dist = (fd_->start_pt_.head<2>() - expl_manager_->ed_->next_pos_).norm();

    // 在收集距离内的检测都认为是相关的
    // 更精确的判断需要从检测点云获取物体位置
    return dist <= VLM_COLLECT_DISTANCE;

    // 方案2: 更精确的判断（需要修改检测消息）
    // TODO: 从检测点云获取物体中心位置，与目标object位置比较
}

bool ExplorationFSM::selectBestFramesForValidation(
    std::vector<VLMCandidateFrame>& selected)
{
    selected.clear();

    if (vlm_candidate_frames_.empty()) {
        return false;
    }

    // 按置信度排序，选择最好的1-2帧
    std::vector<VLMCandidateFrame> sorted_frames(
        vlm_candidate_frames_.begin(), vlm_candidate_frames_.end());

    std::sort(sorted_frames.begin(), sorted_frames.end(),
              [](const VLMCandidateFrame& a, const VLMCandidateFrame& b) {
                  return a.detection_confidence > b.detection_confidence;
              });

    // 选择最多2帧
    int num_to_select = std::min(2, static_cast<int>(sorted_frames.size()));
    for (int i = 0; i < num_to_select; i++) {
        selected.push_back(sorted_frames[i]);
    }

    return !selected.empty();
}

void ExplorationFSM::clearVLMCandidateFrames()
{
    vlm_candidate_frames_.clear();
}
```

### 3.4 修改 Python VLM 服务

**文件**: `qwen_vlm_validation_node.py`

需要支持接收多帧图像进行验证。

#### 3.4.1 修改服务定义

**文件**: 新建或修改 `ValidateObject.srv`

```
# Request
string target_object
int32 num_views
sensor_msgs/Image[] candidate_images  # 新增：候选图像数组
---
# Response
bool is_valid
float32 confidence
string raw_response
int32 views_confirmed
int32 views_total
```

#### 3.4.2 修改验证逻辑

```python
def validate_with_multiple_images(self, images, target_object):
    """
    使用多张图像进行验证（AND逻辑）

    Args:
        images: List of numpy arrays (RGB format)
        target_object: target object name

    Returns:
        tuple: (is_valid, confidence, raw_response)
    """
    if not images:
        return False, 0.0, "No images provided"

    # 对每张图像进行验证
    all_valid = True
    total_confidence = 0.0
    responses = []

    for i, image in enumerate(images):
        is_valid, confidence, raw_response = self.validate_single_image(
            image, target_object
        )
        responses.append(f"Image {i+1}: {raw_response}")
        total_confidence += confidence

        if not is_valid:
            all_valid = False
            # AND逻辑：只要有一张失败就整体失败
            break

    avg_confidence = total_confidence / len(images) if images else 0.0
    combined_response = "\n---\n".join(responses)

    return all_valid, avg_confidence, combined_response
```

### 3.5 图像缓存机制

由于C++端只存储元信息，需要在Python端维护图像缓存。

**方案A**: 在Python端维护循环缓存

```python
class ImageRingBuffer:
    def __init__(self, max_size=50, max_age_seconds=10.0):
        self.buffer = {}  # timestamp -> image
        self.max_size = max_size
        self.max_age = max_age_seconds

    def add(self, timestamp, image):
        self.cleanup()
        self.buffer[timestamp] = image

    def get(self, timestamp, tolerance=0.1):
        """Get image by timestamp with tolerance"""
        for ts, img in self.buffer.items():
            if abs((ts - timestamp).to_sec()) < tolerance:
                return img
        return None

    def cleanup(self):
        now = rospy.Time.now()
        expired = [ts for ts in self.buffer
                   if (now - ts).to_sec() > self.max_age]
        for ts in expired:
            del self.buffer[ts]

        # 如果还是太多，删除最旧的
        while len(self.buffer) > self.max_size:
            oldest = min(self.buffer.keys())
            del self.buffer[oldest]
```

**方案B**: C++端通过ROS服务请求特定时间戳的图像

---

## 4. 配置参数

### 4.1 ROS 参数

```yaml
# launch文件或yaml配置
vlm_validation:
  enabled: true
  timeout: 30.0
  num_views: 2

  # 滑动窗口参数
  trigger_distance: 0.40        # 触发VLM验证的距离 (m)
  collect_distance: 2.0         # 开始收集候选帧的距离 (m)
  window_max_frames: 3          # 滑动窗口最大帧数
  min_detection_confidence: 0.3 # 最低检测置信度

  # 图像缓存参数 (Python端)
  image_buffer_size: 50         # 图像缓存大小
  image_max_age: 10.0           # 图像最大保留时间 (s)
```

### 4.2 常量定义

```cpp
// exploration_fsm.h 中的 FSMConstants 命名空间
namespace FSMConstants {
    // ... 现有常量 ...

    // VLM Sliding Window
    constexpr double VLM_TRIGGER_DISTANCE = 0.40;    // 2 × REACH_DISTANCE
    constexpr double VLM_COLLECT_DISTANCE = 2.0;
    constexpr int VLM_WINDOW_MAX_FRAMES = 3;
    constexpr double VLM_MIN_DETECTION_CONFIDENCE = 0.3;
}
```

---

## 5. 测试要点

### 5.1 单元测试

- [ ] 滑动窗口添加/删除帧
- [ ] 帧选择算法（按置信度排序）
- [ ] 图像缓存过期清理
- [ ] 目标物体匹配判断

### 5.2 集成测试

- [ ] 单目标场景：正常收集、验证、锁定流程
- [ ] 多目标场景：途中有同类物体，验证只针对目标
- [ ] 验证失败场景：正确加入黑名单，重新规划
- [ ] Episode重置：状态正确清理

### 5.3 边界条件

- [ ] 窗口为空时触发验证（回退到当前帧）
- [ ] 快速移动导致收集帧过少
- [ ] 目标物体被遮挡导致检测间断

---

## 6. 实施顺序建议

1. **Phase 1**: 基础框架
   - [ ] 添加滑动窗口数据结构
   - [ ] 实现基本的收集逻辑（不含图像）
   - [ ] 实现触发和锁定逻辑

2. **Phase 2**: 图像处理
   - [ ] Python端图像缓存
   - [ ] 修改ROS服务支持多图像
   - [ ] C++和Python端的时间戳同步

3. **Phase 3**: 优化判断
   - [ ] 实现更精确的目标物体匹配
   - [ ] 优化帧选择策略
   - [ ] 添加置信度加权

4. **Phase 4**: 测试和调参
   - [ ] 各场景测试
   - [ ] 参数调优
   - [ ] 性能优化

---

## 7. 相关代码文件索引

| 文件 | 修改内容 | 优先级 |
|------|----------|--------|
| `exploration_fsm.h` | 添加滑动窗口成员和函数声明 | P0 |
| `exploration_fsm.cpp` | 核心逻辑实现 | P0 |
| `qwen_vlm_validation_node.py` | 多图像验证支持 | P1 |
| `ValidateObject.srv` | 服务定义扩展 | P1 |
| `object_map2d.h/cpp` | 已有黑名单支持，无需修改 | - |
| `exploration_manager.cpp` | 已有锁定支持，可能需微调 | P2 |

---

## 8. 风险和注意事项

### 8.1 性能考虑

- 图像缓存可能占用大量内存，需要合理设置缓存大小
- VLM调用是阻塞的，验证期间机器人停止可能影响效率

### 8.2 时序问题

- C++端收集的时间戳需要与Python端的图像缓存对齐
- 网络延迟可能导致图像检索失败

### 8.3 可靠性

- 如果滑动窗口为空，需要有回退策略（使用当前帧）
- VLM服务不可用时的降级处理

---

## 附录：状态转换图

```
                                    ┌──────────────┐
                                    │   EXPLORE    │
                                    └──────┬───────┘
                                           │ 发现高置信度物体
                                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│                          SEARCH_OBJECT                                  │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │ dist > 2.0m: 正常导航                                           │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                              │                                         │
│                              ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │ 2.0m >= dist > 0.4m: 收集候选帧（滑动窗口）                       │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                              │                                         │
│                              ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │ dist <= 0.4m: 触发VLM验证                                       │  │
│  │   ├─ 通过 → 锁定目标 → VLM_LOCKED状态                            │  │
│  │   └─ 失败 → 加入黑名单 → 重新规划                                │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌───────────────┐                          ┌───────────────┐
│  VLM_LOCKED   │                          │   重新规划     │
│  直达目标      │                          │  下一个目标    │
└───────┬───────┘                          └───────────────┘
        │ dist < 0.2m
        ▼
┌───────────────┐
│ REACH_OBJECT  │
└───────────────┘
```

---

## 9. 实现总结 (2026-01-24)

### 9.1 已实现的功能

| 功能 | 状态 | 说明 |
|------|------|------|
| 滑动窗口数据结构 | ✅ | `VLMCandidateFrame` 结构体 |
| 收集阶段锁定 | ✅ | 进入收集距离时锁定目标，防止切换 |
| 候选帧收集 | ✅ | `collectVLMCandidateFrame()` |
| 帧选择算法 | ✅ | `selectBestFramesForValidation()` |
| 目标锁定机制 | ✅ | `enterCollectionPhase()` / `exitCollectionPhase()` |
| Python图像缓存 | ✅ | `ImageRingBuffer` 类 |

### 9.2 关键设计决策

**早期锁定机制**: 一旦机器人进入收集距离（< 2.0m），立即锁定当前目标物体。
- 锁定后，即使 ObjectMap 中置信度更高的物体出现，也不会切换目标
- 确保VLM验证针对正确的物体进行

**强制目标位置**: 在锁定状态下：
```cpp
// 备份锁定位置
Vector2d locked_pos_backup = vlm_locked_target_pos_;
// 调用规划器获取路径
expl_res = expl_manager_->planNextBestPoint(...);
// 恢复锁定目标位置，忽略规划器的目标选择
expl_manager_->ed_->next_pos_ = locked_pos_backup;
```

### 9.3 修改的文件

| 文件 | 修改内容 |
|------|----------|
| `exploration_fsm.h` | 添加常量、数据结构、成员变量 |
| `exploration_fsm.cpp` | 核心逻辑实现、初始化/重置 |
| `qwen_vlm_validation_node.py` | `ImageRingBuffer` 图像缓存 |

### 9.4 新增常量

```cpp
namespace FSMConstants {
  constexpr double VLM_TRIGGER_DISTANCE = 0.40;     // 触发VLM验证
  constexpr double VLM_COLLECT_DISTANCE = 2.0;      // 开始收集帧/锁定目标
  constexpr int VLM_WINDOW_MAX_FRAMES = 3;          // 滑动窗口大小
  constexpr double VLM_MIN_DETECTION_CONFIDENCE = 0.3;
  constexpr double VLM_FRAME_MAX_AGE = 5.0;         // 帧最大保留时间
}
```

### 9.5 后续优化方向

1. **更精确的目标匹配**: 从检测点云获取3D位置，与目标物体位置比较
2. **置信度加权**: 根据检测置信度调整帧的优先级
3. **多帧VLM验证**: 将多个候选帧的时间戳传递给Python端，进行批量验证
