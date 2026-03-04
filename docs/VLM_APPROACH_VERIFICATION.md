# VLM Approach Verification (VLM 逼近验证)

## 概述

VLM Approach Verification 是一种在机器人逼近目标物体时触发的视觉语言模型验证机制。当机器人接近候选目标物体（距离 < 0.6m）时，系统会使用滑动窗口收集的高置信度检测帧调用 VLM API 进行二次验证，以区分真阳性和假阳性检测。

### 核心特性

1. **滑动窗口帧收集**: 只收集更新了当前导航目标物体的检测帧
2. **同步阻塞模式**: VLM 验证期间机器人原地等待，不输出任何动作
3. **动态 Top-K 选择**: 从滑动窗口选择 1-2 个最高置信度帧发送给 VLM
4. **置信度调整**: 根据 VLM 结果显著提升或降低物体置信度
5. **超时保护**: 30秒超时后禁用当前 episode 的 VLM 验证

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           C++ FSM (ExplorationFSM)                       │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────────┐ │
│  │ MapROS      │───>│ ObjectUpdateInfo │───>│ Python                  │ │
│  │ (检测回调)   │    │ 消息             │    │ VLMApproachVerifier     │ │
│  └─────────────┘    └──────────────────┘    │ (滑动窗口收集帧)         │ │
│                                             └─────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                    距离 < 0.6m 触发                                  ││
│  └─────────────────────────────────────────────────────────────────────┘│
│  ┌─────────────┐    ┌──────────────────────┐    ┌─────────────────────┐ │
│  │ ExplorationFSM│──>│ VLMVerificationRequest│──>│ VLMApproachVerifier │ │
│  │ (触发验证)   │    │ 消息                  │    │ (调用 VLM API)      │ │
│  └─────────────┘    └──────────────────────┘    └─────────────────────┘ │
│         │                                              │                 │
│         │ [等待，不输出动作]                            │ [VLM API 调用]  │
│         │                                              │                 │
│         v                                              v                 │
│  ┌─────────────┐    ┌──────────────────────┐    ┌─────────────────────┐ │
│  │ ExplorationFSM│<──│ VLMVerificationResult│<──│ VLMApproachVerifier │ │
│  │ (处理结果)   │    │ 消息                  │    │ (返回结果)          │ │
│  └─────────────┘    └──────────────────────┘    └─────────────────────┘ │
│         │                                                               │
│         v                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │  ObjectMap2D::applyVLMVerificationResult()                          ││
│  │  - 真阳性: 置信度提升到 ~0.85                                        ││
│  │  - 假阳性: 置信度降低到 ~0.05, 标记为无效                            ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

## 消息定义

### 1. VLMVerificationRequest.msg (C++ → Python)

```
# 验证请求
int32 object_id              # 目标物体 ID
string target_category       # 目标物体类别名称
int32 trigger_type           # 触发类型: 0=距离触发, 1=提前触发, 2=救援触发
float64 distance_to_target   # 当前距离目标的距离
std_msgs/Header header       # 消息头
```

### 2. VLMVerificationResult.msg (Python → C++)

```
# 验证结果
int32 object_id              # 验证的物体 ID
bool is_real_target          # 是否为真阳性
float64 vlm_confidence       # VLM 置信度 (0-1)
string reason                # VLM 回复的原因说明
bool timeout                 # 是否超时
float64 duration             # 验证耗时(秒)
std_msgs/Header header       # 消息头
```

### 3. ObjectUpdateInfo.msg (C++ → Python)

```
# 目标物体更新通知
int32 object_id              # 被更新的物体 ID
float64 detection_confidence # 本次检测置信度
float64 distance_to_object   # 当前距离目标的距离
bool is_current_target       # 是否是当前导航目标物体
std_msgs/Header header       # 消息头
```

## 核心参数

### FSM 常量 (exploration_fsm.h)

```cpp
namespace FSMConstants {
// VLM Approach Verification 参数
constexpr double VLM_APPROACH_TRIGGER_DISTANCE = 0.6;   // 距离触发阈值
constexpr double VLM_EARLY_TRIGGER_DISTANCE = 2.0;      // 提前触发最大距离
constexpr int VLM_EARLY_TRIGGER_MIN_FRAMES = 3;         // 提前触发最少帧数
constexpr double VLM_EARLY_TRIGGER_CONFIDENCE = 0.5;    // 提前触发最低置信度
constexpr double VLM_TIMEOUT_SECONDS = 30.0;            // VLM 超时时间
}
```

### VLM 置信度融合参数 (object_map2d.cpp)

```cpp
// 从 ROS 参数加载
nh.param("object/vlm_positive_weight", vlm_positive_weight_, 0.7);   // 正结果权重
nh.param("object/vlm_positive_boost", vlm_positive_boost_, 0.85);    // 正结果目标置信度
nh.param("object/vlm_negative_weight", vlm_negative_weight_, 0.8);   // 负结果权重
nh.param("object/vlm_negative_penalty", vlm_negative_penalty_, 0.05); // 负结果目标置信度
```

### Python 端参数 (vlm_approach_verifier.py)

```python
WINDOW_SIZE = 10           # 滑动窗口最大帧数
MIN_CONFIDENCE = 0.3       # 最低置信度阈值
TOP_K_FRAMES = 2           # 发送给 VLM 的帧数
MIN_FRAMES_FOR_VLM = 1     # VLM 验证最少帧数
VLM_TIMEOUT = 30.0         # VLM 超时时间
VLM_MODEL = "qwen-vl-max"  # 使用的 VLM 模型
```

## 文件修改清单

### 新增文件

| 文件路径 | 描述 |
|---------|------|
| `src/planner/plan_env/msg/VLMVerificationRequest.msg` | VLM 验证请求消息 |
| `src/planner/plan_env/msg/VLMVerificationResult.msg` | VLM 验证结果消息 |
| `src/planner/plan_env/msg/ObjectUpdateInfo.msg` | 目标物体更新通知消息 |
| `src/planner/plan_env/scripts/vlm_approach_verifier.py` | Python VLM 验证节点 |

### 修改文件

| 文件路径 | 修改内容 |
|---------|----------|
| `src/planner/plan_env/CMakeLists.txt` | 添加新消息文件和 Python 脚本 |
| `src/planner/plan_env/include/plan_env/object_map2d.h` | ObjectCluster 添加 VLM 字段，ObjectMap2D 添加 VLM 方法 |
| `src/planner/plan_env/src/object_map2d.cpp` | 实现 applyVLMVerificationResult 等方法 |
| `src/planner/plan_env/include/plan_env/map_ros.h` | 添加 ObjectUpdateInfo 发布者 |
| `src/planner/plan_env/src/map_ros.cpp` | 发布 ObjectUpdateInfo 消息 |
| `src/planner/exploration_manager/include/exploration_manager/exploration_data.h` | FSMData 添加 VLM 状态字段 |
| `src/planner/exploration_manager/include/exploration_manager/exploration_fsm.h` | 添加 VLM 相关常量、发布者、方法声明 |
| `src/planner/exploration_manager/src/exploration_fsm.cpp` | 实现 VLM 触发、等待、回调逻辑 |

## 工作流程详解

### 1. 帧收集阶段

当检测器检测到物体并更新 ObjectMap2D 时：

```cpp
// map_ros.cpp - detectedObjectCloudCallback()
if (current_nav_target_object_id_ >= 0) {
  for (size_t i = 0; i < detected_object_cluster_ids.size(); ++i) {
    int object_id = detected_object_cluster_ids[i];
    if (object_id == current_nav_target_object_id_) {
      // 发布 ObjectUpdateInfo
      plan_env::ObjectUpdateInfo update_msg;
      update_msg.object_id = object_id;
      update_msg.detection_confidence = detected_objects[i].score;
      update_msg.is_current_target = true;
      // ...
      object_update_info_pub_.publish(update_msg);
    }
  }
}
```

Python 端收集这些帧：

```python
# vlm_approach_verifier.py
def object_update_callback(self, msg: ObjectUpdateInfo):
    if not msg.is_current_target:
        return
    if msg.detection_confidence < self.min_confidence:
        return

    frame = CandidateFrame(
        timestamp=msg.header.stamp.to_sec(),
        object_id=msg.object_id,
        confidence=msg.detection_confidence,
        distance=msg.distance_to_object,
        image=self.latest_rgb_image.copy()
    )
    self.candidate_frames.append(frame)
```

### 2. 触发验证阶段

当机器人距离目标 < 0.6m 时：

```cpp
// exploration_fsm.cpp - callActionPlanner()
if (!vlm_collection_locked_ &&
    fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT &&
    dist_to_target < FSMConstants::VLM_APPROACH_TRIGGER_DISTANCE &&
    dist_to_target >= reach_distance) {
  int current_obj_id = expl_manager_->object_map2d_->getCurrentTargetObjectId();
  if (current_obj_id >= 0 && checkVLMTriggerConditions(dist_to_target, current_obj_id)) {
    // VLM 验证触发 - FSM 将等待结果
    return fd_->final_result_;
  }
}
```

触发验证请求：

```cpp
void ExplorationFSM::triggerVLMVerification(int object_id, int trigger_type,
                                             double distance_to_target) {
  // 设置等待状态 - FSM 停止输出动作
  fd_->vlm_waiting_ = true;
  fd_->vlm_target_object_id_ = object_id;

  // 发布验证请求
  plan_env::VLMVerificationRequest req_msg;
  req_msg.object_id = object_id;
  req_msg.target_category = fd_->target_category_;
  req_msg.trigger_type = trigger_type;
  req_msg.distance_to_target = distance_to_target;
  vlm_request_pub_.publish(req_msg);
}
```

### 3. FSM 等待阶段

```cpp
// exploration_fsm.cpp - FSMCallback()
case ROS_STATE::PLAN_ACTION: {
  // VLM 等待状态检查
  if (fd_->vlm_waiting_) {
    ROS_DEBUG_THROTTLE(1.0, "[VLM] Waiting for verification result, no action output");
    exec_timer_.start();
    return;  // 不发布任何动作
  }
  // ...正常规划逻辑
}
```

### 4. VLM API 调用

```python
# vlm_approach_verifier.py
def _call_vlm_api(self, frames: List[CandidateFrame],
                   target_category: str) -> Tuple[bool, float, str]:
    # 构建多图像消息
    content = []
    for frame in frames:
        if frame.image is not None:
            image_url = self._image_to_base64_url(frame.image)
            content.append({"type": "image_url", "image_url": {"url": image_url}})

    content.append({"type": "text", "text": prompt})

    # 调用 API
    completion = self.client.chat.completions.create(
        model=self.model_name,
        messages=[{"role": "user", "content": content}],
        extra_body={'enable_thinking': False}
    )

    # 解析结果
    response_text = completion.choices[0].message.content
    is_real, confidence, reason = self._parse_vlm_response(response_text, target_category)
    return is_real, confidence, reason
```

### 5. 结果处理阶段

```cpp
// exploration_fsm.cpp
void ExplorationFSM::vlmVerificationResultCallback(
    const plan_env::VLMVerificationResultConstPtr& msg) {

  // 检查超时
  if (msg->timeout) {
    fd_->vlm_disabled_this_episode_ = true;
    ROS_WARN("[VLM] Timeout! VLM disabled for rest of episode");
  }

  // 应用 VLM 结果到物体置信度
  if (expl_manager_ && expl_manager_->sdf_map_ &&
      expl_manager_->sdf_map_->object_map2d_) {
    expl_manager_->sdf_map_->object_map2d_->applyVLMVerificationResult(
        msg->object_id, msg->is_real_target, msg->vlm_confidence);
  }

  // 释放等待锁
  fd_->vlm_waiting_ = false;
  fd_->vlm_target_object_id_ = -1;
}
```

### 6. 置信度调整

```cpp
// object_map2d.cpp
void ObjectMap2D::applyVLMVerificationResult(int object_id, bool is_real_target,
                                              double vlm_confidence) {
  ObjectCluster& obj = objects_[object_id];
  double old_confidence = obj.fused_confidence_;

  if (is_real_target) {
    // 真阳性：大幅提升置信度
    // new_conf = (1 - 0.7) * old + 0.7 * 0.85
    double new_confidence = (1.0 - vlm_positive_weight_) * old_confidence +
                            vlm_positive_weight_ * vlm_positive_boost_;
    obj.fused_confidence_ = new_confidence;
    ROS_INFO("[VLM] Object %d CONFIRMED! Confidence: %.3f -> %.3f",
             object_id, old_confidence, new_confidence);
  } else {
    // 假阳性：大幅降低置信度并标记无效
    // new_conf = (1 - 0.8) * old + 0.8 * 0.05
    double new_confidence = (1.0 - vlm_negative_weight_) * old_confidence +
                            vlm_negative_weight_ * vlm_negative_penalty_;
    obj.fused_confidence_ = new_confidence;
    markObjectAsInvalid(object_id);
    ROS_WARN("[VLM] Object %d REJECTED! Confidence: %.3f -> %.3f, marked invalid",
             object_id, old_confidence, new_confidence);
  }

  obj.vlm_verified_ = true;
  obj.vlm_result_ = is_real_target;
}
```

## 使用方法

### 1. 编译

```bash
cd ~/Nav/InfoNav
catkin_make
source devel/setup.bash
```

### 2. 设置环境变量

```bash
# DashScope API Key (阿里云百炼平台)
export DASHSCOPE_API_KEY='sk-your-api-key-here'
```

### 3. 启动 VLM 验证节点

```bash
# 启动 VLM Approach Verifier 节点
rosrun plan_env vlm_approach_verifier.py
```

或者添加到 launch 文件：

```xml
<node pkg="plan_env" type="vlm_approach_verifier.py" name="vlm_approach_verifier" output="screen">
    <param name="rgb_topic" value="/map_ros/rgb"/>
    <param name="model" value="qwen-vl-max"/>
    <param name="timeout" value="30.0"/>
    <param name="window_size" value="10"/>
    <param name="min_confidence" value="0.3"/>
    <param name="top_k" value="2"/>
    <param name="debug_save_images" value="true"/>
</node>
```

### 4. 配置参数

在 launch 文件或参数服务器中配置：

```xml
<!-- VLM 置信度融合参数 -->
<param name="object/vlm_positive_weight" value="0.7"/>
<param name="object/vlm_positive_boost" value="0.85"/>
<param name="object/vlm_negative_weight" value="0.8"/>
<param name="object/vlm_negative_penalty" value="0.05"/>
```

## 调试信息

### 日志输出

VLM 验证过程会输出以下日志：

```
[VLMApproach] Collected frame: obj=3, conf=0.652, dist=1.23, window_size=5
[VLMApproach] Verification request: object_id=3, trigger_type=0, distance=0.58
[VLMApproach] Selected 2/5 frames, confidences: ['0.652', '0.589']
[VLMApproach] Calling VLM API with 2 image(s)...
[VLMApproach] VLM response: DECISION: YES, CONFIDENCE: 0.85, REASON: The chair is visible...
[VLMApproach] Published result: object_id=3, is_real=True, conf=0.850, timeout=False
[VLM] Object 3 CONFIRMED as real target! Confidence: 0.652 -> 0.850
```

### Episode 统计

每个 episode 结束时会输出 VLM 使用统计：

```
[VLM Stats] Episode Summary: used=true, disabled=false, verify_count=2
```

### Debug 文件

当 `debug_save_images=true` 时，验证图像和结果保存在：

```
~/Nav/InfoNav/debug/vlm_approach/
├── 20260207_143052_chair_YES_frame0_conf0.65.jpg
├── 20260207_143052_chair_YES_frame1_conf0.59.jpg
└── 20260207_143052_chair_YES_result.txt
```

## 注意事项

1. **VLM 延迟**: VLM API 调用通常需要 0.5-2 秒，期间机器人会原地等待
2. **超时处理**: 如果 VLM 验证超过 30 秒，该 episode 后续将禁用 VLM
3. **API 费用**: 每次验证会消耗 VLM API 调用配额
4. **网络依赖**: 需要能够访问阿里云 DashScope API

## 与其他模块的交互

### Patience-Aware Navigation

VLM 验证结果会影响物体的 `fused_confidence_`，从而影响 Patience-Aware Navigation 的决策阈值 τ(t)。

### Multi-View VLM Validation

本功能与现有的 Multi-View VLM Validation 独立，可以同时使用：
- Approach Verification: 逼近时验证，使用滑动窗口历史帧
- Multi-View Validation: 到达后验证，使用多视角实时采集

### Object Map

VLM 验证后的物体状态：
- `vlm_verified_`: 是否已验证
- `vlm_result_`: 验证结果
- `vlm_pending_`: 是否正在等待验证
- `vlm_adjusted_confidence_`: VLM 调整后的置信度

## 未来改进方向

1. **Early Trigger**: 当收集到足够多高置信度帧时提前触发验证
2. **Rescue Trigger**: 在即将放弃目标前最后一次验证
3. **多帧融合策略**: 根据帧间相似度动态调整发送给 VLM 的帧数
4. **本地 VLM 支持**: 支持本地部署的 VLM 模型以减少延迟
