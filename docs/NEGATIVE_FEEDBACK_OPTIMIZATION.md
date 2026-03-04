# InfoNav 负反馈机制优化方案

> 基于 Ascent (VLFM) 和 ApexNav 的对比分析

---

## 一、三个项目的负反馈机制对比

### 1.1 对比总览

| 特性 | Ascent (VLFM) | ApexNav | InfoNav |
|------|---------------|---------|---------|
| **实现语言** | Python | C++ | C++ |
| **负反馈触发** | 几何验证（进入FOV后未再次检测） | 观测区域与物体点云重叠 | 观测区域与物体点云重叠 |
| **使用 ITM** | ❌ 不使用 | ✅ 使用 | ✅ 使用 |
| **决策方式** | 二元（删除/保留） | 连续（融合置信度） | 连续（融合置信度） |
| **代码位置** | `object_point_cloud_map.py` | `object_map2d.cpp` | `object_map2d.cpp` |

### 1.2 Ascent (VLFM) 的方法：几何验证法

```python
# ascent/ascent/mapping/object_point_cloud_map.py: L151-186
def update_explored(self, tf_camera_to_episodic, max_depth, cone_fov):
    """
    移除那些：
    1. 最初检测时超出范围 (out-of-range)
    2. 但现在已经进入视野范围内的点云

    如果进入视野后仍然没检测到，说明是误检，直接删除
    """
    for obj in self.clouds:
        within_range = within_fov_cone(camera_pos, camera_yaw, cone_fov, max_depth*0.5, points)

        for range_id in range_ids:
            if range_id == 1:  # 原本就在范围内的检测，保留
                continue
            # 原本超出范围的检测，现在进入范围了 → 删除
            self.clouds[obj] = self.clouds[obj][self.clouds[obj][..., -1] != range_id]
```

**核心思想**：
- 每个检测点携带一个 `range_id` 标记
- `range_id = 1.0` 表示检测时在有效范围内
- `range_id = random()` 表示检测时超出范围（不太可靠）
- 当机器人靠近后，如果超出范围的点进入了 FOV 但没有被重新检测到，就删除

**优点**：简单、直接、不需要额外的 VLM 计算
**缺点**：过于激进，可能删除正确的检测

### 1.3 ApexNav / InfoNav 的方法：ITM 加权融合法

```cpp
// ApexNav & InfoNav: object_map2d.cpp L66-128
void ObjectMap2D::inputObservationObjectsCloud(
    const vector<pcl::PointCloud> observation_clouds,
    const double& itm_score)
{
    for (int i = 0; i < observation_clouds.size(); i++) {
        for (int label = 0; label < 5; ++label) {
            // 计算空间重叠
            overlap_count = computeOverlap(observation_cloud, object.clouds_[label]);

            if (overlap_count > 0) {
                // 使用融合算法降低置信度
                confidence_now = 0.0;  // 负证据
                if (label == 0)
                    confidence_now = itm_score;  // ← 问题在这里！

                // 加权融合
                new_confidence = fusion(old_confidence, confidence_now, overlap_count);
            }
        }
    }
}
```

**核心思想**：
- 当观测到某区域但检测器未检测到物体时，将观测作为"负证据"
- 对于 `label != 0`：`confidence_now = 0.0`（纯负证据）
- 对于 `label == 0`：`confidence_now = itm_score`（软证据）

---

## 二、InfoNav 的问题分析

### 2.1 ITM 分数来源错误

**当前数据流**：
```
habitat_evaluation.py
    │
    │ get_multi_source_cosine_with_ig()
    │ 返回 hypotheses_data (多个假说的scores)
    │
    ▼
fused_cosine = Σ(weight × score) / Σ(weight)  ← 所有假说的加权平均！
    │
    │ 发布到 /blip2/cosine_score
    ▼
map_ros.cpp 订阅 → itm_score_
    │
    ▼
inputObservationObjectsCloud(..., itm_score_)
    │
    ▼
if (label == 0)
    confidence_now = itm_score_;  // ← 用错了！
```

**问题**：`fused_cosine` 包含了：
- `room_type` 假说分数（房间类型匹配度）
- `target_object` 假说分数（目标物体匹配度）
- `co_occurrence` 假说分数（共现物体匹配度）
- `part_attribute` 假说分数（部件属性匹配度）

这个融合分数**不能代表**"当前视野中是否存在目标物体"！

### 2.2 语义不一致

负反馈的语义应该是：
```
"我观测了物体所在的区域，VLM 对【目标物体】的匹配分数是 X"
```

而不是：
```
"我观测了物体所在的区域，【所有HSVM假说】的加权融合分数是 X"
```

---

## 三、优化方案

### 方案 A：学习 Ascent，禁用 ITM 负反馈（推荐）

**修改位置**：`src/planner/plan_env/src/object_map2d.cpp`

```cpp
// 修改前 (L115-117)
double confidence_now = 0.0;  // Negative evidence has zero confidence
if (label == 0)
    confidence_now = itm_score;  // Use ITM score for primary label

// 修改后
double confidence_now = 0.0;  // 所有标签都使用纯负证据
// 移除 ITM score 的特殊处理
```

**优点**：
- 简单直接
- 与 Ascent 的思路一致
- 不需要修改 Python 端

**缺点**：
- 可能过于激进地降低正确检测的置信度

---

### 方案 B：使用正确的 target_object ITM 分数

**步骤 1**：修改 Python 端，提取 target_object 分数

```python
# habitat_evaluation.py L545-552 修改

# 找到 target_object 类型的假说分数
target_object_score = 0.0
for h in hypotheses_data:
    if h["type"] == "target_object":
        target_object_score = h["score"]
        break

# 发布目标物体的分数，而不是融合分数
publish_float64(itm_score_pub, target_object_score)
```

**步骤 2**：（可选）创建专用话题

```python
# 创建新的发布者
target_itm_pub = rospy.Publisher("/habitat/target_object_itm", Float64, queue_size=10)

# 发布
publish_float64(target_itm_pub, target_object_score)
```

**步骤 3**：（可选）C++ 端订阅新话题

```cpp
// map_ros.cpp
target_itm_sub_ = node_.subscribe("/habitat/target_object_itm", 10,
    &MapROS::targetItmCallback, this);
```

**优点**：
- 语义正确
- 保留软证据机制

**缺点**：
- 需要修改多处代码
- 需要确保 hypotheses_data 中有 target_object 类型

---

### 方案 C：完全禁用负反馈机制

**修改位置**：`src/planner/exploration_manager/launch/algorithm.xml`

```xml
<!-- 修改前 -->
<param name="object/use_observation" value="true" type="bool"/>

<!-- 修改后 -->
<param name="object/use_observation" value="false" type="bool"/>
```

**优点**：
- 最简单，一行配置
- 不影响正向检测的置信度融合

**缺点**：
- 完全放弃负反馈，误检可能需要更长时间才能被清除

---

### 方案 D：增强型几何验证（借鉴 Ascent）

在 InfoNav 中实现类似 Ascent 的几何验证机制：

```cpp
// 新增函数：基于视野锥验证
void ObjectMap2D::validateObjectsInFOV(
    const Vector3d& camera_pos,
    double camera_yaw,
    double fov,
    double max_depth,
    const vector<DetectedObject>& current_detections)
{
    for (auto& object : objects_) {
        // 检查物体是否在当前 FOV 内
        if (isInFOVCone(camera_pos, camera_yaw, fov, max_depth, object.average_)) {
            // 检查当前帧是否重新检测到
            bool re_detected = false;
            for (const auto& det : current_detections) {
                if (computeOverlap(det.cloud, object.clouds_[0]) > threshold) {
                    re_detected = true;
                    break;
                }
            }

            // 如果在 FOV 内但未被重新检测，标记为可疑
            if (!re_detected) {
                object.suspicious_count_++;
                if (object.suspicious_count_ >= MAX_SUSPICIOUS_COUNT) {
                    // 大幅降低置信度或删除
                    object.confidence_scores_[0] *= 0.5;
                }
            } else {
                object.suspicious_count_ = 0;  // 重置
            }
        }
    }
}
```

**优点**：
- 结合几何验证和置信度融合
- 更加鲁棒

**缺点**：
- 实现复杂度高
- 需要仔细调参

---

## 四、推荐实施计划

### Phase 1：快速修复（立即）

**选择方案 A 或 C**：

```cpp
// 方案 A：修改 object_map2d.cpp L115-117
double confidence_now = 0.0;
// 移除: if (label == 0) confidence_now = itm_score;
```

或

```xml
<!-- 方案 C：修改 algorithm.xml -->
<param name="object/use_observation" value="false" type="bool"/>
```

### Phase 2：正确实现（1-2天）

**选择方案 B**：

1. 修改 `habitat_evaluation.py`，提取 `target_object` 分数
2. 验证 `hypotheses_data` 中确实包含 `target_object` 类型
3. 测试新的 ITM 分数是否正确反映目标物体存在性

### Phase 3：增强优化（可选）

**选择方案 D**：

1. 在 `ObjectMap2D` 中实现几何验证函数
2. 添加 `suspicious_count_` 计数器
3. 结合几何验证和置信度融合

---

## 五、实验验证建议

### 5.1 消融实验

| 配置 | 说明 | 预期效果 |
|------|------|----------|
| Baseline | 当前实现（错误的 ITM） | 基准 |
| NoObs | `use_observation=false` | 误检清除更慢 |
| ZeroConf | `confidence_now=0.0` for all | 更激进的负反馈 |
| CorrectITM | 使用 target_object 分数 | 更准确的负反馈 |
| GeomValid | 几何验证 + 置信度融合 | 最鲁棒 |

### 5.2 评估指标

1. **假阳性率 (FPR)**：误检目标物体的比例
2. **召回率 (Recall)**：正确检测目标物体的比例
3. **导航成功率 (SR)**：成功到达目标的比例
4. **SPL**：成功率加权的路径长度

### 5.3 测试场景

- **高假阳性场景**：plant, bed, tv（FPR > 25%）
- **正常场景**：chair, toilet（FPR < 18%）
- **边缘情况**：远距离检测、遮挡恢复

---

## 六、代码参考

### 6.1 InfoNav 相关文件

| 文件 | 说明 |
|------|------|
| `src/planner/plan_env/src/object_map2d.cpp` | 物体地图实现 |
| `src/planner/plan_env/src/map_ros.cpp` | ROS 接口，订阅 ITM |
| `src/planner/exploration_manager/launch/algorithm.xml` | 参数配置 |
| `habitat_evaluation.py` | Python 端，发布 ITM |

### 6.2 Ascent 相关文件

| 文件 | 说明 |
|------|------|
| `ascent/mapping/object_point_cloud_map.py` | 物体地图（几何验证） |

### 6.3 ApexNav 相关文件

| 文件 | 说明 |
|------|------|
| `src/planner/plan_env/src/object_map2d.cpp` | 物体地图（与 InfoNav 相同） |

---

## 七、总结

### 问题根源

InfoNav 和 ApexNav 的负反馈机制使用了**语义错误的 ITM 分数**：
- 应该使用：目标物体的 ITM 匹配分数
- 实际使用：所有 HSVM 假说的加权融合分数

### 推荐方案

1. **短期**：禁用 ITM 特殊处理（方案 A）或完全禁用负反馈（方案 C）
2. **中期**：使用正确的 target_object 分数（方案 B）
3. **长期**：实现几何验证 + 置信度融合（方案 D）

### Ascent 的启示

Ascent 的几何验证方法虽然简单，但逻辑清晰：
- 不依赖 VLM 分数
- 基于"再次观测是否能检测到"的直觉
- 二元决策避免了复杂的置信度计算

这种方法在实践中可能比复杂的 ITM 融合更加鲁棒。

---

*文档生成时间: 2026-01-21*
*分析项目: InfoNav, ApexNav, Ascent (VLFM)*
