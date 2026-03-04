# VLM 多视角观察点条件放宽总结

## 问题
原有的 VLM 多视角观察点（viewpoint）计算条件过于严格，导致找不到有效的观察位置，无法进行目标物体的多角度校验。

**原日志**:
```
[INFO] [1769162021.170858069]: [VLM Viewpoint] Object 29: center=(4.20,1.36), size=(2.47,3.89), optimal_dist=3.00
[INFO] [1769162021.170846637]: [VLM Viewpoint] Found 0 valid viewpoints for object 29
[WARN] [1769162021.170864369]: [VLM Multi-View] No valid observation viewpoints found, using current position
```

## 修改内容

### 1. 放宽 FOV 覆盖要求
**文件**: `src/planner/plan_env/src/object_map2d.cpp` (第 1072 行)

| 项目 | 旧值 | 新值 | 说明 |
|------|------|------|------|
| FOV 填充因子 | 0.6 | 0.1 | 对象只需占视野 10% 即可，不需要占据 60% |

**含义**: 
- 旧：物体必须填充相机视野的 60%，导致需要很近的距离
- 新：物体只需在相机视野中清晰可见即可（10% 覆盖率）

### 2. 放宽距离约束
| 参数 | 旧值 | 新值 | 说明 |
|------|------|------|------|
| 最小安全距离 | 0.5m | 0.3m | 可以更接近物体 |
| 最大观察距离 | 3.0m | 5.0m | 允许从更远的地方观察 |

**含义**: 观察点的距离范围更宽松，更容易找到有效的观察位置

### 3. 移除硬性的线性视觉检查
**文件**: `src/planner/plan_env/src/object_map2d.cpp` (第 1148-1159 行)

**旧逻辑**:
```cpp
if (vp.has_line_of_sight) {
    viewpoints.push_back(vp);  // 只有视线通畅才加入
}
```

**新逻辑**:
```cpp
viewpoints.push_back(vp);  // 总是加入，视线是可选项
```

**含义**: 
- 不再要求观察点与目标物体之间视线完全通畅
- 允许物体被部分遮挡的观察角度
- VLM 仍然可以进行校验（可能看不到完整物体，但可以看到部分特征）

### 4. 放宽地图安全性检查
**文件**: `src/planner/plan_env/src/object_map2d.cpp` (第 1130-1141 行)

**旧逻辑**:
```cpp
// 不允许在膨胀区 + 严格的障碍物距离检查 (>= robot_radius)
if (sdf_map_->getInflateOccupancy(vp.position) == 1) continue;
if (dist_to_obstacle < robot_radius) continue;
```

**新逻辑**:
```cpp
// 允许在膨胀区 + 放宽的障碍物距离检查 (>= robot_radius * 0.8)
// 实际路径规划会处理安全性
if (dist_to_obstacle < robot_radius * 0.8) continue;
```

**含义**: 
- 观察点可以在安全膨胀区内，只要不会实际碰撞
- 路径规划器会在实际导航时处理安全距离

## 预期效果

✅ **更容易找到观察点**
- 现在会为大多数目标物体找到至少一个观察位置
- 支持多角度的 VLM 校验
- 即使物体部分被遮挡也能进行观察

✅ **VLM 校验更可靠**
- 有多个观察角度，可以用多视角 AND 逻辑提高置信度
- 减少误判（多个视角都确认才算成功）

✅ **导航效率提升**
- 不再因为找不到观察点而失败
- 能够更灵活地选择观察位置

## 测试建议

1. 重新运行导航任务，观察 VLM Viewpoint 日志
2. 检查是否能找到有效的观察点
3. 观察 `debug/vlm_validation/` 目录中的校验结果

## 参数调整点

如果仍需进一步调整，可修改的参数：

```cpp
// 第 1072-1081 行
double fov_fill_factor = 0.1;     // 进一步减小为 0.05 允许更远距离观察
const double min_safe_distance = 0.3;    // 可减少至 0.2
const double max_observation_dist = 5.0; // 可增加至 6.0-7.0
const double robot_radius = 0.18;        // 可减小系数至 0.5-0.7
```

---

**编译日期**: 2026-01-23
**修改文件**: `src/planner/plan_env/src/object_map2d.cpp`
**相关代码行**: 1060-1180
