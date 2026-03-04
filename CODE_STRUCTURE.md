# InfoNav 代码架构

## 项目概述

**InfoNav** 是一个基于 Habitat 仿真环境的智能体导航系统，融合了视觉语言模型(VLM)和大语言模型(LLM)技术，用于目标物体搜索和导航任务。该项目采用 **Python + C++ 混合架构**，使用 ROS 进行模块间通信。

## 目录结构

```
InfoNav/
├── habitat_evaluation.py          # [主入口] 评估循环、Habitat环境管理
├── params.py                      # 状态码、动作定义、结果类型常量
├── infonav_environment.yaml       # Conda环境配置
├── pyproject.toml                 # 项目元数据、Python依赖
├── llm_hypothesis_analyzer.py     # LLM假说分析工具
├── vlm_environment_analyzer.py    # VLM环境分析工具
├── habitat_manual_control.py      # 手动控制脚本
│
├── src/                           # [核心代码] C++/Python混合
│   └── planner/
│       ├── exploration_manager/   # [C++ ROS节点] 探索决策和状态机
│       │   ├── include/
│       │   │   ├── exploration_fsm.h
│       │   │   ├── exploration_manager.h
│       │   │   └── exploration_data.h
│       │   ├── src/
│       │   │   ├── exploration_node.cpp
│       │   │   ├── exploration_fsm.cpp
│       │   │   └── exploration_manager.cpp
│       │   ├── launch/
│       │   └── config/
│       │
│       ├── plan_env/              # 规划环境库
│       │   ├── include/plan_env/
│       │   │   ├── frontier_map2d.h    # 前沿点检测
│       │   │   ├── object_map2d.h      # 物体位置管理
│       │   │   ├── sdf_map2d.h         # 障碍物栅格地图
│       │   │   └── value_map2d.h       # 语义价值地图
│       │   ├── src/
│       │   └── msg/                    # ROS消息定义
│       │
│       ├── path_searching/        # A*路径搜索
│       │   ├── include/
│       │   │   └── astar2d.h
│       │   └── src/
│       │       └── astar2d.cpp
│       │
│       └── utils/
│           ├── lkh_mtsp_solver/   # LKH TSP求解器
│           └── vis_utils/         # 可视化工具
│
├── vlm/                           # 视觉语言模型模块
│   ├── detector/
│   │   ├── grounding_dino.py      # GroundingDINO物体检测
│   │   ├── detections.py          # 检测结果处理
│   │   └── D-FINE/                # D-FINE检测器
│   ├── itm/
│   │   └── blip2itm_client.py     # BLIP2图像-文本匹配
│   ├── RedNet/                    # 语义分割模型
│   ├── utils/
│   │   ├── get_object_utils.py    # 物体检测融合
│   │   ├── get_itm_message.py     # ITM消息生成
│   │   └── http_vlm_client.py     # HTTP客户端
│   └── server_wrapper.py          # VLM服务器包装
│
├── llm/                           # 大语言模型模块
│   ├── prompt/
│   │   ├── get_llm_answer.py      # LLM查询接口
│   │   └── value_map_hypothesis.py # 语义假说加载
│   ├── client/
│   │   └── ollama_answer.py       # Ollama客户端
│   ├── answer_reader/
│   │   └── answer_reader.py       # LLM答案解析
│   ├── utils/
│   └── answers/                   # LLM回答缓存
│
├── basic_utils/                   # 基础工具库
│   ├── object_point_cloud_utils/
│   │   └── object_point_cloud.py  # 点云生成
│   ├── failure_check/
│   │   └── failure_check.py       # 失败检测
│   ├── record_episode/
│   └── logging/
│
├── habitat2ros/                   # Habitat→ROS发布器
│   └── habitat_publisher.py
│
├── GroundingDINO/                 # [外部库] GroundingDINO
├── yolov7/                        # [外部库] YOLOv7
│
├── config/                        # 配置文件
│   ├── habitat_eval_hm3dv1.yaml
│   ├── habitat_eval_hm3dv2.yaml
│   └── habitat_eval_mp3d.yaml
│
├── env/                           # Habitat环境数据
├── data/                          # 数据集
├── docs/                          # 项目文档
├── debug/                         # 调试工具
├── scripts/                       # 辅助脚本
├── logs/                          # 运行日志
├── build/                         # CMake构建输出
├── devel/                         # 开发版本
└── install/                       # 安装版本
```

## 核心模块说明

### 1. 主评估循环 (habitat_evaluation.py)

- Habitat 仿真环境管理
- ROS 通信协调
- VLM 推理调度
- 评估指标计算

### 2. C++ 探索管理器 (exploration_manager)

- **ExplorationFSM**: 状态机实现 (10ms定时器)
- **ExplorationManager**: 规划器（前沿探索、物体导航）
- 支持多种探索策略: DISTANCE, SEMANTIC, HYBRID, TSP_DIST

### 3. 规划环境库 (plan_env)

| 模块 | 功能 |
|------|------|
| frontier_map2d | 前沿点检测 |
| object_map2d | 物体位置管理 |
| sdf_map2d | 障碍物栅格地图 |
| value_map2d | 语义价值地图 |
| map_ros | ROS接口 |

### 4. VLM 模块

- **GroundingDINO**: 开放词汇物体检测
- **BLIP2 ITM**: 图像-文本匹配
- **RedNet**: 语义分割

### 5. LLM 模块

- 语义假说生成
- 房间-物体关联推理
- Ollama 本地 LLM 支持

## 数据流

```
Habitat仿真环境
    ↓ RGB/Depth/Odometry
habitat_evaluation.py (主循环)
    ├─→ VLM推理 → /detector/clouds_with_scores
    ├─→ 语义得分 → /habitat/semantic_scores
    └─→ 等待 /habitat/plan_action
        ↓
exploration_manager (ROS节点)
    ├─ 栅格地图更新
    ├─ 前沿检测
    ├─ 路径规划 (A* + TSP)
    └─ 状态机转移
        ↓
    /habitat/plan_action → Habitat执行动作
```

## 特色机制

1. **接近检测验证**: 6帧滑动窗口验证目标
2. **无进展检测**: 50步无改善标记DORMANT
3. **软到达检测**: 振荡时触发软到达
4. **检测方向优先**: 从物体首次检测方向接近

## 代码统计

- Python 文件: 50+ 个
- C++ 文件: 30+ 个
- 配置文件: 30+ 个
- 文档: 10+ 个

生成时间: 2026-03-04
