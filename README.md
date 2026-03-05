# InfoNav

**InfoNav: A Unified Value Framework Integrating Semantics Value and Information Gain for Zero-Shot Navigation**

> **Paper:** Coming soon

📄 [Architecture Overview (PDF)](airchitecture.pdf)

## TODO

- [ ] Release real-world deployment code
- [ ] Release ROS2 support version

## Overview

InfoNav is an intelligent navigation system for embodied agents in indoor environments, combining Vision-Language Models (VLM) and Large Language Models (LLM) for semantic-aware object navigation tasks. The system proposes a unified value framework that integrates semantic value and information gain for zero-shot navigation.

InfoNav is built on top of [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) and uses ROS for inter-module communication. The system integrates:

- **VLM-based Object Detection**: GroundingDINO, BLIP2 for open-vocabulary detection and image-text matching
- **LLM-based Semantic Reasoning**: Room-object association and hypothesis generation
- **Unified Value Framework**: Combining semantic value with information gain
- **Hybrid Exploration Strategy**: Frontier-based exploration with semantic value maps
- **Robust Navigation Planning**: A* path planning with TSP optimization for multi-target scenarios

## System Requirements

- **OS**: Ubuntu 20.04 / 22.04
- **GPU**: NVIDIA GPU with CUDA support (RTX 3090 or higher recommended)
- **Python**: 3.9+
- **ROS**: ROS Noetic

## Installation

### 1. Create Conda Environment

```bash
# Create environment with Habitat-Sim
conda create -n infonav python=3.9 habitat-sim=0.3.1 withbullet -c conda-forge -c aihabitat

conda activate infonav
```

### 2. Install ROS Noetic

Follow the official [ROS Noetic installation guide](http://wiki.ros.org/noetic/Installation/Ubuntu).

### 3. Clone Repository

```bash
git clone https://github.com/Pandakingxbc/InfoNav.git
cd InfoNav
```

### 4. Install Python Dependencies

```bash
pip install -e .
```

This will install all required dependencies including:
- `transformers`, `timm` - Deep learning models
- `open3d` - 3D point cloud processing
- `opencv-python` - Image processing
- `openai`, `ollama` - LLM clients
- `groundingdino`, `mobile_sam` - VLM models

### 5. Build ROS Packages

```bash
# Initialize catkin workspace
catkin init
catkin config --extend /opt/ros/noetic

# Build
catkin build
source devel/setup.bash
```

### 6. Download Model Weights

Download the required model weights:

```bash
# GroundingDINO weights
mkdir -p weights
wget -P weights/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Additional model weights (RedNet, etc.)
# Please refer to the respective model repositories
```

### 7. Download Dataset

Download HM3D or MP3D dataset following [Habitat documentation](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md).

```bash
# Example directory structure
data/
└── scene_datasets/
    └── hm3d/
        ├── train/
        ├── val/
        └── ...
```

## Usage

### Quick Start

1. **Start ROS Core**
```bash
roscore
```

2. **Launch Exploration Node**
```bash
roslaunch exploration_manager exploration.launch
```

3. **Run Evaluation**
```bash
python habitat_evaluation.py --config config/habitat_eval_hm3dv2.yaml
```

### Configuration

Main configuration files:
- `config/habitat_eval_hm3dv1.yaml` - HM3D-v1 dataset config
- `config/habitat_eval_hm3dv2.yaml` - HM3D-v2 dataset config
- `config/habitat_eval_mp3d.yaml` - MP3D dataset config
- `src/planner/exploration_manager/config/algorithm.xml` - Algorithm parameters

### Manual Control

For testing and debugging:
```bash
python habitat_manual_control.py
```

## Project Structure

```
InfoNav/
├── habitat_evaluation.py      # Main evaluation loop
├── params.py                  # Global constants
├── src/planner/               # C++ ROS planning modules
│   ├── exploration_manager/   # FSM and exploration logic
│   ├── plan_env/              # Map representations
│   └── path_searching/        # A* path planning
├── vlm/                       # Vision-Language Models
├── llm/                       # Large Language Models
├── basic_utils/               # Utility functions
├── habitat2ros/               # Habitat-ROS bridge
└── config/                    # Configuration files
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{infonav2025,
  title={InfoNav: A Unified Value Framework Integrating Semantics Value and Information Gain for Zero-Shot Navigation},
  author={},
  journal={},
  year={2025}
}
```

## License

This project is released under the MIT License.

## Acknowledgements

This project builds upon several excellent open-source projects:
- [Habitat-Sim](https://github.com/facebookresearch/habitat-sim)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [BLIP2](https://github.com/salesforce/LAVIS)

## Contact

For questions or issues, please open an issue on GitHub.
