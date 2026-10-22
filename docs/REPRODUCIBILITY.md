# Reproducing InfoNav experiments

This document records the public evaluation protocol for the NeurIPS 2026 release. The paper remains the authoritative source for the full experimental design and reported tables.

## Supported benchmarks

| Dataset flag | Habitat episodes | Scene dataset |
| --- | --- | --- |
| `hm3dv1` | HM3D ObjectNav v1 validation | HM3D v0.1-compatible scenes |
| `hm3dv2` | HM3D ObjectNav v2 validation | HM3D v0.2 scenes |
| `mp3d` | MP3D ObjectNav validation | Matterport3D scenes |

Dataset files are not redistributed. Follow the access and license instructions linked from the main README.

## Release protocol

The full InfoNav planner uses:

```text
detector_mode=2                 # D-FINE + GroundingDINO
dfine_num_top_queries=15        # applied by vlm/detector/dfine.py
exploration_policy=4            # WTRP over top-k fused-value frontiers
use_multi_source=true           # hierarchical semantic sources
fusion_alpha=0.8
use_ig_fusion=true
use_adaptive_alpha=true
alpha_base=0.8
tau_conf=0.8
top_k_value=8
wtrp_temperature=0.3
wtrp_hysteresis_ratio=0.8
wtrp_max_brute_force=8
```

The checked-in dataset configurations additionally fix the episode iterator, maximum step count, success distance, camera geometry, and detector thresholds. Record any CLI or ROS override with the result.

## Run one evaluation

Start the following processes from the repository root in separate terminals.

1. ROS master:

   ```bash
   roscore
   ```

2. Perception services:

   ```bash
   conda activate infonav
   ./scripts/start_vlm_servers.sh restart infonav
   ```

3. Planner:

   ```bash
   conda activate infonav
   source /opt/ros/noetic/setup.bash
   source devel/setup.bash
   roslaunch exploration_manager exploration.launch \
     dataset:=hm3dv2 \
     exploration_policy:=4 \
     use_multi_source:=true \
     fusion_alpha:=0.8 \
     use_ig_fusion:=true \
     use_adaptive_alpha:=true \
     alpha_base:=0.8 \
     tau_conf:=0.8 \
     top_k_value:=8 \
     wtrp_temperature:=0.3 \
     wtrp_hysteresis_ratio:=0.8 \
     wtrp_max_brute_force:=8
   ```

4. Evaluator:

   ```bash
   conda activate infonav
   source /opt/ros/noetic/setup.bash
   source devel/setup.bash
   python habitat_evaluation.py \
     --dataset hm3dv2 \
     --detector-mode 2 \
     --control-rate-hz 10
   ```

Replace `hm3dv2` consistently in the planner and evaluator commands for the other benchmarks.

## Cached versus online semantic hypotheses

The default command uses checked-in semantic responses under `llm/answers/` where supported. For online generation, export credentials locally and add `--realtime-hypothesis`:

```bash
export DASHSCOPE_API_KEY="..."
export DEEPSEEK_API_KEY="..."
python habitat_evaluation.py \
  --dataset hm3dv2 \
  --detector-mode 2 \
  --realtime-hypothesis \
  --vlm-api-type dashscope \
  --vlm-model qwen-vl-max \
  --llm-model deepseek-chat
```

Never commit API keys, generated credentials, or private dataset URLs.

## Outputs and provenance

The evaluator writes per-episode logs and JSON metrics below the configured `video_output_path`. Long-run scripts also write a protocol file containing the Git commit, dirty-worktree state, seed, model endpoints, and ROS parameters. Preserve these files with any reported aggregate.

For a result to be comparable, archive at least:

- the Git commit and whether tracked files were modified;
- dataset version, split, ordered episode range, and seed;
- detector and checkpoint names;
- LLM/VLM model identifiers and whether cached responses were used;
- success distance, step budget, and control rate;
- all exploration and fusion parameters; and
- `episode_metrics.jsonl` plus the aggregate report.

## Ablations

Use explicit planner arguments so the changed factor is visible:

| Experiment | Override |
| --- | --- |
| Distance-greedy | `exploration_policy:=0` |
| Value-greedy | `exploration_policy:=1` |
| No information-gain fusion | `use_ig_fusion:=false` |
| Fixed fusion weight | `use_adaptive_alpha:=false fusion_alpha:=0.8` |
| Single-source semantic map | `use_multi_source:=false` |

Do not describe value-greedy as simply “without WTRP”: it is a replacement frontier-selection policy. Keep the episode order and every other parameter fixed for paired comparisons.

## Known release constraints

- Full evaluation requires Ubuntu 20.04/ROS Noetic and a CUDA-capable GPU; it is not exercised by the lightweight GitHub Actions checks.
- HM3D and MP3D require external access approval and substantial local storage.
- Online hypotheses depend on external APIs and can change as hosted models change. Cached responses are preferable for deterministic comparisons.
- The historical long-run scripts predate this release cleanup. Inspect their generated protocol file before treating a run as paper-equivalent.
