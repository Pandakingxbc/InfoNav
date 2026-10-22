# InfoNav Configuration System

This directory contains configuration files for the InfoNav object navigation system.

## Configuration Files

### 1. `default.yaml` - Unified Configuration
**NEW in Phase 6: Configuration Management**

Centralizes all hardcoded parameters for:
- VLM server ports (BLIP2, GroundingDINO, SAM, YOLOv7, D-FINE)
- Exploration parameters (alpha_base, tau_conf, fusion_alpha)
- Patience-aware navigation thresholds (tau_high, tau_low, T_max)
- Object detection defaults
- Path configurations

**Usage:**
```python
import yaml
from pathlib import Path

config_path = Path(__file__).parent / "config" / "default.yaml"
if config_path.exists():
    with open(config_path) as f:
        CONFIG = yaml.safe_load(f)
else:
    CONFIG = {}  # Use hardcoded defaults

# Access parameters with fallback
blip2_port = CONFIG.get('vlm', {}).get('blip2_itm', {}).get('port', 12182)
alpha_base = CONFIG.get('exploration', {}).get('adaptive_alpha', {}).get('alpha_base', 0.8)
```

**Backward Compatibility:**
If `default.yaml` is missing, the system uses hardcoded defaults. This ensures existing deployments continue to work without modification.

### 2. `habitat_eval_*.yaml` - Dataset-Specific Configs
Hydra configuration files for different datasets:
- `habitat_eval_hm3dv1.yaml` - HM3D-v1 dataset
- `habitat_eval_hm3dv2.yaml` - HM3D-v2 dataset (default)
- `habitat_eval_mp3d.yaml` - MP3D dataset

These files configure:
- Habitat simulator settings
- Dataset paths
- Detector mode (YOLOv7 vs D-FINE)
- LLM client settings
- Episode constraints

### 3. `logging_config.yaml` - Logging Configuration
Controls verbosity levels across Python and ROS components.

**Log Levels:**
- `VERBOSE`: All debug info, timing breakdowns, VLM details
- `NORMAL`: Basic progress info, episode summaries (default)
- `MINIMAL`: Only episode results and errors
- `SILENT`: Errors only

**Environment Variable Overrides:**
```bash
export APEXNAV_LOG_LEVEL=MINIMAL
export APEXNAV_VLM_DETAIL=false
export APEXNAV_TIMING_DETAIL=false
```

## Configuration Hierarchy

Parameters are loaded in the following priority order:

1. **Environment variables** (highest priority)
   - `USE_PARALLEL_BLIP2`: Enable parallel BLIP2 servers
   - `APEXNAV_LOG_LEVEL`: Override logging level

2. **Hydra config files** (`habitat_eval_*.yaml`)
   - Dataset-specific settings
   - CLI overrides: `python habitat_evaluation.py key=value`

3. **Unified config** (`default.yaml`)
   - VLM server ports
   - Exploration parameters
   - Patience thresholds

4. **Hardcoded defaults** (lowest priority)
   - Used when config file is missing
   - Ensures backward compatibility

## Key Parameters Reference

### VLM Server Ports
```yaml
vlm:
  blip2_itm:
    port: 12182  # BLIP2-ITM server
  grounding_dino:
    port: 12181  # GroundingDINO server
  sam:
    port: 12183  # SAM segmentation server
  yolov7:
    port: 12184  # YOLOv7 detector (Mode 1)
  dfine:
    port: 12185  # D-FINE detector (Mode 2)
```

### Exploration Parameters
```yaml
exploration:
  policy: 4  # 0: distance, 1: value(greedy), 2: TSP, 4: WTRP
  fusion_alpha: 0.8  # V_total = alpha * V_sem + (1-alpha) * V_ig
  adaptive_alpha:
    enabled: true
    alpha_base: 0.8  # Base alpha when no semantic cues
    tau_conf: 0.8    # Confidence threshold for adaptation
```

### Patience-Aware Navigation
```yaml
patience_nav:
  T_max: 450        # Maximum episode steps
  tau_high: 0.4     # Starting threshold (strict)
  tau_low: 0.2      # Ending threshold (relaxed)

  # Target-specific thresholds by object size
  threshold_by_size:
    large:          # bed, couch, bathtub
      tau_high: 0.85
      tau_low: 0.55
    medium_large:   # tv, counter, cabinet
      tau_high: 0.85
      tau_low: 0.5
    medium:         # chair, table, sink
      tau_high: 0.75
      tau_low: 0.5
    small:          # plant, pillow, towel
      tau_high: 0.65
      tau_low: 0.4
```

## Testing Configuration

The lightweight GitHub Actions workflow parses all YAML configuration files on every push and pull request. Locally, the same check can be run with:

```bash
python - <<'PY'
from pathlib import Path
import yaml

for path in sorted(Path("config").glob("*.yaml")):
    yaml.safe_load(path.read_text(encoding="utf-8"))
    print(f"parsed {path}")
PY
```

## Migration Guide

### For Developers
If you previously used hardcoded values:

**Before (hardcoded):**
```python
BLIP2_PORT = 12182
alpha_base = 0.8
tau_high = 0.4
```

**After (config-based):**
```python
# In module initialization
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"
if CONFIG_PATH.exists():
    with open(CONFIG_PATH) as f:
        CONFIG = yaml.safe_load(f)
else:
    CONFIG = {}

# In code
BLIP2_PORT = CONFIG.get('vlm', {}).get('blip2_itm', {}).get('port', 12182)
alpha_base = CONFIG.get('exploration', {}).get('adaptive_alpha', {}).get('alpha_base', 0.8)
tau_high = CONFIG.get('patience_nav', {}).get('tau_high', 0.4)
```

### For System Administrators
To customize parameters without modifying code:

1. Edit `/path/to/InfoNav/config/default.yaml`
2. Modify desired parameters
3. Restart the system

Example customization:
```yaml
# Increase BLIP2 timeout for slower systems
vlm:
  blip2_itm:
    port: 12182
    timeout: 120  # seconds

# Adjust exploration aggressiveness
exploration:
  fusion_alpha: 0.9  # More semantic-driven (less exploration)

# Relax patience thresholds for difficult environments
patience_nav:
  tau_high: 0.35  # Start more relaxed
  tau_low: 0.15   # End more relaxed
```

## Phase 6 Implementation Summary

**Goal:** Create unified configuration system for centralized parameter management.

**Changes:**
1. Created `config/default.yaml` with all hardcoded parameters
2. Modified `habitat_evaluation.py` to load config on startup
3. Updated VLM client modules (`get_object_utils.py`, `get_itm_message.py`)
4. Maintained backward compatibility (config file optional)
5. Added automated YAML parsing in `.github/workflows/quality.yml`

**Benefits:**
- ✓ Centralized parameter management
- ✓ Easy tuning without code modification
- ✓ Backward compatible (optional config file)
- ✓ Self-documenting (YAML comments)
- ✓ Environment variable overrides supported

## Troubleshooting

### Config file not loaded
**Problem:** System uses hardcoded defaults instead of config file.

**Solution:**
1. Verify file exists: `ls -la /path/to/InfoNav/config/default.yaml`
2. Check file permissions: `chmod 644 config/default.yaml`
3. Validate YAML syntax: `python -c "import yaml; yaml.safe_load(open('config/default.yaml'))"`

### YAML parsing error
**Problem:** `yaml.scanner.ScannerError` when loading config.

**Solution:**
1. Check YAML syntax (indentation, colons, quotes)
2. Use online YAML validator
3. Check for special characters in values

### Parameter not applied
**Problem:** Changed config value but system still uses old value.

**Solution:**
1. Restart Python process (config loaded on import)
2. Check parameter path is correct (nested dict access)
3. Verify fallback default matches expected value
4. Check if environment variable is overriding config

## Related Documentation

- [Main README.md](../README.md) - Project overview
- [Logging Documentation](logging_config.yaml) - Logging system details
- [Habitat Config](habitat_eval_hm3dv2.yaml) - Dataset configuration
- [ROS Launch Files](../src/planner/exploration_manager/launch/) - ROS parameter server config

## Support

For questions or issues with configuration:
1. Run the YAML parsing command in the testing section above.
2. Check log output for "[Config]" messages
3. Review this README for parameter reference
4. Submit issue with config file and error message
