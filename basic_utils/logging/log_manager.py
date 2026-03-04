"""
Logging Manager for ApexNav

Provides centralized logging control with environment variable support
and configuration file integration.

Usage:
    from basic_utils.logging.log_manager import LogManager

    logger = LogManager()

    # Conditional logging
    if logger.should_log_vlm_detail():
        print(f"[VLM Detail] ...")

    # Or use wrapper
    logger.log_vlm_detail(f"Total: {t_total:.3f}s")
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


class LogManager:
    """Centralized logging manager for ApexNav"""

    # Log levels
    VERBOSE = "VERBOSE"
    NORMAL = "NORMAL"
    MINIMAL = "MINIMAL"
    SILENT = "SILENT"

    def __init__(self, config_path: str = None):
        """
        Initialize logging manager

        Args:
            config_path: Path to logging_config.yaml (optional)
        """
        self.config = self._load_config(config_path)
        self.log_level = self._get_log_level()
        self.step_counter = 0

    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load logging configuration from YAML file"""
        if config_path is None:
            # Default path
            repo_root = Path(__file__).parent.parent.parent
            config_path = repo_root / "config" / "logging_config.yaml"

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Return default config
            return {
                'logging': {
                    'global_level': 'NORMAL',
                    'components': {
                        'vlm': {'show_detail': False, 'show_semantic_scores': False},
                        'timing': {'show_step_breakdown': False, 'show_summary_only': True, 'print_interval': 10},
                        'detection': {'show_summary': True, 'show_detail': False},
                        'record': {'enable_rotation': True, 'max_size_mb': 5}
                    }
                }
            }

    def _get_log_level(self) -> str:
        """Get effective log level from environment or config"""
        # Environment variable takes precedence
        env_level = os.environ.get('APEXNAV_LOG_LEVEL', '').upper()
        if env_level in [self.VERBOSE, self.NORMAL, self.MINIMAL, self.SILENT]:
            return env_level

        # Fall back to config file
        return self.config.get('logging', {}).get('global_level', self.NORMAL).upper()

    def _get_component_setting(self, component: str, setting: str, env_var: str = None) -> bool:
        """
        Get component-specific setting

        Args:
            component: Component name (e.g., 'vlm', 'timing')
            setting: Setting name (e.g., 'show_detail')
            env_var: Environment variable to check (optional)

        Returns:
            Boolean setting value
        """
        # Check environment variable first
        if env_var and env_var in os.environ:
            return os.environ.get(env_var, 'false').lower() == 'true'

        # Check level-specific config
        level_config = self.config.get('logging', {}).get('levels', {}).get(self.log_level, {})
        if setting in level_config:
            return level_config[setting]

        # Fall back to component-specific config
        return self.config.get('logging', {}).get('components', {}).get(component, {}).get(setting, False)

    # VLM logging controls
    def should_log_vlm_detail(self) -> bool:
        """Should log detailed VLM timing information?"""
        return self._get_component_setting('vlm', 'show_detail', 'APEXNAV_VLM_DETAIL')

    def should_log_semantic_scores(self) -> bool:
        """Should log multi-source semantic scores?"""
        return self._get_component_setting('vlm', 'show_semantic_scores', 'APEXNAV_SEMANTIC_SCORES')

    def log_vlm_detail(self, message: str):
        """Log VLM detail message if enabled"""
        if self.should_log_vlm_detail():
            print(message)

    def log_semantic_scores(self, message: str):
        """Log semantic scores if enabled"""
        if self.should_log_semantic_scores():
            print(message)

    # Timing logging controls
    def should_log_timing_breakdown(self) -> bool:
        """Should log detailed timing breakdown?"""
        return self._get_component_setting('timing', 'show_step_breakdown', 'APEXNAV_TIMING_DETAIL')

    def should_log_timing_summary(self) -> bool:
        """Should log timing summary only?"""
        if self.log_level == self.SILENT:
            return False
        return self._get_component_setting('timing', 'show_summary_only')

    def get_timing_print_interval(self) -> int:
        """Get timing print interval (steps)"""
        interval = self.config.get('logging', {}).get('components', {}).get('timing', {}).get('print_interval', 10)
        return max(1, interval)

    def should_print_timing_this_step(self, step: int) -> bool:
        """Should print timing for this step?"""
        if not self.should_log_timing_breakdown() and not self.should_log_timing_summary():
            return False

        interval = self.get_timing_print_interval()
        if interval == 0:
            return True
        return step % interval == 0

    def log_timing_breakdown(self, message: str):
        """Log timing breakdown if enabled"""
        if self.should_log_timing_breakdown():
            print(message)

    def log_timing_summary(self, message: str):
        """Log timing summary if enabled"""
        if self.should_log_timing_summary():
            print(message)

    # Detection logging controls
    def should_log_detection_detail(self) -> bool:
        """Should log detection details?"""
        return self._get_component_setting('detection', 'show_detail')

    # Record file controls
    def should_rotate_record_file(self) -> bool:
        """Should enable record file rotation?"""
        return self._get_component_setting('record', 'enable_rotation')

    def get_max_record_size_bytes(self) -> int:
        """Get max record file size in bytes before rotation"""
        size_mb = self.config.get('logging', {}).get('components', {}).get('record', {}).get('max_size_mb', 5)
        return size_mb * 1024 * 1024

    def get_keep_rotations(self) -> int:
        """Get number of rotated files to keep"""
        return self.config.get('logging', {}).get('components', {}).get('record', {}).get('keep_rotations', 3)

    # General logging
    def log_info(self, message: str):
        """Log info message (respects MINIMAL and SILENT levels)"""
        if self.log_level not in [self.SILENT]:
            print(message)

    def log_episode_summary(self, message: str):
        """Log episode summary (shown in MINIMAL mode)"""
        if self.log_level != self.SILENT:
            print(message)

    def log_error(self, message: str):
        """Log error message (always shown)"""
        print(f"[ERROR] {message}")

    def increment_step(self):
        """Increment internal step counter"""
        self.step_counter += 1


# Global singleton instance
_log_manager_instance = None

def get_log_manager() -> LogManager:
    """Get global LogManager instance"""
    global _log_manager_instance
    if _log_manager_instance is None:
        _log_manager_instance = LogManager()
    return _log_manager_instance
