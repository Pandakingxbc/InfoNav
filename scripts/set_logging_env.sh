#!/bin/bash
# Set environment variables for ApexNav logging control
#
# Usage:
#   source scripts/set_logging_env.sh MINIMAL
#   source scripts/set_logging_env.sh NORMAL
#   source scripts/set_logging_env.sh VERBOSE
#   source scripts/set_logging_env.sh SILENT

LOG_LEVEL=${1:-NORMAL}

echo "Setting ApexNav logging level to: $LOG_LEVEL"

# Set Python logging level
export APEXNAV_LOG_LEVEL=$LOG_LEVEL

# Configure component-specific settings based on level
case $LOG_LEVEL in
  VERBOSE)
    export APEXNAV_VLM_DETAIL=true
    export APEXNAV_SEMANTIC_SCORES=true
    export APEXNAV_TIMING_DETAIL=true
    export ROSCONSOLE_CONFIG_FILE=""  # Use default (INFO)
    echo "  - VLM Detail: ON"
    echo "  - Semantic Scores: ON"
    echo "  - Timing Detail: ON"
    echo "  - ROS Console: INFO"
    ;;

  NORMAL)
    export APEXNAV_VLM_DETAIL=false
    export APEXNAV_SEMANTIC_SCORES=false
    export APEXNAV_TIMING_DETAIL=false
    export ROSCONSOLE_CONFIG_FILE=""  # Use default (INFO)
    echo "  - VLM Detail: OFF"
    echo "  - Semantic Scores: OFF"
    echo "  - Timing Detail: OFF (summary only)"
    echo "  - ROS Console: INFO"
    ;;

  MINIMAL)
    export APEXNAV_VLM_DETAIL=false
    export APEXNAV_SEMANTIC_SCORES=false
    export APEXNAV_TIMING_DETAIL=false
    # Create minimal ROS console config
    cat > /tmp/rosconsole_minimal.config << 'EOF'
log4j.logger.ros=WARN
log4j.logger.ros.plan_env=WARN
log4j.logger.ros.exploration_manager=WARN
EOF
    export ROSCONSOLE_CONFIG_FILE=/tmp/rosconsole_minimal.config
    echo "  - VLM Detail: OFF"
    echo "  - Semantic Scores: OFF"
    echo "  - Timing Detail: OFF"
    echo "  - ROS Console: WARN"
    ;;

  SILENT)
    export APEXNAV_VLM_DETAIL=false
    export APEXNAV_SEMANTIC_SCORES=false
    export APEXNAV_TIMING_DETAIL=false
    # Create silent ROS console config
    cat > /tmp/rosconsole_silent.config << 'EOF'
log4j.logger.ros=ERROR
log4j.logger.ros.plan_env=ERROR
log4j.logger.ros.exploration_manager=ERROR
EOF
    export ROSCONSOLE_CONFIG_FILE=/tmp/rosconsole_silent.config
    echo "  - VLM Detail: OFF"
    echo "  - Semantic Scores: OFF"
    echo "  - Timing Detail: OFF"
    echo "  - ROS Console: ERROR"
    ;;

  *)
    echo "Unknown log level: $LOG_LEVEL"
    echo "Valid options: VERBOSE, NORMAL, MINIMAL, SILENT"
    return 1
    ;;
esac

echo ""
echo "Environment configured successfully!"
echo "You can now run your evaluation with reduced logging."
