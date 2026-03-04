#!/bin/bash
# Auto-restart evaluation script for ApexNav
# Runs habitat_evaluation in batches with exploration_node restart
# Only restarts exploration.launch, keeps roscore and rviz running

set -e  # Exit on error

# Configuration
EPISODES_PER_BATCH=5        # Restart every N episodes
DATASET=${1:-hm3dv2}        # Default to hm3dv2
START_EPISODE=${2:-0}       # Starting episode number
TOTAL_EPISODES=${3:-1000}   # Total episodes to run

# Conda environment
CONDA_ENV="sg_apexnav"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  InfoNav Auto-Restart Evaluation Script       ║${NC}"
echo -e "${BLUE}╠════════════════════════════════════════════════╣${NC}"
echo -e "${BLUE}║  Dataset: ${DATASET}${NC}"
echo -e "${BLUE}║  Episodes per batch: ${EPISODES_PER_BATCH}${NC}"
echo -e "${BLUE}║  Start episode: ${START_EPISODE}${NC}"
echo -e "${BLUE}║  Total episodes: ${TOTAL_EPISODES}${NC}"
echo -e "${BLUE}║  Conda env: ${CONDA_ENV}${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
echo ""

# Check conda environment
if ! conda info --envs | grep -q "^${CONDA_ENV} "; then
    echo -e "${RED}[Error]${NC} Conda environment '${CONDA_ENV}' not found!"
    echo "Available environments:"
    conda info --envs
    exit 1
fi

# Check if already running
if pgrep -f "habitat_evaluation.py" > /dev/null; then
    echo -e "${RED}[Error]${NC} habitat_evaluation.py is already running!"
    echo "Please stop it first: pkill -f habitat_evaluation.py"
    exit 1
fi

# Function to check if ROS core is running
check_roscore() {
    if ! pgrep -x "rosmaster" > /dev/null; then
        echo -e "${RED}[Error]${NC} ROS master is not running!"
        echo "Please start: roscore"
        exit 1
    fi
    echo -e "${GREEN}[Check]${NC} ROS master is running"
}

# Function to kill exploration_node only
kill_exploration_node() {
    echo -e "${YELLOW}[Cleanup]${NC} Stopping exploration_node..."
    pkill -f exploration_node || true
    sleep 2

    # Force kill if still running
    if pgrep -f exploration_node > /dev/null; then
        echo -e "${YELLOW}[Cleanup]${NC} Force killing exploration_node..."
        pkill -9 -f exploration_node 2>/dev/null || true
        sleep 1
    fi

    echo -e "${GREEN}[Cleanup]${NC} exploration_node stopped"
}

# Function to restart exploration node
restart_exploration_node() {
    echo -e "${GREEN}[Restart]${NC} Launching exploration.launch..."

    # Source ROS workspace
    source ./devel/setup.bash

    # Start exploration.launch in background
    nohup roslaunch exploration_manager exploration.launch \
        > /tmp/exploration_node_batch_${batch_num}.log 2>&1 &

    # Wait for node to start
    echo -e "${YELLOW}[Wait]${NC} Waiting for exploration_node to initialize..."
    sleep 4

    # Check if node started successfully
    local max_attempts=10
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if pgrep -f exploration_node > /dev/null; then
            echo -e "${GREEN}[Restart]${NC} exploration_node started successfully"
            return 0
        fi
        attempt=$((attempt + 1))
        echo -e "${YELLOW}[Wait]${NC} Attempt $attempt/$max_attempts..."
        sleep 1
    done

    echo -e "${RED}[Error]${NC} Failed to start exploration_node after $max_attempts attempts"
    echo "Check log: /tmp/exploration_node_batch_${batch_num}.log"
    return 1
}

# Trap Ctrl+C to cleanup
trap 'echo -e "\n${RED}[Interrupt]${NC} Caught Ctrl+C, cleaning up..."; kill_exploration_node; exit 130' INT TERM

# Check prerequisites
check_roscore

# Source ROS workspace
echo -e "${GREEN}[Setup]${NC} Sourcing ROS workspace..."
source ./devel/setup.bash

# Main loop
completed_episodes=$START_EPISODE
batch_num=0

while [ $completed_episodes -lt $TOTAL_EPISODES ]; do
    batch_num=$((batch_num + 1))
    start_episode=$completed_episodes
    remaining=$((TOTAL_EPISODES - completed_episodes))

    # Calculate episodes for this batch
    if [ $remaining -lt $EPISODES_PER_BATCH ]; then
        episodes_this_batch=$remaining
    else
        episodes_this_batch=$EPISODES_PER_BATCH
    fi

    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Batch ${batch_num}: Episodes ${start_episode}-$((start_episode + episodes_this_batch - 1))${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════${NC}"

    # Restart exploration node for new batch (except first batch)
    if [ $batch_num -gt 1 ]; then
        kill_exploration_node
        if ! restart_exploration_node; then
            echo -e "${RED}[Error]${NC} Cannot continue without exploration_node"
            exit 1
        fi
        # Extra wait after restart to ensure everything is ready
        echo -e "${YELLOW}[Wait]${NC} Waiting for system stabilization..."
        sleep 2
    else
        # First batch: check if exploration_node is running
        if ! pgrep -f exploration_node > /dev/null; then
            echo -e "${YELLOW}[Warning]${NC} exploration_node not running, starting it..."
            if ! restart_exploration_node; then
                echo -e "${RED}[Error]${NC} Cannot start exploration_node"
                exit 1
            fi
        else
            echo -e "${GREEN}[Check]${NC} exploration_node is already running"
        fi
    fi

    # Run habitat evaluation for this batch using conda
    echo -e "${GREEN}[Running]${NC} Starting evaluation batch ${batch_num}..."

    # Run in conda environment
    conda run -n ${CONDA_ENV} --no-capture-output \
        python habitat_evaluation.py \
        --dataset ${DATASET} \
        test_epi_num=${start_episode} \
        2>&1 | tee -a evaluation_batch_${batch_num}.log

    # Check if evaluation succeeded
    eval_exit_code=${PIPESTATUS[0]}
    if [ $eval_exit_code -eq 0 ]; then
        completed_episodes=$((completed_episodes + episodes_this_batch))
        echo -e "${GREEN}[Success]${NC} Batch ${batch_num} completed (${completed_episodes}/${TOTAL_EPISODES} episodes done)"
    else
        echo -e "${RED}[Error]${NC} Batch ${batch_num} failed with exit code $eval_exit_code!"
        echo "Check log: evaluation_batch_${batch_num}.log"
        echo "Last exploration_node log: /tmp/exploration_node_batch_${batch_num}.log"
        kill_exploration_node
        exit 1
    fi

    # Sleep between batches (if not last batch)
    if [ $completed_episodes -lt $TOTAL_EPISODES ]; then
        echo -e "${YELLOW}[Wait]${NC} Batch complete, preparing for next batch..."
        sleep 2
    fi
done

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          All batches completed!                ║${NC}"
echo -e "${GREEN}║  Total episodes: ${completed_episodes}/${TOTAL_EPISODES}${NC}"
echo -e "${GREEN}║  Batches run: ${batch_num}${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════╝${NC}"

echo -e "${BLUE}[Info]${NC} exploration_node is still running"
echo -e "${BLUE}[Info]${NC} To stop it: pkill -f exploration_node"
echo -e "${GREEN}[Done]${NC} Evaluation complete!"
