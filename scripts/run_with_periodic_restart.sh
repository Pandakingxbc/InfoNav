#!/bin/bash
###############################################################################
#                    InfoNav Periodic Restart Evaluation Script
###############################################################################
#
# USAGE:
#   ./run_with_periodic_restart.sh [EPISODES_PER_BATCH] [DATASET] [MAX_BATCHES] [STUCK_TIMEOUT]
#
# PARAMETERS:
#   EPISODES_PER_BATCH  - Number of episodes to run before restarting (default: 10)
#   DATASET             - Dataset name to use (default: hm3dv2)
#   MAX_BATCHES         - Maximum number of batches to run (default: 200)
#   STUCK_TIMEOUT       - Seconds without progress before auto-restart (default: 180)
#
# EXAMPLES:
#   # Run with all defaults (10 episodes per batch, hm3dv2 dataset)
#   ./run_with_periodic_restart.sh
#
#   # Run 5 episodes per batch
#   ./run_with_periodic_restart.sh 5
#
#   # Run 20 episodes per batch on mp3d dataset
#   ./run_with_periodic_restart.sh 20 mp3d
#
#   # Full custom: 15 episodes, hm3dv2, max 100 batches, 300s stuck timeout
#   ./run_with_periodic_restart.sh 15 hm3dv2 100 300
#
# PREREQUISITES:
#   1. ROS master must be running:  roscore
#   2. Conda environment 'infonav' must exist
#   3. Run from the catkin workspace root (where devel/setup.bash exists)
#
# FEATURES:
#   - Automatic checkpoint resume (uses habitat_evaluation.py's continue.txt)
#   - Auto-restart exploration_node when agent gets stuck
#   - Graceful shutdown on Ctrl+C
#   - Colored progress output
#
###############################################################################
# Simple periodic restart script for InfoNav evaluation
# Runs habitat_evaluation.py in batches with exploration.launch restart
# Leverages habitat_evaluation.py's built-in checkpoint mechanism

set -e

# Configuration
EPISODES_PER_BATCH=${1:-10}     # How many episodes before restart (default: 5)
DATASET=${2:-hm3dv2}            # Dataset to use
MAX_BATCHES=${3:-200}           # Maximum number of batches (safety limit)
STUCK_TIMEOUT=${4:-180}         # Seconds without progress before restart (default: 3 minutes)

# Conda environment
CONDA_ENV="infonav"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   ApexNav Periodic Restart Evaluation         ║${NC}"
echo -e "${BLUE}╠════════════════════════════════════════════════╣${NC}"
echo -e "${BLUE}║  Dataset: ${DATASET}${NC}"
echo -e "${BLUE}║  Episodes per batch: ${EPISODES_PER_BATCH}${NC}"
echo -e "${BLUE}║  Stuck timeout: ${STUCK_TIMEOUT}s${NC}"
echo -e "${BLUE}║  Conda env: ${CONDA_ENV}${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
echo ""

# Check prerequisites
if ! pgrep -x "rosmaster" > /dev/null; then
    echo -e "${RED}[Error]${NC} ROS master is not running! Start: roscore"
    exit 1
fi

if ! conda info --envs | grep -q "^${CONDA_ENV} "; then
    echo -e "${RED}[Error]${NC} Conda environment '${CONDA_ENV}' not found!"
    exit 1
fi

# Function to get current episode count from continue.txt (habitat's checkpoint file)
get_current_episode() {
    local continue_file="videos/test_${DATASET}_val/continue.txt"

    if [ ! -f "$continue_file" ]; then
        echo "0"
        return
    fi

    # Extract num_total from continue.txt (same as habitat_evaluation.py's read_record())
    # Pattern: "No.266 task is finished" -> extract 266
    local count=$(grep -oP "No\.\K\d+(?= task is finished)" "$continue_file" | head -1)

    if [ -z "$count" ]; then
        echo "0"
    else
        echo "$count"
    fi
}

# Function to kill exploration_node
kill_exploration() {
    echo -e "${YELLOW}[Cleanup]${NC} Stopping exploration_node..."
    pkill -f exploration_node 2>/dev/null || true
    sleep 2
    pkill -9 -f exploration_node 2>/dev/null || true
    sleep 1
    echo -e "${GREEN}[Cleanup]${NC} exploration_node stopped"
}

# Function to kill habitat_evaluation.py
kill_evaluation() {
    echo -e "${YELLOW}[Cleanup]${NC} Stopping habitat_evaluation.py..."
    pkill -INT -f "habitat_evaluation.py" 2>/dev/null || true
    sleep 3
    pkill -9 -f "habitat_evaluation.py" 2>/dev/null || true
    sleep 1
    echo -e "${GREEN}[Cleanup]${NC} habitat_evaluation.py stopped"
}

# Function to check if agent is stuck by monitoring ROS action topic
check_agent_stuck() {
    # Check if /habitat/plan_action topic has any messages in the last few seconds
    # Returns 0 if stuck (no messages), 1 if active
    local timeout_check=5
    local result=$(timeout ${timeout_check}s rostopic echo -n 1 /habitat/plan_action 2>/dev/null)
    if [ -z "$result" ]; then
        return 0  # Stuck - no action received
    else
        return 1  # Active - action received
    fi
}

# Function to start exploration_node
start_exploration() {
    echo -e "${GREEN}[Start]${NC} Launching exploration.launch..."
    source ./devel/setup.bash

    nohup roslaunch exploration_manager exploration.launch \
        > /tmp/exploration_restart_$$.log 2>&1 &

    local pid=$!
    echo -e "${YELLOW}[Wait]${NC} Waiting for exploration_node to initialize..."
    sleep 4

    # Verify it started
    for i in {1..10}; do
        if pgrep -f exploration_node > /dev/null; then
            echo -e "${GREEN}[Start]${NC} exploration_node started (PID: $(pgrep -f exploration_node))"
            return 0
        fi
        sleep 1
    done

    echo -e "${RED}[Error]${NC} Failed to start exploration_node"
    return 1
}

# Function to run evaluation for N episodes
run_evaluation_batch() {
    local start_count=$1
    local continue_file="videos/test_${DATASET}_val/continue.txt"
    echo -e "${GREEN}[Run]${NC} Starting evaluation batch from episode ${start_count}..."
    echo -e "${BLUE}[Info]${NC} This batch will run ${EPISODES_PER_BATCH} episodes"

    # Safety check: ensure no other habitat_evaluation.py is running
    if pgrep -f "habitat_evaluation.py" > /dev/null; then
        echo -e "${RED}[Error]${NC} Another habitat_evaluation.py is already running!"
        echo -e "${YELLOW}[Info]${NC} Waiting for it to finish (max 60 seconds)..."

        local wait_count=0
        while pgrep -f "habitat_evaluation.py" > /dev/null; do
            sleep 2
            wait_count=$((wait_count + 1))

            # Timeout after 30 attempts (60 seconds)
            if [ $wait_count -ge 30 ]; then
                echo -e "${RED}[Timeout]${NC} Previous instance still running after 60s, force killing..."
                pkill -9 -f "habitat_evaluation.py"
                sleep 2
                break
            fi
        done
        echo -e "${GREEN}[Info]${NC} Previous instance finished"
    fi

    # Start habitat_evaluation in background
    conda run -n ${CONDA_ENV} --no-capture-output \
        python habitat_evaluation.py --dataset ${DATASET} &

    local eval_pid=$!
    echo -e "${YELLOW}[Monitor]${NC} habitat_evaluation.py running (PID: ${eval_pid})"

    # Monitor progress
    local target_count=$((start_count + EPISODES_PER_BATCH))
    local current_count=$start_count

    local prev_count=$start_count
    local last_activity_time=$(date +%s)
    local stuck_check_interval=30  # Check for stuck every 30 seconds

    while kill -0 $eval_pid 2>/dev/null; do
        current_count=$(get_current_episode)
        local current_time=$(date +%s)

        # Only check for batch completion if count has increased (episode just finished)
        if [ $current_count -gt $prev_count ]; then
            echo -e "${BLUE}[Progress]${NC} Completed task ${current_count}"
            prev_count=$current_count
            last_activity_time=$current_time  # Reset activity timer

            # Check if we've reached target after episode completion
            if [ $current_count -ge $target_count ]; then
                echo -e "${GREEN}[Batch Complete]${NC} Reached target: ${current_count} episodes"
                echo -e "${YELLOW}[Wait]${NC} Waiting for current episode to finish (if any)..."

                # Wait for continue.txt to stop being modified (episode has fully completed)
                # Check if file is idle for 10 consecutive seconds
                local idle_count=0
                local last_mtime=$(stat -c %Y "$continue_file" 2>/dev/null || echo "0")

                while [ $idle_count -lt 2 ]; do
                    sleep 5
                    local current_mtime=$(stat -c %Y "$continue_file" 2>/dev/null || echo "0")

                    if [ "$current_mtime" = "$last_mtime" ]; then
                        idle_count=$((idle_count + 1))
                        echo -e "${BLUE}[Wait]${NC} File idle for $((idle_count * 5)) seconds..."
                    else
                        idle_count=0
                        last_mtime=$current_mtime
                        echo -e "${YELLOW}[Wait]${NC} Episode still writing, waiting..."
                    fi

                    # Safety: max 30 seconds total wait
                    if [ $idle_count -ge 6 ]; then
                        break
                    fi
                done

                echo -e "${YELLOW}[Stop]${NC} Stopping habitat_evaluation.py gracefully..."

                # Send SIGINT (Ctrl+C) for graceful shutdown
                kill -INT $eval_pid 2>/dev/null || true

                # Wait up to 30 seconds for graceful shutdown
                for i in {1..30}; do
                    if ! kill -0 $eval_pid 2>/dev/null; then
                        echo -e "${GREEN}[Stop]${NC} habitat_evaluation.py stopped gracefully"
                        # Additional wait to ensure all file I/O is complete
                        sleep 2
                        return 0
                    fi
                    sleep 1
                done

                # Force kill if still running
                echo -e "${YELLOW}[Stop]${NC} Force killing habitat_evaluation.py..."
                kill -9 $eval_pid 2>/dev/null || true
                sleep 2
                return 0
            fi
        fi

        # Check for stuck agent
        local time_since_activity=$((current_time - last_activity_time))
        if [ $time_since_activity -ge $STUCK_TIMEOUT ]; then
            echo -e "${RED}[Stuck Detected]${NC} No progress for ${time_since_activity} seconds (timeout: ${STUCK_TIMEOUT}s)"

            # Double-check by monitoring ROS action topic
            echo -e "${YELLOW}[Check]${NC} Verifying agent status via ROS topic..."
            if check_agent_stuck; then
                echo -e "${RED}[Confirmed]${NC} Agent is stuck - no ROS actions received"
                echo -e "${YELLOW}[Recovery]${NC} Restarting exploration_node..."

                # Kill evaluation and exploration
                kill_evaluation
                kill_exploration
                sleep 2

                # Restart exploration
                if ! start_exploration; then
                    echo -e "${RED}[Error]${NC} Failed to restart exploration_node"
                    return 1
                fi
                sleep 3

                # Return to trigger a new batch (episode will continue from checkpoint)
                echo -e "${GREEN}[Recovery]${NC} Exploration restarted, continuing from checkpoint..."
                return 0
            else
                echo -e "${GREEN}[OK]${NC} Agent is active, resetting timer"
                last_activity_time=$current_time
            fi
        fi

        # Check every 5 seconds instead of 2 to reduce race conditions
        sleep 5
    done

    # Process exited on its own
    wait $eval_pid
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}[Complete]${NC} habitat_evaluation.py completed all episodes"
        return 0
    else
        echo -e "${RED}[Error]${NC} habitat_evaluation.py exited with code ${exit_code}"
        return 1
    fi
}

# Trap Ctrl+C
trap 'echo -e "\n${RED}[Interrupt]${NC} Stopping..."; kill_exploration; exit 130' INT TERM

# Check if exploration_node is already running
if ! pgrep -f exploration_node > /dev/null; then
    echo -e "${YELLOW}[Warning]${NC} exploration_node not running, starting it..."
    start_exploration || exit 1
else
    echo -e "${GREEN}[Check]${NC} exploration_node is already running"
fi

# Main loop
batch_num=0
while [ $batch_num -lt $MAX_BATCHES ]; do
    batch_num=$((batch_num + 1))

    # Get current progress
    current_episodes=$(get_current_episode)

    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Batch ${batch_num}: Starting from episode ${current_episodes}${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════${NC}"

    # Run this batch
    if ! run_evaluation_batch $current_episodes; then
        echo -e "${RED}[Error]${NC} Batch ${batch_num} failed"
        break
    fi

    # Check if we're done (all episodes completed)
    new_count=$(get_current_episode)
    if [ $new_count -eq $current_episodes ]; then
        echo -e "${GREEN}[Done]${NC} No new episodes completed, evaluation finished"
        break
    fi

    echo -e "${GREEN}[Progress]${NC} Completed ${new_count} total episodes"

    # 10 second pause between batches
    echo -e "${YELLOW}[Pause]${NC} Waiting 10 seconds before next batch..."
    sleep 10

    # Restart exploration_node for next batch
    echo -e "${BLUE}[Restart]${NC} Preparing for next batch..."
    kill_exploration
    sleep 2

    if ! start_exploration; then
        echo -e "${RED}[Error]${NC} Failed to restart exploration_node"
        exit 1
    fi

    sleep 2
done

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║            Evaluation Complete                 ║${NC}"
echo -e "${GREEN}║  Total episodes: $(get_current_episode)${NC}"
echo -e "${GREEN}║  Batches run: ${batch_num}${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════╝${NC}"

echo -e "${BLUE}[Info]${NC} exploration_node is still running"
echo -e "${BLUE}[Info]${NC} To stop: pkill -f exploration_node"
echo -e "${GREEN}[Done]${NC}"
