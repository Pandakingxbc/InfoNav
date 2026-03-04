#!/bin/bash

# ApexNav VLM Servers Manager using tmux
# Usage: ./start_vlm_servers.sh [start|stop|restart|status] [env_name]
# Example: ./start_vlm_servers.sh start infonav

#运行查看脚本的用户必须确保已安装tmux并且已创建所需的conda环境。
#运行
#tmux capture-pane -t vlm_servers:blip2_itm -p
#tmux capture-pane -t vlm_servers:grounding_dino -p
#tmux capture-pane -t vlm_servers:sam -p
#tmux capture-pane -t vlm_servers:dfine -p

#以查看各个服务器的输出。
#./scripts/start_vlm_servers.sh status

set -e

# Configuration
SESSION_NAME="vlm_servers"
CONDA_ENV="${2:-infonav}"  # Default to sg_apexnav, can be overridden with second argument

# Check if parallel BLIP2 is requested
USE_PARALLEL_BLIP2="${USE_PARALLEL_BLIP2:-false}"

if [ "$USE_PARALLEL_BLIP2" = "true" ]; then
    # 6 servers mode: Add second BLIP2 for parallel inference (2x speedup) + yolov7
    SERVERS=(
        "grounding_dino:12181:vlm.detector.grounding_dino"
        "blip2_1:12182:vlm.itm.blip2itm"
        "blip2_2:12192:vlm.itm.blip2itm"
        "sam:12183:vlm.segmentor.sam"
        "yolov7:12184:vlm.detector.yolov7"
        "dfine:12185:vlm.detector.dfine"
    )
else
    # 5 servers mode: Standard configuration + yolov7
    SERVERS=(
        "grounding_dino:12181:vlm.detector.grounding_dino"
        "blip2_itm:12182:vlm.itm.blip2itm"
        "sam:12183:vlm.segmentor.sam"
        "yolov7:12184:vlm.detector.yolov7"
        "dfine:12185:vlm.detector.dfine"
    )
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if tmux is installed
check_tmux() {
    if ! command -v tmux &> /dev/null; then
        echo -e "${RED}Error: tmux is not installed. Please install tmux first.${NC}"
        echo "  Ubuntu/Debian: sudo apt-get install tmux"
        echo "  MacOS: brew install tmux"
        exit 1
    fi
}

# Check if conda environment exists
check_conda_env() {
    if ! conda env list | grep -q "^$CONDA_ENV "; then
        echo -e "${RED}Error: Conda environment '$CONDA_ENV' not found.${NC}"
        echo "Available environments:"
        conda env list
        exit 1
    fi
}

# Start all VLM servers
start_servers() {
    check_tmux
    check_conda_env
    
    # Check if session already exists
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo -e "${YELLOW}Session '$SESSION_NAME' already exists. Use 'restart' to restart servers.${NC}"
        return 1
    fi
    
    echo -e "${BLUE}Creating tmux session: $SESSION_NAME${NC}"
    echo -e "${BLUE}Using conda environment: $CONDA_ENV${NC}"

    # Create tmux session with first pane
    tmux new-session -d -s "$SESSION_NAME" -x 200 -y 50

    # Layout depends on number of servers
    num_servers=${#SERVERS[@]}

    if [ "$num_servers" -eq 6 ]; then
        echo -e "${BLUE}Layout: 6-pane split view (2x3 grid)${NC}"
        echo -e "${YELLOW}Parallel BLIP2 mode enabled (2x speedup) + YOLOv7${NC}"

        # Create 6 panes: 2x3 grid
        # Split horizontally to get 2 columns
        tmux split-window -h -t "$SESSION_NAME"
        # Split left column into 3 rows
        tmux split-window -v -t "$SESSION_NAME:0.0"
        tmux split-window -v -t "$SESSION_NAME:0.0"
        # Split right column into 3 rows
        tmux split-window -v -t "$SESSION_NAME:0.3"
        tmux split-window -v -t "$SESSION_NAME:0.3"

        # Layout: 6 panes in 2x3 grid
    elif [ "$num_servers" -eq 5 ]; then
        echo -e "${BLUE}Layout: 5-pane split view (2x2 + 1)${NC}"
        echo -e "${YELLOW}Standard mode + YOLOv7${NC}"

        # Create 5 panes: 2x2 grid + bottom center
        # Split horizontally to get 2 columns
        tmux split-window -h -t "$SESSION_NAME"
        # Split left column vertically (panes 0,1)
        tmux split-window -v -t "$SESSION_NAME:0.0"
        # Split right column vertically (panes 2,3)
        tmux split-window -v -t "$SESSION_NAME:0.2"
        # Add 5th pane at bottom center
        tmux split-window -v -t "$SESSION_NAME:0.1"

        # Layout: Pane 0 (top-left), Pane 1 (bottom-left), Pane 2 (top-right), Pane 3 (bottom-right), Pane 4 (bottom-center)
    else
        echo -e "${BLUE}Layout: 4-pane split view (2x2 grid)${NC}"

        # Split into 4 panes (2x2 grid)
        tmux split-window -h -t "$SESSION_NAME"
        tmux split-window -v -t "$SESSION_NAME:0.0"
        tmux split-window -v -t "$SESSION_NAME:0.2"

        # Layout: Pane 0 (top-left), Pane 1 (bottom-left), Pane 2 (top-right), Pane 3 (bottom-right)
    fi

    # Start each server in a different pane
    local pane_index=0
    for server_config in "${SERVERS[@]}"; do
        IFS=':' read -r server_name port module <<< "$server_config"

        echo -e "${BLUE}Starting $server_name on port $port in pane $pane_index...${NC}"

        # Set pane title and start server
        tmux select-pane -t "$SESSION_NAME:0.$pane_index" -T "$server_name:$port"
        tmux send-keys -t "$SESSION_NAME:0.$pane_index" "echo '=== $server_name (port $port) ==='" Enter
        tmux send-keys -t "$SESSION_NAME:0.$pane_index" "conda activate $CONDA_ENV" Enter
        sleep 0.5
        tmux send-keys -t "$SESSION_NAME:0.$pane_index" "python -m $module --port $port" Enter

        sleep 1
        pane_index=$((pane_index + 1))
    done

    # Select the first pane
    tmux select-pane -t "$SESSION_NAME:0.0"

    echo -e "${GREEN}✓ All VLM servers have been started successfully!${NC}"
    echo -e "${YELLOW}Conda environment: $CONDA_ENV${NC}"

    if [ "$USE_PARALLEL_BLIP2" = "true" ]; then
        echo -e "${GREEN}Parallel BLIP2 mode: ENABLED${NC} - Expected 2x speedup!"
    fi

    echo ""
    echo -e "${BLUE}Available commands:${NC}"
    echo -e "  Attach to session:  ${GREEN}tmux attach -t $SESSION_NAME${NC}"
    echo -e "  Stop servers:       ${GREEN}./start_vlm_servers.sh stop${NC}"
    echo -e "  Restart servers:    ${GREEN}./start_vlm_servers.sh restart [env_name]${NC}"
    echo -e "  Check status:       ${GREEN}./start_vlm_servers.sh status${NC}"
    echo ""
    echo -e "${YELLOW}Server URLs:${NC}"
    for server_config in "${SERVERS[@]}"; do
        IFS=':' read -r server_name port module <<< "$server_config"
        echo -e "  $server_name: http://localhost:$port"
    done
    echo ""

    # Show appropriate layout diagram
    if [ "$USE_PARALLEL_BLIP2" = "true" ]; then
        echo -e "${BLUE}Pane Layout (2x3, Parallel BLIP2 + YOLOv7):${NC}"
        echo -e "  ┌─────────────────┬─────────────────┐"
        echo -e "  │ grounding_dino  │ sam             │"
        echo -e "  │ (12181)         │ (12183)         │"
        echo -e "  ├─────────────────┼─────────────────┤"
        echo -e "  │ blip2_1 ★       │ yolov7          │"
        echo -e "  │ (12182)         │ (12184)         │"
        echo -e "  ├─────────────────┼─────────────────┤"
        echo -e "  │ blip2_2 ★       │ dfine           │"
        echo -e "  │ (12192)         │ (12185)         │"
        echo -e "  └─────────────────┴─────────────────┘"
    else
        echo -e "${BLUE}Pane Layout (2x2 + 1, Standard + YOLOv7):${NC}"
        echo -e "  ┌─────────────────┬─────────────────┐"
        echo -e "  │ grounding_dino  │ sam             │"
        echo -e "  │ (12181)         │ (12183)         │"
        echo -e "  ├─────────────────┼─────────────────┤"
        echo -e "  │ blip2_itm       │ yolov7          │"
        echo -e "  │ (12182)         │ (12184)         │"
        echo -e "  ├─────────────────┴─────────────────┤"
        echo -e "  │ dfine                             │"
        echo -e "  │ (12185)                           │"
        echo -e "  └───────────────────────────────────┘"
    fi

    echo ""
    echo -e "${YELLOW}tmux navigation tips:${NC}"
    echo -e "  Switch panes: ${GREEN}Ctrl+b${NC} then arrow keys (↑↓←→)"
    echo -e "  Zoom pane:    ${GREEN}Ctrl+b${NC} then ${GREEN}z${NC} (toggle fullscreen for selected pane)"
    echo -e "  Detach:       ${GREEN}Ctrl+b${NC} then ${GREEN}d${NC}"
    sleep 2
    echo ""
    echo -e "${YELLOW}All servers are initializing. Attach to session to monitor progress.${NC}"
}

# Stop all servers
stop_servers() {
    check_tmux
    
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo -e "${YELLOW}Session '$SESSION_NAME' is not running.${NC}"
        return 0
    fi
    
    echo -e "${BLUE}Stopping all VLM servers...${NC}"
    tmux kill-session -t "$SESSION_NAME"
    
    echo -e "${GREEN}✓ All VLM servers have been stopped.${NC}"
}

# Restart all servers
restart_servers() {
    echo -e "${BLUE}Restarting VLM servers...${NC}"
    stop_servers
    sleep 1
    start_servers
}

# Check status of servers
status_servers() {
    check_tmux
    
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo -e "${RED}✗ Session '$SESSION_NAME' is not running.${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✓ Session '$SESSION_NAME' is running.${NC}"
    echo ""
    echo -e "${BLUE}Active windows:${NC}"
    tmux list-windows -t "$SESSION_NAME" -F "#{window_name} (#{window_panes} pane)"
    echo ""
    echo -e "${BLUE}To attach to the session: tmux attach -t $SESSION_NAME${NC}"
}

# Main logic
case "${1:-start}" in
    start)
        start_servers
        ;;
    stop)
        stop_servers
        ;;
    restart)
        restart_servers
        ;;
    status)
        status_servers
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status} [env_name]"
        echo ""
        echo "Commands:"
        echo "  start   - Start all VLM servers (default)"
        echo "  stop    - Stop all VLM servers"
        echo "  restart - Restart all VLM servers"
        echo "  status  - Show status of VLM servers"
        echo ""
        echo "Examples:"
        echo "  $0 start                        # Use default sg_apexnav environment"
        echo "  $0 start sg_apexnav             # Use sg_apexnav environment (same as default)"
        echo "  $0 restart apexnav              # Restart with apexnav environment"
        exit 1
        ;;
esac
