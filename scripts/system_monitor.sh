#!/bin/bash

# Comprehensive System Monitor for ApexNav
# Shows: VLM servers, GPU, habitat_evaluation process, disk usage

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

while true; do
    clear
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  ${GREEN}ApexNav System Monitor${NC} - $(date '+%Y-%m-%d %H:%M:%S')                    ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # ═══════════════════════════════════════════════════════════════════════
    # VLM Servers Status
    # ═══════════════════════════════════════════════════════════════════════
    echo -e "${YELLOW}┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓${NC}"
    echo -e "${YELLOW}┃${NC} ${CYAN}VLM Servers${NC}                                                       ${YELLOW}┃${NC}"
    echo -e "${YELLOW}┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛${NC}"

    if tmux has-session -t vlm_servers 2>/dev/null; then
        echo -e "${GREEN}  ✓ VLM servers session is running${NC}"
        echo ""

        # Get process info
        vlm_procs=$(ps aux | grep -E "python.*vlm\.(detector|itm|segmentor)" | grep -v grep)

        if [ ! -z "$vlm_procs" ]; then
            echo -e "${BLUE}  Server        Port   CPU%   MEM%   RSS(MB)  Status${NC}"
            echo -e "${BLUE}  ────────────────────────────────────────────────────${NC}"

            # Parse each VLM process
            echo "$vlm_procs" | while IFS= read -r line; do
                cpu=$(echo "$line" | awk '{print $3}')
                mem=$(echo "$line" | awk '{print $4}')
                rss=$(echo "$line" | awk '{print int($6/1024)}')
                cmd=$(echo "$line" | awk '{for(i=11;i<=NF;i++) printf $i" "; print ""}')

                # Extract server name and port
                if [[ $cmd == *"grounding_dino"* ]]; then
                    server="GroundingDINO"
                    port="12181"
                elif [[ $cmd == *"blip2itm"* ]]; then
                    server="BLIP2 ITM ★"
                    port="12182"
                elif [[ $cmd == *"sam"* ]]; then
                    server="SAM"
                    port="12183"
                elif [[ $cmd == *"dfine"* ]]; then
                    server="D-FINE"
                    port="12185"
                else
                    server="Unknown"
                    port="?????"
                fi

                # Color code by CPU usage
                if (( $(echo "$cpu > 50" | bc -l) )); then
                    color="${RED}"
                elif (( $(echo "$cpu > 20" | bc -l) )); then
                    color="${YELLOW}"
                else
                    color="${GREEN}"
                fi

                printf "  ${color}%-14s${NC} %-6s %-6s %-6s %-8s ${GREEN}Running${NC}\n" \
                    "$server" "$port" "$cpu%" "$mem%" "${rss}MB"
            done
        fi
    else
        echo -e "${RED}  ✗ VLM servers are NOT running${NC}"
        echo -e "${YELLOW}    Start with: ./scripts/start_vlm_servers.sh start${NC}"
    fi
    echo ""

    # ═══════════════════════════════════════════════════════════════════════
    # GPU Status
    # ═══════════════════════════════════════════════════════════════════════
    echo -e "${YELLOW}┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓${NC}"
    echo -e "${YELLOW}┃${NC} ${CYAN}GPU Status${NC}                                                        ${YELLOW}┃${NC}"
    echo -e "${YELLOW}┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛${NC}"

    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null | \
        awk -F', ' '{
            util = $3
            mem_used = $4
            mem_total = $5
            mem_pct = int(mem_used * 100 / mem_total)
            temp = $6

            # Color by utilization
            if (util > 80) util_color = "\033[0;31m"  # Red
            else if (util > 50) util_color = "\033[1;33m"  # Yellow
            else util_color = "\033[0;32m"  # Green

            # Color by memory
            if (mem_pct > 90) mem_color = "\033[0;31m"
            else if (mem_pct > 70) mem_color = "\033[1;33m"
            else mem_color = "\033[0;32m"

            printf "  GPU %s: %s\n", $1, $2
            printf "    Utilization: %s%3s%%%s  |  Memory: %s%5s/%5s MB (%3s%%)%s  |  Temp: %2s°C\n",
                util_color, util, "\033[0m",
                mem_color, mem_used, mem_total, mem_pct, "\033[0m",
                temp
        }'

    # GPU Processes
    gpu_procs=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null)
    if [ ! -z "$gpu_procs" ]; then
        echo ""
        echo -e "${BLUE}  GPU Processes:${NC}"
        echo "$gpu_procs" | awk -F', ' '{
            pid = $1
            name = $2
            mem = $3

            # Highlight BLIP2
            if (index(name, "python") > 0) {
                printf "    ${GREEN}PID %-7s${NC} %-20s ${YELLOW}%5s MB${NC}\n", pid, name, mem
            } else {
                printf "    PID %-7s %-20s %5s MB\n", pid, name, mem
            }
        }'
    fi
    echo ""

    # ═══════════════════════════════════════════════════════════════════════
    # Habitat Evaluation Process
    # ═══════════════════════════════════════════════════════════════════════
    echo -e "${YELLOW}┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓${NC}"
    echo -e "${YELLOW}┃${NC} ${CYAN}Habitat Evaluation${NC}                                                ${YELLOW}┃${NC}"
    echo -e "${YELLOW}┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛${NC}"

    habitat_proc=$(ps aux | grep "habitat_evaluation.py" | grep -v grep)
    if [ ! -z "$habitat_proc" ]; then
        echo -e "${GREEN}  ✓ habitat_evaluation.py is running${NC}"
        echo ""

        pid=$(echo "$habitat_proc" | awk '{print $2}')
        cpu=$(echo "$habitat_proc" | awk '{print $3}')
        mem=$(echo "$habitat_proc" | awk '{print $4}')
        rss=$(echo "$habitat_proc" | awk '{print int($6/1024)}')
        runtime=$(ps -p $pid -o etime= | tr -d ' ')

        echo -e "${BLUE}  PID:${NC}     $pid"
        echo -e "${BLUE}  CPU:${NC}    ${cpu}%"
        echo -e "${BLUE}  Memory:${NC} ${mem}% (${rss}MB RSS)"
        echo -e "${BLUE}  Runtime:${NC} $runtime"
    else
        echo -e "${YELLOW}  ○ habitat_evaluation.py is NOT running${NC}"
    fi
    echo ""

    # ═══════════════════════════════════════════════════════════════════════
    # ROS Nodes
    # ═══════════════════════════════════════════════════════════════════════
    echo -e "${YELLOW}┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓${NC}"
    echo -e "${YELLOW}┃${NC} ${CYAN}ROS Nodes${NC}                                                         ${YELLOW}┃${NC}"
    echo -e "${YELLOW}┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛${NC}"

    ros_proc=$(ps aux | grep "exploration_node\|tsp_node" | grep -v grep)
    if [ ! -z "$ros_proc" ]; then
        echo "$ros_proc" | while IFS= read -r line; do
            cpu=$(echo "$line" | awk '{print $3}')
            mem=$(echo "$line" | awk '{print $4}')

            if [[ $line == *"exploration_node"* ]]; then
                echo -e "${GREEN}  ✓ exploration_node${NC}  CPU: ${cpu}%  MEM: ${mem}%"
            elif [[ $line == *"tsp_node"* ]]; then
                echo -e "${GREEN}  ✓ tsp_node${NC}          CPU: ${cpu}%  MEM: ${mem}%"
            fi
        done
    else
        echo -e "${YELLOW}  ○ ROS nodes not detected${NC}"
    fi
    echo ""

    # ═══════════════════════════════════════════════════════════════════════
    # Disk & Memory
    # ═══════════════════════════════════════════════════════════════════════
    echo -e "${YELLOW}┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓${NC}"
    echo -e "${YELLOW}┃${NC} ${CYAN}System Resources${NC}                                                  ${YELLOW}┃${NC}"
    echo -e "${YELLOW}┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛${NC}"

    # Disk usage
    disk_info=$(df -h / | tail -1)
    disk_used=$(echo "$disk_info" | awk '{print $3}')
    disk_total=$(echo "$disk_info" | awk '{print $2}')
    disk_pct=$(echo "$disk_info" | awk '{print $5}' | tr -d '%')

    if [ $disk_pct -gt 90 ]; then
        disk_color="${RED}"
    elif [ $disk_pct -gt 70 ]; then
        disk_color="${YELLOW}"
    else
        disk_color="${GREEN}"
    fi

    echo -e "${BLUE}  Disk:${NC}   ${disk_color}${disk_used}/${disk_total} (${disk_pct}%)${NC}"

    # Memory usage
    mem_info=$(free -h | grep "Mem:")
    mem_used=$(echo "$mem_info" | awk '{print $3}')
    mem_total=$(echo "$mem_info" | awk '{print $2}')
    mem_pct=$(free | grep Mem | awk '{print int($3/$2 * 100)}')

    if [ $mem_pct -gt 90 ]; then
        mem_color="${RED}"
    elif [ $mem_pct -gt 70 ]; then
        mem_color="${YELLOW}"
    else
        mem_color="${GREEN}"
    fi

    echo -e "${BLUE}  Memory:${NC} ${mem_color}${mem_used}/${mem_total} (${mem_pct}%)${NC}"

    # ROS log size
    if [ -d ~/.ros/log ]; then
        ros_log_size=$(du -sh ~/.ros/log 2>/dev/null | awk '{print $1}')
        echo -e "${BLUE}  ROS Log:${NC} ${ros_log_size}"
    fi

    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Press Ctrl+C to exit | Refreshing every 2 seconds...${NC}"

    sleep 2
done
