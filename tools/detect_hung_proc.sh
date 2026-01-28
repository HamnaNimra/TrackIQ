#!/bin/bash
# this script detects and kills hung processes based on runtime duration
# Usage: ./detech_hung_proc.sh
# 1. Finds all processes matching a pattern (e.g., "trtexec")
# 2. Checks how long each has been running
# 3. Kills any running longer than threshold (e.g., 30 minutes)
# 4. Logs what was killed
# 5. Sends the parent SIGTERM first, then SIGKILL if needed

# Bonus: Handle zombie processes correctly

PROCESS_PATTERN="trtexec"
MAX_RUNTIME_MINUTES=30
LOG_FILE="/var/log/process_monitor.log"

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

get_process_runtime_minutes() {
    local pid=$1
    local etime

    etime=$(ps -p "$pid" -o etimes= 2>/dev/null | tr -d ' ')
    [ -z "$etime" ] && echo 0 && return

    echo $((etime / 60))
}

kill_process_gracefully() {
    local pid=$1
    local process_name=$2

    log_message "Attempting to kill PID $pid ($process_name)..."
    kill -TERM "$pid" 2>/dev/null

    for i in {1..10}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            log_message "PID $pid terminated gracefully"
            return 0
        fi
        sleep 1
    done

    log_message "PID $pid still running, sending SIGKILL"
    kill -KILL "$pid" 2>/dev/null
    sleep 1

    if ! kill -0 "$pid" 2>/dev/null; then
        log_message "PID $pid killed forcefully"
    else
        log_message "ERROR: Failed to kill PID $pid"
    fi
}

check_for_zombies() {
    local zombies
    zombies=$(ps -eo pid,ppid,stat,comm | awk '$3 ~ /^Z/')

    if [ -n "$zombies" ]; then
        log_message "WARNING: Zombie processes detected:"
        while read pid ppid stat cmd; do
            log_message "  Zombie PID $pid (parent: $ppid, cmd: $cmd)"
        done <<< "$zombies"
    fi
}

main() {
    log_message "Starting process monitor for pattern: $PROCESS_PATTERN"

    ps -eo pid,comm | grep "$PROCESS_PATTERN" | grep -v grep | while read pid cmd; do
        runtime_minutes=$(get_process_runtime_minutes "$pid")
        log_message "Found PID $pid ($cmd), runtime: ${runtime_minutes} min"

        if [ "$runtime_minutes" -gt "$MAX_RUNTIME_MINUTES" ]; then
            log_message "PID $pid exceeded ${MAX_RUNTIME_MINUTES} min"
            kill_process_gracefully "$pid" "$cmd"
        fi
    done

    check_for_zombies
    log_message "Process monitor check complete"
}

main