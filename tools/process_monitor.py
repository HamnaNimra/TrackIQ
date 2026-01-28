#!/usr/bin/env python3
"""
Process Monitor with Zombie Prevention
Demonstrates proper subprocess management to avoid zombie processes

Key Concepts:
- Zombies occur when parent doesn't call wait() on terminated child
- Must call wait()/communicate() to collect exit status
- Even when killing a process, must still wait() to clean up

Usage:
    python process_monitor.py <command> [--timeout SECONDS]
"""

import subprocess
import time
import sys
import argparse
from datetime import datetime


def check_for_zombies():
    """Check if there are any zombie processes"""
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=5)

        zombie_count = 0
        for line in result.stdout.split("\n"):
            if " Z " in line or " Z+ " in line:
                print(f"[ZOMBIE DETECTED] {line}")
                zombie_count += 1

        return zombie_count
    except Exception as e:
        print(f"Error checking for zombies: {e}")
        return 0


def run_with_timeout(command: str, timeout_seconds: int, check_interval: float = 0.5):
    """
    Run command with timeout and proper cleanup

    Args:
        command: Shell command to execute
        timeout_seconds: Maximum execution time
        check_interval: How often to check if process is still running

    Returns:
        (exit_code, stdout, stderr)
    """
    print(f"[{datetime.now()}] Starting: {command}")
    print(f"[INFO] Timeout: {timeout_seconds}s")

    try:
        # Start the process
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        print(f"[INFO] Process started with PID: {process.pid}")

        # Monitor with timeout
        start_time = time.time()

        try:
            # Wait with timeout
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            elapsed = time.time() - start_time
            exit_code = process.returncode

            print(f"[{datetime.now()}] Process completed normally")
            print(f"[INFO] Exit code: {exit_code}")
            print(f"[INFO] Elapsed time: {elapsed:.2f}s")

            return exit_code, stdout, stderr

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            print(f"[{datetime.now()}] Process exceeded timeout!")
            print(f"[INFO] Elapsed: {elapsed:.2f}s")
            print(f"[ACTION] Terminating process {process.pid}...")

            # Try graceful termination first
            process.terminate()

            try:
                # Give it 5 seconds to terminate gracefully
                stdout, stderr = process.communicate(timeout=5)
                print("[INFO] Process terminated gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate
                print(f"[ACTION] Force killing process {process.pid}...")
                process.kill()

                # CRITICAL: Must still call communicate() to collect zombie
                stdout, stderr = process.communicate()
                print(f"[INFO] Process killed (exit code: {process.returncode})")

            return -1, stdout, stderr

    except Exception as e:
        print(f"[ERROR] Exception while running process: {e}")
        return -1, None, None


def demonstrate_zombie_prevention():
    """Demonstrate what happens with and without proper wait()"""

    print("=" * 80)
    print("DEMONSTRATION: Proper Subprocess Management")
    print("=" * 80)

    # Check initial zombie count
    initial_zombies = check_for_zombies()
    print(f"\n[INITIAL] Zombie processes: {initial_zombies}\n")

    # Test 1: Normal completion
    print("\nTEST 1: Normal process completion")
    print("-" * 80)
    exit_code, stdout, stderr = run_with_timeout("echo 'Hello World' && sleep 1", timeout_seconds=5)
    print(f"[RESULT] Exit code: {exit_code}")
    if stdout:
        print(f"[STDOUT] {stdout.strip()}")

    # Test 2: Timeout and kill
    print("\n\nTEST 2: Process timeout and kill")
    print("-" * 80)
    exit_code, stdout, stderr = run_with_timeout("sleep 100", timeout_seconds=2)
    print(f"[RESULT] Exit code: {exit_code} (negative indicates timeout)")

    # Check final zombie count
    print("\n" + "=" * 80)
    final_zombies = check_for_zombies()
    print(f"[FINAL] Zombie processes: {final_zombies}")

    if final_zombies == initial_zombies:
        print("[SUCCESS] No zombies created - proper cleanup performed!")
    else:
        print(f"[WARNING] Created {final_zombies - initial_zombies} zombie(s)")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run command with timeout and zombie prevention")
    parser.add_argument("command", nargs="?", help="Command to run")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds (default: 30)")
    parser.add_argument("--demo", action="store_true", help="Run demonstration mode")

    args = parser.parse_args()

    if args.demo or not args.command:
        demonstrate_zombie_prevention()
    else:
        exit_code, stdout, stderr = run_with_timeout(args.command, args.timeout)

        if stdout:
            print("\n[STDOUT]")
            print(stdout)
        if stderr:
            print("\n[STDERR]")
            print(stderr)

        sys.exit(0 if exit_code == 0 else 1)


if __name__ == "__main__":
    main()
