#!/usr/bin/env python3
"""
GPU Performance Monitor with Anomaly Detection
Monitors GPU utilization and detects performance issues

Use Cases:
- Thermal throttling detection
- Process crash detection
- Resource contention identification
- Performance regression analysis

Usage:
    python gpu_monitor.py [--window SIZE] [--threshold PERCENT] [--interval SEC]
"""

import subprocess
import time
import argparse
from collections import deque
from datetime import datetime
from typing import Optional, Dict


class GPUMonitor:
    """Monitor GPU performance and detect anomalies"""
    
    def __init__(self, window_size: int = 10, drop_threshold: float = 20.0,
                 interval: float = 2.0, log_file: str = 'gpu_anomalies.log'):
        """
        Initialize GPU monitor
        
        Args:
            window_size: Number of samples for rolling average
            drop_threshold: Percent drop to consider anomalous
            interval: Seconds between samples
            log_file: Path to anomaly log file
        """
        self.window_size = window_size
        self.drop_threshold = drop_threshold
        self.interval = interval
        self.log_file = log_file
        
        self.util_history = deque(maxlen=window_size)
        self.temp_history = deque(maxlen=window_size)
        self.power_history = deque(maxlen=window_size)
    
    def get_gpu_metrics(self) -> Optional[Dict[str, float]]:
        """Query nvidia-smi for GPU metrics"""
        try:
            result = subprocess.run(
                ['nvidia-smi', 
                 '--query-gpu=utilization.gpu,temperature.gpu,power.draw',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                return {
                    'utilization': float(values[0]),
                    'temperature': float(values[1]),
                    'power': float(values[2])
                }
            else:
                print(f"[ERROR] nvidia-smi failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("[ERROR] nvidia-smi timeout")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to query GPU: {e}")
            return None
    
    def detect_anomalies(self, current: Dict[str, float]) -> None:
        """Detect and log anomalies"""
        
        if len(self.util_history) < self.window_size:
            return
        
        # Calculate averages
        avg_util = sum(self.util_history) / len(self.util_history)
        avg_temp = sum(self.temp_history) / len(self.temp_history)
        avg_power = sum(self.power_history) / len(self.power_history)
        
        timestamp = datetime.now().isoformat()
        anomalies = []
        
        # Check utilization drop
        util_drop = avg_util - current['utilization']
        if util_drop > self.drop_threshold:
            anomalies.append(
                f"Utilization drop: {avg_util:.1f}% → {current['utilization']:.1f}% "
                f"(Δ {util_drop:.1f}%)"
            )
        
        # Check thermal throttling
        if current['temperature'] > 85:
            anomalies.append(
                f"High temperature: {current['temperature']:.1f}°C (thermal throttling likely)"
            )
        
        # Check power anomalies
        power_drop = avg_power - current['power']
        if power_drop > 20:  # >20W drop
            anomalies.append(
                f"Power drop: {avg_power:.1f}W → {current['power']:.1f}W "
                f"(Δ {power_drop:.1f}W)"
            )
        
        # Log anomalies
        if anomalies:
            message = f"{timestamp} - ANOMALY DETECTED:\n"
            for anomaly in anomalies:
                message += f"  • {anomaly}\n"
            
            print(f"\n[ALERT] {message}")
            
            with open(self.log_file, 'a') as f:
                f.write(message + '\n')
    
    def run(self):
        """Main monitoring loop"""
        print("="*80)
        print("GPU PERFORMANCE MONITOR")
        print("="*80)
        print(f"Window size:      {self.window_size} samples")
        print(f"Drop threshold:   {self.drop_threshold}%")
        print(f"Sample interval:  {self.interval}s")
        print(f"Log file:         {self.log_file}")
        print("="*80 + "\n")
        
        try:
            while True:
                metrics = self.get_gpu_metrics()
                
                if metrics:
                    # Add to history
                    self.util_history.append(metrics['utilization'])
                    self.temp_history.append(metrics['temperature'])
                    self.power_history.append(metrics['power'])
                    
                    # Check for anomalies
                    self.detect_anomalies(metrics)
                    
                    # Display current status
                    if len(self.util_history) >= self.window_size:
                        avg_util = sum(self.util_history) / len(self.util_history)
                        status = (
                            f"[{datetime.now().strftime('%H:%M:%S')}] "
                            f"Util: {metrics['utilization']:5.1f}% "
                            f"(avg: {avg_util:5.1f}%) | "
                            f"Temp: {metrics['temperature']:5.1f}°C | "
                            f"Power: {metrics['power']:6.1f}W | "
                            f"Samples: {len(self.util_history)}"
                        )
                    else:
                        status = (
                            f"[{datetime.now().strftime('%H:%M:%S')}] "
                            f"Warming up... {len(self.util_history)}/{self.window_size} samples"
                        )
                    
                    print(status)
                
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print("\n\n[INFO] Monitoring stopped by user")
            print(f"[INFO] Anomaly log: {self.log_file}")


def main():
    parser = argparse.ArgumentParser(description='Monitor GPU performance')
    parser.add_argument('--window', type=int, default=10,
                       help='Rolling average window size (default: 10)')
    parser.add_argument('--threshold', type=float, default=20.0,
                       help='Utilization drop threshold %% (default: 20.0)')
    parser.add_argument('--interval', type=float, default=2.0,
                       help='Sample interval in seconds (default: 2.0)')
    parser.add_argument('--logfile', type=str, default='gpu_anomalies.log',
                       help='Anomaly log file (default: gpu_anomalies.log)')
    
    args = parser.parse_args()
    
    monitor = GPUMonitor(
        window_size=args.window,
        drop_threshold=args.threshold,
        interval=args.interval,
        log_file=args.logfile
    )
    
    monitor.run()


if __name__ == "__main__":
    main()