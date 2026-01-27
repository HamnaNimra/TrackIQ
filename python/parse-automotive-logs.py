#!/usr/bin/env python3
“””
Automotive Performance Log Parser
Extracts timestamps where latency exceeds specified threshold

Author: Hamna
Target: NVIDIA Edge AI / Automotive Performance Engineering
“””

import re
import sys
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class LatencyEvent:
“”“Represents a single latency measurement event”””
timestamp: datetime
frame_id: int
inference_latency: float
e2e_latency: float

```
def __str__(self):
    return (f"{self.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} | "
            f"Frame {self.frame_id:04d} | "
            f"Inference: {self.inference_latency:6.1f}ms | "
            f"E2E: {self.e2e_latency:6.1f}ms")
```

class PerformanceLogParser:
“”“Parser for automotive performance engineering logs”””

```
# Regex pattern to match PERF log lines with latency data
PERF_PATTERN = re.compile(
    r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})'  # timestamp
    r'.*?Frame (\d+):'                                 # frame number
    r'.*?Inference latency: ([\d.]+)ms'               # inference latency
    r'.*?E2E latency: ([\d.]+)ms'                     # end-to-end latency
)

def __init__(self, log_file: str):
    """Initialize parser with log file path"""
    self.log_file = log_file
    self.events: List[LatencyEvent] = []
    
def parse(self) -> List[LatencyEvent]:
    """Parse the log file and extract all latency events"""
    with open(self.log_file, 'r') as f:
        for line in f:
            match = self.PERF_PATTERN.search(line)
            if match:
                timestamp_str, frame_id, inference_lat, e2e_lat = match.groups()
                
                event = LatencyEvent(
                    timestamp=datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f'),
                    frame_id=int(frame_id),
                    inference_latency=float(inference_lat),
                    e2e_latency=float(e2e_lat)
                )
                self.events.append(event)
    
    return self.events

def filter_by_inference_latency(self, threshold_ms: float) -> List[LatencyEvent]:
    """Return events where inference latency exceeds threshold"""
    return [e for e in self.events if e.inference_latency > threshold_ms]

def filter_by_e2e_latency(self, threshold_ms: float) -> List[LatencyEvent]:
    """Return events where end-to-end latency exceeds threshold"""
    return [e for e in self.events if e.e2e_latency > threshold_ms]

def get_statistics(self) -> Dict[str, float]:
    """Calculate latency statistics"""
    if not self.events:
        return {}
    
    inference_latencies = [e.inference_latency for e in self.events]
    e2e_latencies = [e.e2e_latency for e in self.events]
    
    return {
        'total_frames': len(self.events),
        'inference_avg': sum(inference_latencies) / len(inference_latencies),
        'inference_min': min(inference_latencies),
        'inference_max': max(inference_latencies),
        'e2e_avg': sum(e2e_latencies) / len(e2e_latencies),
        'e2e_min': min(e2e_latencies),
        'e2e_max': max(e2e_latencies),
    }
```

def print_statistics(stats: Dict[str, float]) -> None:
“”“Pretty print statistics”””
print(”\n” + “=”*70)
print(“LATENCY STATISTICS”)
print(”=”*70)
print(f”Total frames analyzed:    {stats[‘total_frames’]}”)
print(f”\nInference Latency:”)
print(f”  Average:  {stats[‘inference_avg’]:6.1f} ms”)
print(f”  Min:      {stats[‘inference_min’]:6.1f} ms”)
print(f”  Max:      {stats[‘inference_max’]:6.1f} ms”)
print(f”\nEnd-to-End Latency:”)
print(f”  Average:  {stats[‘e2e_avg’]:6.1f} ms”)
print(f”  Min:      {stats[‘e2e_min’]:6.1f} ms”)
print(f”  Max:      {stats[‘e2e_max’]:6.1f} ms”)
print(”=”*70 + “\n”)

def main():
“”“Main execution function”””
# Configuration
LOG_FILE = ‘/home/claude/automotive_perf.log’
INFERENCE_THRESHOLD = 100.0  # ms
E2E_THRESHOLD = 100.0        # ms

```
print("NVIDIA Automotive Performance Log Analyzer")
print("="*70)
print(f"Log file: {LOG_FILE}")
print(f"Latency threshold: {INFERENCE_THRESHOLD} ms\n")

# Parse the log file
parser = PerformanceLogParser(LOG_FILE)
parser.parse()

# Get statistics
stats = parser.get_statistics()
if stats:
    print_statistics(stats)

# Filter high-latency events (inference)
high_inference_events = parser.filter_by_inference_latency(INFERENCE_THRESHOLD)

print(f"FRAMES WITH INFERENCE LATENCY > {INFERENCE_THRESHOLD}ms")
print("="*70)
if high_inference_events:
    print(f"Found {len(high_inference_events)} frames exceeding threshold:\n")
    for event in high_inference_events:
        print(f"  {event}")
else:
    print(f"No frames found with inference latency > {INFERENCE_THRESHOLD}ms")

print("\n" + "="*70)

# Filter high-latency events (E2E)
high_e2e_events = parser.filter_by_e2e_latency(E2E_THRESHOLD)

print(f"\nFRAMES WITH E2E LATENCY > {E2E_THRESHOLD}ms")
print("="*70)
if high_e2e_events:
    print(f"Found {len(high_e2e_events)} frames exceeding threshold:\n")
    for event in high_e2e_events:
        print(f"  {event}")
else:
    print(f"No frames found with E2E latency > {E2E_THRESHOLD}ms")

print("\n" + "="*70)

# Calculate violation rate
if stats:
    inference_violation_rate = (len(high_inference_events) / stats['total_frames']) * 100
    e2e_violation_rate = (len(high_e2e_events) / stats['total_frames']) * 100
    
    print(f"\nVIOLATION RATES:")
    print(f"  Inference latency > {INFERENCE_THRESHOLD}ms: {inference_violation_rate:.1f}%")
    print(f"  E2E latency > {E2E_THRESHOLD}ms:       {e2e_violation_rate:.1f}%")
    print("="*70 + "\n")
```

if **name** == “**main**”:
main()