#!/usr/bin/env python3
"""
NVIDIA Automotive Perflab Benchmark Analyzer
Calculates P99 latency and comprehensive percentile statistics

Author: Hamna
Target: NVIDIA Performance Engineering - Edge AI Automotive
"""

import csv
import numpy as np
import statistics
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime


@dataclass
class BenchmarkSample:
    """Represents a single benchmark measurement"""
    timestamp: datetime
    workload: str
    batch_size: int
    latency_ms: float
    power_w: float
    gpu_util: float
    dla_util: float
    memory_bw: float
    temp_c: float


class PercentileCalculator:
    """Calculate percentile statistics for benchmark data"""
    
    def __init__(self):
        self.samples_by_workload: Dict[str, List[BenchmarkSample]] = defaultdict(list)
        self.all_samples: List[BenchmarkSample] = []
    
    def load_data(self, csv_file: str) -> None:
        """Load benchmark data from CSV file"""
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            
            for line in reader:
                # Skip comments and headers
                if not line or line[0].startswith('#'):
                    continue
                
                try:
                    sample = BenchmarkSample(
                        timestamp=datetime.strptime(line[0], '%Y-%m-%d %H:%M:%S.%f'),
                        workload=line[1],
                        batch_size=int(line[2]),
                        latency_ms=float(line[3]),
                        power_w=float(line[4]),
                        gpu_util=float(line[5]),
                        dla_util=float(line[6]),
                        memory_bw=float(line[7]),
                        temp_c=float(line[8])
                    )
                    
                    self.samples_by_workload[sample.workload].append(sample)
                    self.all_samples.append(sample)
                    
                except (ValueError, IndexError) as e:
                    # Skip malformed lines
                    continue
    
    def calculate_percentile(self, data: List[float], percentile: float) -> float:
        """
        Calculate percentile using numpy's interpolation method
        percentile: value between 0-100
        """
        return np.percentile(data, percentile)
    
    def calculate_percentiles(self, latencies: List[float]) -> Dict[str, float]:
        """Calculate common percentiles for latency data"""
        if not latencies:
            return {}
        
        percentiles = {
            'min': min(latencies),
            'p50': self.calculate_percentile(latencies, 50),  # median
            'p90': self.calculate_percentile(latencies, 90),
            'p95': self.calculate_percentile(latencies, 95),
            'p99': self.calculate_percentile(latencies, 99),
            'p99.9': self.calculate_percentile(latencies, 99.9),
            'max': max(latencies),
            'mean': statistics.mean(latencies),
            'stddev': statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        }
        
        return percentiles
    
    def analyze_workload(self, workload: str) -> Dict[str, float]:
        """Analyze percentiles for a specific workload"""
        samples = self.samples_by_workload[workload]
        latencies = [s.latency_ms for s in samples]
        return self.calculate_percentiles(latencies)
    
    def analyze_all_workloads(self) -> Dict[str, Dict[str, float]]:
        """Analyze percentiles for all workloads"""
        results = {}
        for workload in self.samples_by_workload.keys():
            results[workload] = self.analyze_workload(workload)
        return results
    
    def get_power_metrics(self, workload: str) -> Dict[str, float]:
        """Calculate power consumption statistics"""
        samples = self.samples_by_workload[workload]
        power_values = [s.power_w for s in samples]
        
        return {
            'avg_power': statistics.mean(power_values),
            'max_power': max(power_values),
            'min_power': min(power_values),
        }
    
    def get_thermal_metrics(self, workload: str) -> Dict[str, float]:
        """Calculate thermal statistics"""
        samples = self.samples_by_workload[workload]
        temps = [s.temp_c for s in samples]
        
        return {
            'avg_temp': statistics.mean(temps),
            'max_temp': max(temps),
            'min_temp': min(temps),
        }
    
    def calculate_throughput(self, workload: str) -> float:
        """Calculate throughput in FPS (frames per second)"""
        samples = self.samples_by_workload[workload]
        if not samples:
            return 0.0
        
        latencies = [s.latency_ms for s in samples]
        avg_latency_ms = statistics.mean(latencies)
        
        # Throughput = 1000ms / avg_latency_ms
        return 1000.0 / avg_latency_ms
    
    def get_sla_compliance(self, workload: str, sla_ms: float) -> Dict[str, float]:
        """
        Calculate SLA compliance metrics
        sla_ms: Service Level Agreement latency threshold in milliseconds
        """
        samples = self.samples_by_workload[workload]
        latencies = [s.latency_ms for s in samples]
        
        violations = sum(1 for lat in latencies if lat > sla_ms)
        total = len(latencies)
        
        return {
            'total_samples': total,
            'violations': violations,
            'compliance_rate': ((total - violations) / total) * 100 if total > 0 else 0.0,
            'violation_rate': (violations / total) * 100 if total > 0 else 0.0,
        }


def print_separator(char='=', length=90):
    """Print a separator line"""
    print(char * length)


def print_workload_analysis(workload: str, stats: Dict[str, float], 
                           power: Dict[str, float], thermal: Dict[str, float],
                           throughput: float, sample_count: int):
    """Pretty print analysis for a single workload"""
    print(f"\n{workload.upper()}")
    print_separator('-')
    print(f"Samples: {sample_count}")
    print(f"\nLatency Statistics (ms):")
    print(f"  Min:          {stats['min']:8.2f}")
    print(f"  Mean:         {stats['mean']:8.2f}")
    print(f"  Median (P50): {stats['p50']:8.2f}")
    print(f"  P90:          {stats['p90']:8.2f}")
    print(f"  P95:          {stats['p95']:8.2f}")
    print(f"  P99:          {stats['p99']:8.2f}  ← 99th Percentile")
    print(f"  P99.9:        {stats['p99.9']:8.2f}")
    print(f"  Max:          {stats['max']:8.2f}")
    print(f"  StdDev:       {stats['stddev']:8.2f}")
    
    print(f"\nThroughput:")
    print(f"  {throughput:.1f} FPS (frames per second)")
    
    print(f"\nPower Consumption (W):")
    print(f"  Average: {power['avg_power']:.1f}")
    print(f"  Range:   {power['min_power']:.1f} - {power['max_power']:.1f}")
    
    print(f"\nThermal (°C):")
    print(f"  Average: {thermal['avg_temp']:.1f}")
    print(f"  Range:   {thermal['min_temp']:.1f} - {thermal['max_temp']:.1f}")


def print_sla_analysis(workload: str, sla_metrics: Dict[str, float], sla_threshold: float):
    """Print SLA compliance analysis"""
    print(f"\nSLA Compliance (Threshold: {sla_threshold}ms):")
    print(f"  Total Samples:    {sla_metrics['total_samples']}")
    print(f"  Violations:       {sla_metrics['violations']}")
    print(f"  Compliance Rate:  {sla_metrics['compliance_rate']:.2f}%")
    print(f"  Violation Rate:   {sla_metrics['violation_rate']:.2f}%")
    
    if sla_metrics['compliance_rate'] >= 99.0:
        status = "✓ PASS"
    elif sla_metrics['compliance_rate'] >= 95.0:
        status = "⚠ WARNING"
    else:
        status = "✗ FAIL"
    
    print(f"  Status:           {status}")


def generate_summary_table(analyzer: PercentileCalculator):
    """Generate a summary table comparing all workloads"""
    print("\n")
    print_separator()
    print("WORKLOAD COMPARISON - P99 LATENCY SUMMARY")
    print_separator()
    
    # Header
    print(f"{'Workload':<30} {'Samples':>8} {'Mean':>8} {'P99':>8} {'Max':>8} {'FPS':>8}")
    print_separator('-')
    
    # Sort workloads by P99 latency
    workload_stats = []
    for workload in analyzer.samples_by_workload.keys():
        stats = analyzer.analyze_workload(workload)
        throughput = analyzer.calculate_throughput(workload)
        sample_count = len(analyzer.samples_by_workload[workload])
        
        workload_stats.append({
            'name': workload,
            'samples': sample_count,
            'mean': stats['mean'],
            'p99': stats['p99'],
            'max': stats['max'],
            'fps': throughput,
        })
    
    # Sort by P99 latency
    workload_stats.sort(key=lambda x: x['p99'])
    
    for ws in workload_stats:
        print(f"{ws['name']:<30} {ws['samples']:>8} {ws['mean']:>8.2f} "
              f"{ws['p99']:>8.2f} {ws['max']:>8.2f} {ws['fps']:>8.1f}")
    
    print_separator()


def main():
    """Main execution function"""
    
    print("NVIDIA AUTOMOTIVE PERFLAB BENCHMARK ANALYZER")
    print_separator()
    print("Platform: NVIDIA Drive Orin AGX")
    print("Analysis: P99 Latency & Performance Metrics\n")
    
    # Initialize analyzer
    analyzer = PercentileCalculator()
    
    # Load benchmark data
    csv_file = '/home/claude/automotive_benchmark_data.csv'
    print(f"Loading data from: {csv_file}")
    analyzer.load_data(csv_file)
    
    total_samples = len(analyzer.all_samples)
    workload_count = len(analyzer.samples_by_workload)
    print(f"Loaded {total_samples} samples across {workload_count} workloads\n")
    
    print_separator()
    
    # Analyze each workload
    for workload in sorted(analyzer.samples_by_workload.keys()):
        stats = analyzer.analyze_workload(workload)
        power = analyzer.get_power_metrics(workload)
        thermal = analyzer.get_thermal_metrics(workload)
        throughput = analyzer.calculate_throughput(workload)
        sample_count = len(analyzer.samples_by_workload[workload])
        
        print_workload_analysis(workload, stats, power, thermal, throughput, sample_count)
        
        # SLA analysis (example thresholds)
        sla_thresholds = {
            'resnet50_detection': 20.0,
            'yolox_l_tracking': 35.0,
            'segformer_road_seg': 25.0,
            'efficientdet_lane': 20.0,
            'transformer_fusion': 55.0,
            'mobilenetv3_ped': 10.0,
            'bevformer_3d': 100.0,
            'pointpillars_lidar': 40.0,
            'multitask_network': 75.0,
            'planning_trajectory': 30.0,
        }
        
        if workload in sla_thresholds:
            sla_metrics = analyzer.get_sla_compliance(workload, sla_thresholds[workload])
            print_sla_analysis(workload, sla_metrics, sla_thresholds[workload])
        
        print_separator()
    
    # Generate summary comparison table
    generate_summary_table(analyzer)
    
    print("\n📊 Analysis complete!")
    print("\nKey Performance Indicators:")
    print("  • P99 latency: 99% of requests complete within this time")
    print("  • Lower P99 values indicate more consistent performance")
    print("  • P99 is critical for real-time automotive safety applications")
    print("\n")


if __name__ == "__main__":
    main()