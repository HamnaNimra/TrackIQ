#!/usr/bin/env python3
"""
Batching Trade-off Analysis
Demonstrates the relationship between batch size, throughput, and latency

Concept:
- Larger batches increase throughput (images/sec) by amortizing overhead
- Larger batches increase per-image latency (each image waits for batch)
- This is a fundamental trade-off in inference optimization
Usage:
    python batching_tradeoff.py [--images NUM] [--batch-sizes LIST]

Disclaimer: This code is for educational purposes only.
It demonstrates batching trade-offs in model inference in kernel launches and processing.
Author: Hamna
"""

import time
import argparse
from typing import List, Tuple


def simulate_inference(batch_size: int, base_overhead: float = 0.01, 
                      time_per_image: float = 0.005) -> float:
    """
    Simulate model inference with batching
    
    Args:
        batch_size: Number of images in batch
        base_overhead: Fixed kernel launch overhead (seconds)
        time_per_image: Processing time per image (seconds)
    
    Returns:
        Total batch processing time (seconds)
    """
    total_time = base_overhead + (batch_size * time_per_image)
    time.sleep(total_time)
    return total_time


def benchmark_batching(total_images: int, batch_size: int) -> Tuple[float, float]:
    """
    Benchmark inference with given batch size
    
    Args:
        total_images: Total number of images to process
        batch_size: Images per batch
    
    Returns:
        (throughput in images/sec, average latency per image in seconds)
    """
    start = time.time()
    
    num_batches = total_images // batch_size
    total_batch_time = 0.0
    
    for _ in range(num_batches):
        batch_time = simulate_inference(batch_size)
        total_batch_time += batch_time
    
    end = time.time()
    wall_time = end - start
    
    # Calculate metrics
    throughput = total_images / wall_time  # images/sec
    avg_latency_per_image = total_batch_time / total_images  # sec/image
    
    return throughput, avg_latency_per_image


def analyze_tradeoff(total_images: int, batch_sizes: List[int]):
    """Analyze and display batching trade-offs"""
    
    print("="*80)
    print("BATCHING TRADE-OFF ANALYSIS")
    print("="*80)
    print(f"Total images to process: {total_images}")
    print(f"Base overhead: 10ms (kernel launch)")
    print(f"Per-image processing: 5ms\n")
    
    print(f"{'Batch Size':<12} | {'Throughput':<18} | {'Avg Latency':<18} | {'Batch Time':<12}")
    print(f"{'':12} | {'(images/sec)':<18} | {'(ms/image)':<18} | {'(ms)':<12}")
    print("-"*80)
    
    results = []
    
    for bs in batch_sizes:
        throughput, latency = benchmark_batching(total_images, bs)
        batch_time = (0.01 + bs * 0.005) * 1000  # Convert to ms
        
        results.append({
            'batch_size': bs,
            'throughput': throughput,
            'latency': latency * 1000,  # Convert to ms
            'batch_time': batch_time
        })
        
        print(f"{bs:<12} | {throughput:>18.2f} | {latency*1000:>18.2f} | {batch_time:>12.2f}")
    
    print("="*80)
    
    # Analysis
    print("\nKEY INSIGHTS:")
    print("-"*80)
    
    min_latency = min(results, key=lambda x: x['latency'])
    max_throughput = max(results, key=lambda x: x['throughput'])
    
    print(f"• Lowest latency:     batch_size={min_latency['batch_size']} "
          f"({min_latency['latency']:.2f}ms per image)")
    print(f"• Highest throughput: batch_size={max_throughput['batch_size']} "
          f"({max_throughput['throughput']:.2f} images/sec)")
    
    print("\nTRADE-OFF EXPLANATION:")
    print("-"*80)
    print("As batch size increases:")
    print("  ✓ Throughput INCREASES - Fixed overhead amortized over more images")
    print("  ✗ Latency INCREASES - Each image waits for entire batch to process")
    print("\nChoice depends on use case:")
    print("  • Real-time (autonomous driving): Prefer low latency → small batches")
    print("  • Batch processing (data center): Prefer high throughput → large batches")
    print("="*80 + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Demonstrate batching trade-offs')
    parser.add_argument('--images', type=int, default=100,
                       help='Total images to process (default: 100)')
    parser.add_argument('--batch-sizes', type=int, nargs='+', 
                       default=[1, 4, 8, 16, 32],
                       help='Batch sizes to test (default: 1 4 8 16 32)')
    
    args = parser.parse_args()
    
    analyze_tradeoff(args.images, args.batch_sizes)


if __name__ == "__main__":
    main()