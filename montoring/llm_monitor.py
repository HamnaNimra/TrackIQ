"""
LLM Inference Memory Monitor     
--------------------------------
1. Monitors GPU memory during LLM inference
2. Tracks memory growth per token generated
3. Calculates KV cache size based on model config
4. Detects when we're approaching OOM
5. Suggests optimal batch size or sequence length

Given model config:
- num_layers = 32
- num_heads = 32  
- head_size = 128
- max_sequence_length = 2048
- batch_size = variable
- precision = fp16

Target: NVIDIA LLM Optimization Toolkit 
Author: Hamna

Usage: 
    python llm_monitor.py [--duration SEC] [--interval SEC]
"""

import subprocess
import time
import math

class KVCacheAnalyzer:
    def __init__(self, num_layers, num_heads, head_size, max_seq_len, precision="fp16"):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = head_size
        self.max_seq_len = max_seq_len
        self.precision = precision
        
        # Bytes per element
        self.bytes_per_element = 2 if precision == "fp16" else 4  # fp16 or fp32
        
    def calculate_kv_cache_size_per_token(self, batch_size=1):
        """
        Calculate KV cache memory for one token
        
        KV cache stores keys and values for each layer
        Shape: [batch_size, num_heads, seq_len, head_size]
        We have 2 tensors (K and V) per layer
        """
        # Size for one layer, one token
        size_per_layer = (
            batch_size * 
            self.num_heads * 
            self.head_size * 
            self.bytes_per_element * 
            2  # K and V
        )
        
        # Multiply by number of layers
        total_per_token = size_per_layer * self.num_layers
        
        return total_per_token
    
    def calculate_total_kv_cache(self, batch_size, sequence_length):
        """Calculate total KV cache memory for full sequence"""
        per_token = self.calculate_kv_cache_size_per_token(batch_size)
        total = per_token * sequence_length
        
        return total
    
    def get_gpu_memory_usage(self):
        """Query current GPU memory usage"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', 
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True
            )
            used, total = result.stdout.strip().split(',')
            return int(used), int(total)
        except:
            return None, None
    
    def bytes_to_readable(self, bytes_val):
        """Convert bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_val < 1024:
                return f"{bytes_val:.2f} {unit}"
            bytes_val /= 1024
        return f"{bytes_val:.2f} TB"
    
    def recommend_batch_size(self, available_memory_mb, target_seq_len):
        """
        Given available memory, recommend safe batch size
        Leave 20% buffer for other allocations
        """
        available_bytes = available_memory_mb * 1024 * 1024 * 0.8  # 80% usage
        
        # Calculate KV cache for batch_size=1
        kv_per_batch = self.calculate_total_kv_cache(1, target_seq_len)
        
        # How many batches can we fit?
        max_batch = int(available_bytes / kv_per_batch)
        
        return max(1, max_batch)
    
    def monitor_inference(self, duration_seconds=30, sample_interval=1):
        """
        Monitor GPU memory during inference
        Track growth rate
        """
        print(f"Monitoring GPU memory for {duration_seconds} seconds...")
        print(f"{'Time':>6} | {'Used (MB)':>10} | {'Total (MB)':>10} | {'Growth (MB/s)':>15}")
        print("-" * 60)
        
        start_time = time.time()
        measurements = []
        
        while time.time() - start_time < duration_seconds:
            used, total = self.get_gpu_memory_usage()
            if used is not None:
                elapsed = time.time() - start_time
                measurements.append((elapsed, used))
                
                # Calculate growth rate
                if len(measurements) > 1:
                    time_delta = measurements[-1][0] - measurements[-2][0]
                    mem_delta = measurements[-1][1] - measurements[-2][1]
                    growth_rate = mem_delta / time_delta if time_delta > 0 else 0
                else:
                    growth_rate = 0
                
                print(f"{elapsed:6.1f} | {used:10} | {total:10} | {growth_rate:15.2f}")
            
            time.sleep(sample_interval)
    
        # Analysis
        if len(measurements) >= 2:
            initial_mem = measurements[0][1]
            final_mem = measurements[-1][1]
            total_growth = final_mem - initial_mem
            avg_growth_rate = total_growth / duration_seconds
            
            print(f"\n=== Analysis ===")
            print(f"Total memory growth: {total_growth} MB")
            print(f"Average growth rate: {avg_growth_rate:.2f} MB/s")
            
            # Predict OOM
            if avg_growth_rate > 0:
                _, total_mem = self.get_gpu_memory_usage()
                available = total_mem - final_mem
                time_to_oom = available / avg_growth_rate if avg_growth_rate > 0 else float('inf')
                print(f"Estimated time to OOM: {time_to_oom:.1f} seconds")
                
                if time_to_oom < 60:
                    print("⚠️  WARNING: OOM imminent! Reduce batch size or sequence length")

def main():
    # Example: Llama-2 style model
    analyzer = KVCacheAnalyzer(
        num_layers=32,
        num_heads=32,
        head_size=128,
        max_seq_len=2048,
        precision="fp16"
    )
    
    print("=== KV Cache Memory Calculator ===\n")
    
    # Calculate for different scenarios
    batch_size = 4
    seq_length = 512
    
    kv_cache_size = analyzer.calculate_total_kv_cache(batch_size, seq_length)
    print(f"KV cache size for batch={batch_size}, seq_len={seq_length}:")
    print(f"  {analyzer.bytes_to_readable(kv_cache_size)}\n")
    
    # Show scaling
    print("KV cache scaling:")
    print(f"{'Batch':>6} | {'Seq Len':>8} | {'KV Cache Size':>15}")
    print("-" * 35)
    for bs in [1, 2, 4, 8]:
        for sl in [256, 512, 1024, 2048]:
            size = analyzer.calculate_total_kv_cache(bs, sl)
            print(f"{bs:>6} | {sl:>8} | {analyzer.bytes_to_readable(size):>15}")
    
    # Recommend batch size
    print("\n=== Batch Size Recommendations ===")
    used, total = analyzer.get_gpu_memory_usage()
    if used is not None:
        available = total - used
        print(f"Available GPU memory: {available} MB")
        
        for seq_len in [512, 1024, 2048]:
            rec_batch = analyzer.recommend_batch_size(available, seq_len)
            print(f"Seq length {seq_len}: Recommended batch size = {rec_batch}")
    
    # Optional: Live monitoring
    # analyzer.monitor_inference(duration_seconds=30)

if __name__ == "__main__":
    main()
