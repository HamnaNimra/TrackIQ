"""
LLM Memory Calculator
Calculates memory requirements for LLMs based on model size, batch size, context length,
 and precision.        
1. Calculates memory for model weights
2. Calculates KV cache memory
3. Estimates activation memory
4. Provides total memory requirement
5. Suggests max batch size or context length given available memory

Why it matters:
Helps optimize deployment of large language models on hardware with limited memory. 
allows users to understand trade-offs between batch size, context length, and precision (fp16 vs fp32).
Enables better planning for real-time applications where memory constraints are critical.
Allows users to avoid out-of-memory errors by estimating requirements beforehand.

what-if analysis:  
    how large a model can fit in 80GB GPU memory at batch size 1 and 128k context?
    and what is the max context length for 70B model at batch size 1?

    answer:    
    it will not fit, need ~95GB
    max context length for batch=1 is ~94k tokens
    allows users to make informed decisions on model deployment configurations.

usage:
    python llm_memory_calculator.py

Author: Hamna
Target: NVIDIA LLM Optimization Toolkit



"""


class LLMMemoryCalculator:
    def __init__(self, num_params_billions, num_layers, num_heads, head_size, precision="fp16"):
        self.num_params = num_params_billions * 1e9
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = head_size
        self.precision = precision
        
        self.bytes_per_param = 2 if precision == "fp16" else 4
        
    def calculate_model_weights(self):
        """Memory for model parameters"""
        return self.num_params * self.bytes_per_param
    
    def calculate_kv_cache(self, batch_size, context_length):
        """Memory for KV cache"""
        # K and V for each layer
        kv_size_per_layer = (
            2 *  # K and V
            batch_size *
            self.num_heads *
            context_length *
            self.head_size *
            self.bytes_per_param
        )
        return kv_size_per_layer * self.num_layers
    
    def calculate_activations(self, batch_size, context_length):
        """Rough estimate of activation memory"""
        # Very rough: ~4x hidden size per token
        hidden_size = self.num_heads * self.head_size
        activation_per_token = hidden_size * 4 * self.bytes_per_param
        return activation_per_token * batch_size * context_length * self.num_layers * 0.1  # rough estimate
    
    def calculate_total(self, batch_size, context_length):
        """Total memory requirement"""
        weights = self.calculate_model_weights()
        kv = self.calculate_kv_cache(batch_size, context_length)
        activations = self.calculate_activations(batch_size, context_length)
        overhead = (weights + kv + activations) * 0.1  # 10% overhead
        
        total = weights + kv + activations + overhead
        
        return {
            'weights_gb': weights / 1e9,
            'kv_cache_gb': kv / 1e9,
            'activations_gb': activations / 1e9,
            'overhead_gb': overhead / 1e9,
            'total_gb': total / 1e9
        }
    
    def find_max_batch_or_context(self, available_memory_gb, fixed_param, fixed_value):
        """
        Given available memory, find max batch or context
        fixed_param: 'batch_size' or 'context_length'
        """
        available_bytes = available_memory_gb * 1e9
        
        # Binary search for max value
        low, high = 1, 10000
        result = 1
        
        while low <= high:
            mid = (low + high) // 2
            
            if fixed_param == 'batch_size':
                mem = self.calculate_total(fixed_value, mid)
            else:  # context_length
                mem = self.calculate_total(mid, fixed_value)
            
            if mem['total_gb'] * 1e9 <= available_bytes:
                result = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return result

# Example usage
calculator = LLMMemoryCalculator(
    num_params_billions=70,
    num_layers=80,
    num_heads=64,
    head_size=128,
    precision="fp16"
)

print("=== 70B Model Memory Requirements ===\n")

# Check if it fits
mem = calculator.calculate_total(batch_size=1, context_length=128000)
print(f"Batch size: 1, Context length: 128k tokens")
print(f"  Model weights:  {mem['weights_gb']:.2f} GB")
print(f"  KV cache:       {mem['kv_cache_gb']:.2f} GB")
print(f"  Activations:    {mem['activations_gb']:.2f} GB")
print(f"  Overhead:       {mem['overhead_gb']:.2f} GB")
print(f"  TOTAL:          {mem['total_gb']:.2f} GB")

available = 80
print(f"\nAvailable GPU memory: {available} GB")
if mem['total_gb'] <= available:
    print("✅ Will fit!")
else:
    print(f"❌ Won't fit - need {mem['total_gb'] - available:.2f} GB more")
    
    # Find max context for batch=1
    max_ctx = calculator.find_max_batch_or_context(available, 'batch_size', 1)
    print(f"\nMax context length for batch=1: {max_ctx} tokens")
