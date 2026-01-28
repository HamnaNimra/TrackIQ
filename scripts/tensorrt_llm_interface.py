"""
Example script demonstrating how to use TensorRT LLM interface for model inference.

TensorRT LLM runtime to load a model engine,
    performs token generation based on a prompt,
    and decodes the output tokens back to text.

usage:
    python scripts/tensorrt_llm_interface.py

Author: Hamna
"""

import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner
import torch

# 1. Need to specify runtime config
config = tensorrt_llm.runtime.GenerationConfig(
    max_new_tokens=128,
    end_id=2,  # EOS token
    pad_id=0,
    temperature=1.0,
    top_k=1
)

# 2. Load model engine with proper batch size
engine_path = "model.engine"
runner = ModelRunner.from_dir(
    engine_dir=engine_path,
    rank=0  # GPU rank if using tensor parallelism
)

# 3. Tokenize input (can't pass raw string)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

prompt = "Hello, how are you?"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# 4. Run inference with proper config
outputs = runner.generate(
    batch_input_ids=input_ids,
    max_new_tokens=config.max_new_tokens,
    end_id=config.end_id,
    pad_id=config.pad_id,
    temperature=config.temperature,
    top_k=config.top_k,
    output_sequence_lengths=True,
    return_dict=True
)

# 5. Decode output tokens
output_ids = outputs['output_ids']
output_text = tokenizer.decode(output_ids[0][0], skip_special_tokens=True)

print(f"Input:  {prompt}")
print(f"Output: {output_text}")

# 6. Clean up
runner.shutdown()

# tensorrt_llm.runtime.shutdown_logger()
# Note: Uncomment the above line if you want to shutdown the logger explicitly

