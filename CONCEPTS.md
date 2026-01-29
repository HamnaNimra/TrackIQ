# 📚 Core Concepts Guide

This document explains key performance engineering concepts used throughout AutoPerfPy.

## Table of Contents
0. [Architecture: trackiq vs autoperfpy](#architecture-trackiq-vs-autoperfpy)
1. [Latency Basics](#latency-basics)
2. [Percentiles (P99, P95, P50)](#percentiles-p99-p95-p50)
3. [Throughput & Batching](#throughput--batching)
4. [LLM Inference Metrics](#llm-inference-metrics)
5. [GPU Memory Management](#gpu-memory-management)
6. [Regression Detection](#regression-detection)
7. [Testing & Quality Assurance](#testing--quality-assurance)

---

## Architecture: trackiq vs autoperfpy

```
┌─────────────────────────────────────────────────────────────────┐
│  autoperfpy (app layer)                                          │
│  CLI • Streamlit UI • TensorRT/automotive benchmarks • profiles  │
│  DNN pipeline • Tegrastats analyzers                             │
└───────────────────────────────┬─────────────────────────────────┘
                                │ imports
┌───────────────────────────────▼─────────────────────────────────┐
│  trackiq (core library)                                          │
│  platform/   collectors/   runner/   config/   results/   compare/│
│  errors.py   analyzers/   reporting/   profiles (registry)      │
└─────────────────────────────────────────────────────────────────┘
```

- **Collectors**: Plugins that gather metrics (synthetic, psutil, NVML, Tegrastats in app). Each implements `start()`, `sample(timestamp)`, `stop()`, `export()` → `CollectorExport`.
- **Runner**: `BenchmarkRunner` runs a collector for a fixed duration and sample interval, returns `CollectorExport` (samples + summary).
- **Results schema**: `CollectorExport` (collector_name, start_time, end_time, samples, summary, config); `AnalysisResult` (name, timestamp, metrics). Summary is nested (e.g. `latency.p99_ms`, `throughput.mean_fps`).
- **Comparison logic**: `trackiq.compare` provides `RegressionDetector`, `RegressionThreshold`, `MetricComparison`. Save baselines, compare current run to baseline, detect regressions by threshold (e.g. latency +5%, throughput -5%). Used by `autoperfpy compare`.

---

## Latency Basics

### What is Latency?
Latency is the **time it takes to complete a single operation** or request.

```
Request ─────────────────────> Response
         |<── Latency ──>|
```

### Why Latency Matters
- **User Experience**: Users notice latencies >100ms as lag
- **Real-time Systems**: Autonomous vehicles need <50ms latency
- **Chatbots**: Users expect first token within 1-2 seconds (TTFT)

### Types of Latency
| Type | Description | Example |
|------|-------------|---------|
| **Inference Latency** | Time to run the ML model | 25ms for YOLO detection |
| **End-to-End Latency** | Full pipeline time (input → output) | 30ms (25ms inference + 5ms preprocessing) |
| **Time-to-First-Token** | Latency before first output appears | 800ms for LLM |
| **Time-per-Token** | Latency to generate each additional token | 50ms per token for LLM |

---

## Percentiles (P99, P95, P50)

### Why Percentiles?
Not all latencies are the same. Some requests are fast, some are slow:

```
Latencies (sorted): [10ms, 12ms, 15ms, 18ms, 20ms, 22ms, ..., 45ms, 50ms, 55ms]
                     ↓              ↓              ↓              ↓              ↓
                    P1            P50           P90            P99           P100
                  (Fastest)     (Median)    (Slow)        (Very Slow)    (Slowest)
```

### Understanding the Metrics

#### **P50 (Median)**
- **Definition**: 50% of requests are faster, 50% are slower
- **Use Case**: Understand typical/average performance
- **Example**: If P50 = 25ms, then typical inference takes 25ms

#### **P95**
- **Definition**: 95% of requests are faster than this, only 5% are slower
- **Use Case**: Understand "good" performance
- **Example**: If P95 = 30ms, then 95% of requests complete within 30ms

#### **P99 (Most Important)**
- **Definition**: 99% of requests are faster than this, only 1% are slower
- **Use Case**: Understand worst-case (but not extreme) performance
- **Example**: If P99 = 45ms, then 99% of requests complete within 45ms
- **Why it matters**: 1% of requests might be from your most important users!

### Real-World Example

```python
latencies = [10, 11, 12, ..., 44, 45]  # 100 measurements

P50 = 25ms   # 50 requests faster, 50 slower
P95 = 30ms   # 95 requests faster, 5 slower
P99 = 45ms   # 99 requests faster, 1 slower
```

### The Problem with Averages

```
Scenario A: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
Average: 25ms, P99: 29ms → Consistent performance ✓

Scenario B: [5, 5, 5, 5, 5, 5, 5, 5, 5, 1005]
Average: 105ms, P99: 1005ms → Unpredictable spikes ✗
```

**Lesson**: Always look at P99 to catch problems with percentiles!

---

## Throughput & Batching

### Throughput
Throughput is **how many requests you can handle per second**.

```
Throughput = Requests per Second (RPS) or Images per Second (IPS)

Example: 
- Inference latency: 10ms per image
- Throughput: 1 / 0.01s = 100 images/second
```

### The Batching Trade-off

When you increase **batch size**, something interesting happens:

#### Small Batch (Size = 1)
```
Input: [Image1]
  │
  ├─ Overhead: 1ms (GPU kernel setup)
  ├─ Processing: 5ms (computation)
  └─ Total: 6ms per image
  
Throughput: 1/0.006 = 167 images/sec
```

#### Large Batch (Size = 16)
```
Input: [Image1, Image2, ..., Image16]
  │
  ├─ Overhead: 1ms (GPU kernel setup once)
  ├─ Processing: 5ms × 16 = 80ms (computation amortized)
  └─ Total: 81ms for 16 images = 5.06ms per image
  
Throughput: 1/0.00506 = 197 images/sec ✓ (higher!)

BUT per-image latency in batch: 81ms (image waits for batch!) ✗
```

### The Trade-off Visualization

```
         Latency
            ↑
            │      Small batch: Low latency, low throughput
            │      •
            │        \
            │         \____  Large batch: High throughput, high latency
            │              •
            └────────────────────→ Throughput
```

### Choosing Batch Size

| Use Case | Batch Size | Reason |
|----------|-----------|--------|
| **Real-time (Chatbot)** | 1 | Minimize latency for responsiveness |
| **Inference Service** | 8-32 | Balance latency & throughput |
| **Batch Processing** | 64-256 | Maximize throughput, latency doesn't matter |
| **Automotive** | 1-4 | Real-time constraints (<100ms) |

---

## LLM Inference Metrics

### Two Phases of LLM Inference

#### **Phase 1: Prefill (Processing Input)**
```
User Input: "What is machine learning?"
           ↓
     Process all tokens
           ↓
  Generate First Token (output)
           
Time: ~800ms  ← This is TTFT (Time-To-First-Token)
Problem: User sees nothing for 800ms!
```

#### **Phase 2: Decode (Generating Output)**
```
"Machine learning is..." (already generated)
           ↓
    Generate one more token
           ↓
  "Machine learning is a..."
           
Time: ~50ms per token  ← This is Time-Per-Token
Benefit: User sees tokens appearing in real-time (streaming)
```

### Key Metrics

| Metric | Value | Impact |
|--------|-------|--------|
| **TTFT** | 800ms | How long user waits before first response |
| **Time-Per-Token** | 50ms | How fast response streams |
| **Throughput** | 20 tokens/sec | Maximum concurrent users |

### Example Timeline

```
t=0ms      ━━━━━━━ User sends request
t=800ms    ━━━━━━━ First token appears (TTFT) 🎯
t=850ms    ━━━━━━━ Second token appears (50ms per token)
t=900ms    ━━━━━━━ Third token appears (50ms per token)
...
```

### Optimizing Each Phase

**Prefill Optimization**:
- Batch multiple user requests together
- Use smaller model for fast prefill
- Pre-cache common prompts

**Decode Optimization**:
- Process one token at a time (KV cache is already populated)
- Reduce model size for faster token generation
- Use speculative decoding (guess next tokens)

---

## GPU Memory Management

### Types of GPU Memory During Inference

```
GPU Memory Layout:
┌─────────────────────────────────────┐
│ Model Weights (Static)              │
│ - 7B LLM: ~14GB (fp16)              │
├─────────────────────────────────────┤
│ KV Cache (Grows per token)          │
│ - Increases with sequence length    │
├─────────────────────────────────────┤
│ Activation Memory (Temporary)       │
│ - Used during forward pass          │
├─────────────────────────────────────┤
│ Free Memory                         │
└─────────────────────────────────────┘
```

### KV Cache Calculation

For each token generated, we store:
- **K (Key)** vector: `layers × heads × head_size × batch_size`
- **V (Value)** vector: `layers × heads × head_size × batch_size`

**Example**:
```
Model: 32 layers, 32 heads, 128 head_size, batch_size=1, fp16 (2 bytes)

Memory per token:
= 2 × 32 × 32 × 128 × 1 × 2 bytes
= 524,288 bytes
= ~512 KB per token

For 2000 token sequence:
= 512 KB × 2000 = 1 GB just for KV cache!
```

### OOM (Out of Memory) Prevention

```python
# Calculate maximum sequence length
max_seq_len = available_gpu_memory / (kv_cache_size_per_token)

# Monitor during inference
if current_memory > threshold:
    reduce_batch_size()
    or
    reduce_max_sequence_length()
```

---

## Summary Table

| Concept | Key Takeaway |
|---------|--------------|
| **Latency** | Time for one operation; measure P99, not average |
| **P99** | 99% of requests faster than this; catches outliers |
| **Throughput** | Requests per second; improves with larger batches |
| **Batching Trade-off** | Bigger batches = higher throughput but higher latency |
| **TTFT** | Time before first token; critical for user experience |
| **Time-per-Token** | Speed of token generation; enables streaming |
| **KV Cache** | Grows linearly with sequence length; major memory consumer |

---

## Further Reading

- **NVIDIA Triton Inference Server**: https://docs.nvidia.com/deeplearning/triton-inference-server/
- **LLM Inference Optimization**: https://www.microsoft.com/en-us/research/publication/
- **GPU Memory Best Practices**: https://docs.nvidia.com/deeplearning/cudnn/

---

**Now you understand the core concepts! Check the examples in [README.md](README.md) to see them in action.** 🚀
