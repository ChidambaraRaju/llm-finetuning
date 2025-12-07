# Lesson Notes: Techniques to Reduce GPU Memory Requirements

This lesson covers the practical "bag of tricks" you can use to squeeze large models onto a single GPU. Before scaling to multi-GPU setups (which are complex and expensive), you should master these techniques to maximize the efficiency of the hardware you already have.

These optimizations focus on four key areas: **Activations**, **Precision**, **Sequence Efficiency**, and **Attention**.

## 1. Activation Management (Trading Compute for Memory)

Activations (the intermediate values stored during the forward pass) are often the biggest memory bottleneck, scaling with batch size and sequence length.

### Gradient Accumulation
Big batch sizes stabilize training but consume huge amounts of VRAM. Gradient accumulation allows you to simulate a large batch size without holding it all in memory at once.

* **How it works:** Instead of running a batch of 64 samples at once, you run 8 "micro-batches" of 8 samples each. You calculate gradients for each micro-batch and *accumulate* (add) them up. You only update the model weights after all 8 micro-batches are finished.
* **Result:** You get the mathematical equivalent of a large batch size with the memory footprint of a small micro-batch.
* **Trade-off:** Training takes slightly longer due to overhead, but memory usage stays flat.


### Gradient Checkpointing
This technique drastically reduces activation memory by trading it for extra computation.

* **How it works:** Normally, you store *all* intermediate activations from the forward pass to use during the backward pass. With checkpointing, you **throw away** most of these activations. When the backward pass needs them, it **re-computes** them on the fly from specific "checkpoint" layers.
* **Result:** Can reduce activation memory by 4x-5x, often fitting a model that would otherwise crash.
* **Trade-off:** Increases training time by ~20-30% because of the re-computation.


## 2. Precision Choices (Reducing State Memory)

The optimizer states (especially for Adam) can take up more memory than the model weights themselves.

### Mixed Precision (Recap)
Storing weights and gradients in **BF16** or **FP16** cuts memory usage in half compared to FP32. This is standard practice.

### 8-Bit Optimizers
Standard Adam keeps two states (momentum and variance) for every parameter in **FP32** (8 bytes per param).
* **The Fix:** 8-bit optimizers (like `paged_adamw_8bit` from `bitsandbytes`) compress these states into 8-bit integers.
* **Impact:** Reduces optimizer memory from **8GB** down to **2GB** for a 1B parameter model—a massive 6GB saving.
* **Performance:** Almost no loss in accuracy for fine-tuning tasks.

## 3. Sequence-Level Efficiency (Reducing Wasted Compute)

Standard padding wastes memory on useless "pad" tokens.

### Dynamic Padding
Instead of padding every sentence in the entire dataset to the maximum length (e.g., 2048), you pad each *batch* only to the length of the longest sequence *in that specific batch*.
* **Result:** Drastically reduces the number of padding tokens processed, saving memory and speed.

### Sequence Packing
This is even more efficient. You concatenate multiple short examples into one long sequence (separated by an EOS token) to fill the context window completely.
* **Example:** Instead of three sequences `[120, 180, 200]` all padded to 512 (wasting ~1000 tokens), you pack them into a single 512 sequence with only 12 padding tokens.
* **Result:** Zero wasted memory on padding.

## 4. Attention Optimizations

### FlashAttention
Standard attention computes a massive $N \times N$ matrix that scales quadratically with sequence length.
* **How it works:** FlashAttention uses a "tiling" approach to compute attention in small blocks directly in the GPU's fast SRAM, avoiding the need to write the huge attention matrix to HBM (main GPU memory).
* **Impact:** Reduces attention memory from Quadratic ($O(N^2)$) to Linear ($O(N)$).
* **Benefit:** Enables training on much longer sequences (4k, 8k, 32k+) without running out of memory.


---

## Practical Implementation: Hugging Face `TrainingArguments`

You can enable almost all of these optimizations with simple flags in your training config:

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./results",
    # 1. Mixed Precision
    bf16=True, 
    
    # 2. Gradient Accumulation (Batch size 2 * 16 steps = effective batch 32)
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    
    # 3. Gradient Checkpointing (Saves huge activation memory)
    gradient_checkpointing=True,
    
    # 4. 8-bit Optimizer (Requires bitsandbytes)
    optim="paged_adamw_8bit",
    
    # 5. FlashAttention (Automatic if hardware supports it + torch 2.0)
    # torch_compile=True (Optional, can further optimize)
)
```

## Summary Checklist

| Technique | Primary Target | Trade-off | When to use? |
| :--- | :--- | :--- | :--- |
| **Mixed Precision** | Parameters & Gradients | None | Always. |
| **Gradient Accumulation** | Peak Activation Memory | Slightly slower | When batch size doesn't fit. |
| **Gradient Checkpointing** | Peak Activation Memory | ~20% Slower | When you hit OOM despite small batches. |
| **8-Bit Optimizer** | Optimizer States | None | Almost always for fine-tuning. |
| **FlashAttention** | Attention Matrix Memory | None (Faster!) | Always (if GPU supports it). |

---

## Acknowledgements

These notes are based on the "LLM Training: Techniques to Reduce GPU Memory Requirements" lesson from the **LLM Engineering & Deployment Certification Program** by **Ready Tensor**.