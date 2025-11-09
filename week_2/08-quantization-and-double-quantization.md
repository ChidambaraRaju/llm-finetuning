# Lesson Notes: Quantization and Double Quantization

This lesson covers the core concepts of model compression. You've learned that data types like 4-bit (INT4) exist, but this lesson explains **how a model is converted** from its large, high-precision (like FP32) format into these tiny, efficient formats.

This process, **quantization**, is what makes it possible to fit a model that needs 28 GB of memory into less than 4 GB, all without catastrophic loss of performance.

## The Core Idea: Mapping Floats to Integers

Quantization is the process of mapping a continuous range of floating-point numbers (e.g., -2.5 to +3.0) into a small, discrete set of integers (e.g., for INT8, the 256 bins from -128 to 127).

This mapping is defined by two key parameters:
1.  **Scale:** The "width" of each integer bin. For example, a scale of 0.015 might mean that moving from the integer `10` to `11` is equivalent to moving from `0.150` to `0.165` in float space.
2.  **Zero-Point:** The integer value that maps to the real value `0.0`. This is crucial for correctly representing the "true zero" of the original data.

### The Quantization Process
The process involves a few simple steps:
1.  **Find Range:** Find the `min` and `max` float values in the weights.
2.  **Calculate Scale & Zero-Point:** Compute the `scale` and `zero-point` that best map this float range to the target integer range (e.g., -128 to 127).
3.  **Quantize:** Convert every float weight to its corresponding integer using the `scale` and `zero-point`.
4.  **Dequantize:** When the model needs to use the weight, it's converted back to a float using the formula:
    `dequantized_value = (quantized_value - zero_point) * scale`

This process gives us a 4x (for INT8) or 8x (for INT4) memory reduction.

---

## The Outlier Problem (And Its Solution)

There's a major problem with the simple approach above: **outliers**.

What if your model has 1 million weights between -1.0 and 1.0, but *one* single weight has a value of 8.5?

If you use *one global scale* for the entire model, that single outlier will stretch the entire range (from -1.0 to 8.5). All the precision will be wasted, and the 1 million important weights will be "squashed" into just a few integer bins, destroying the model's performance.

### Solution: Blockwise Quantization

Instead of one global scale, we use **Blockwise Quantization**.
* We divide the model's weights into small, independent **chunks** (or "blocks"), typically of 64 or 128 weights.
* We then calculate a *separate* `scale` and `zero-point` for **each block**.

An outlier in one block will only affect that single block; the other 99.9% of the model's weights will be quantized with perfect precision for their local range. This is the standard for all modern quantization libraries.

---

## The Metadata Problem (And Its Solution)

Blockwise quantization is great, but it creates a new, hidden problem: **metadata overhead**.

A 7-billion parameter model might have over 100 million blocks. Each block needs:
* 1 `scale` (stored in FP32 $\rightarrow$ 4 bytes)
* 1 `zero-point` (stored in FP32 $\rightarrow$ 4 bytes)

`109 million blocks * (4 + 4) bytes/block` $\approx$ **872 MB of memory!**
This extra 0.9 GB of metadata eats into our memory savings.

### Solution: Double Quantization (DQQ)

**Double Quantization** is the clever trick of "compressing the compressors." It recognizes that the metadata (all those `scale` and `zero-point` values) is just another set of numbers. So, why not quantize *them* too?

1.  **First Pass (Quantization):** The model weights are quantized (e.g., to 4-bit) using blocks, which creates 109 million 32-bit `scale` and `zero-point` values.
2.  **Second Pass (Double Quantization):** We treat all 109 million `scale` values as a new dataset and *quantize them* (e.g., from 32-bit down to 8-bit). We do the same for all the `zero-point` values.

This second pass adds its own tiny "meta-metadata," but the savings are huge. The 872 MB of overhead is crushed down to about 220 MB.

For a 7B model, this takes the total size from ~4.37 GB (with single quantization) down to **~3.72 GB** (with double quantization), saving an additional ~15%.

---

## NF4: The Optimal 4-bit Format

We've figured out *how* to quantize, but *what* 4-bit format do we use?

* **Standard INT4:** Has 16 evenly spaced bins (e.g., -8, -7, ..., +7).
* **The Problem:** Model weights are *not* evenly distributed. They are in a **normal distribution** (a bell curve), with most values clustered very close to zero. Standard INT4 "wastes" its bins on values that are rarely used.

* **NF4 (NormalFloat-4):**
    This is a special 4-bit format designed *specifically* for normally-distributed model weights.
    Its 16 bins are **not evenly spaced**. They are **dense near zero** (where most weights are) and sparse at the extremes.


This optimized bin spacing means NF4 can represent the original model weights with *far less error* than standard INT4, significantly improving model quality for no extra memory cost.

---

## Putting It All Together: QLoRA

These three techniques form the core of **QLoRA** and are implemented in the `bitsandbytes` library. When you load a model with this config, you are enabling all of them:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# This config object enables all the optimizations
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 1. load_in_4bit=True:
#    Activates 4-bit blockwise quantization.

# 2. bnb_4bit_quant_type="nf4":
#    Uses the superior NormalFloat-4 format for the weights.

# 3. bnb_4bit_use_double_quant=True:
#    Compresses the quantization metadata (scales/zeros) to save more memory.

# 4. bnb_4bit_compute_dtype=torch.bfloat16:
#    Keeps math operations in stable, high-precision BF16, even
#    though the weights are *stored* in 4-bit.

# Load the model with this config
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
```

This combination (Blockwise + DQQ + NF4) is what allows a 14 GB model to run in under 4 GB, enabling the fine-tuning of massive models on consumer GPUs.

---

## Acknowledgements

These notes are based on the "Quantization and Double Quantization: How to Compress LLMs Efficiently" lesson from the **LLM Engineering & Deployment Certification Program** by **Ready Tensor**.