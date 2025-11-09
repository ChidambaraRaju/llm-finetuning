# Lesson Notes: Data Types in Deep Learning

This lesson covers the critical, low-level engineering decision of which **data type** (or **precision**) to use for a model. This single choice determines how much memory the model consumes, how fast it runs, and how stable its training is.

Every number in an LLM (weights, gradients, activations) is stored in a specific format. This lesson explains the trade-offs between them.

## Floating-Point Data Types (Floats)

Floats are used for training and fine-tuning. Every float is made of three parts:
1.  **Sign:** (1 bit) Is the number positive or negative?
2.  **Exponent:** (Multiple bits) Controls the **range** of the number (how large or small it can be, like $10^{38}$).
3.  **Mantissa:** (Multiple bits) Controls the **precision** of the number (how many decimal places it can store).

The key trade-off in all data types is **Range vs. Precision**.

![data-type-comparision](../images/data-types-comparision.png)
---

### FP32 (Float32 or "Full Precision")
* **Total Bits:** 32 (1 sign, 8 exponent, 23 mantissa)
* **Range:** $\pm3.4 \times 10^{38}$ (Huge)
* **Precision:** ~7-8 decimal digits (Very high)
* **Use:** This is the original standard. It's extremely stable and precise but very memory-heavy.
* **Problem:** A 7-billion parameter model in FP32 requires **28 GB** of GPU memory just to store the weights, making it too large for most GPUs.

---

### FP16 (Float16 or "Half Precision")
* **Total Bits:** 16 (1 sign, 5 exponent, 10 mantissa)
* **Range:** $\pm65,504$ (Very small!)
* **Precision:** ~3-4 decimal digits (Good)
* **Use:** Halves the memory requirement (7B model $\rightarrow$ 14 GB).
* **Problem:** The tiny **5-bit exponent** means the range is extremely limited. During training, gradients can easily become larger than 65,504 (**overflow**) or smaller than the minimum value (**underflow**). This causes "NaN" (Not a Number) errors and makes training unstable without complex fixes like loss scaling.

---

### BF16 (BFloat16 or "Brain Float")
* **Total Bits:** 16 (1 sign, 8 exponent, 7 mantissa)
* **Range:** $\pm3.4 \times 10^{38}$ (Huge, same as FP32)
* **Precision:** ~2-3 decimal digits (Lower)
* **Use:** This is the **modern standard for training**.
* **Key Insight:** BF16 keeps the same **8-bit exponent as FP32**, so it has the same massive range and avoids all the overflow/underflow problems of FP16. It sacrifices a bit of precision (7-bit mantissa), but deep learning models are highly resilient to this.
* **Support:** This is the default on modern GPUs (NVIDIA A100/H100, Google TPUs).

---

## Integer Data Types (Quantization)

Integers are used for **inference** and **quantized training** (like QLoRA). They are not floats; they are whole numbers. **Quantization** is the process of mapping a model's high-precision float weights (like BF16) into this much smaller integer range.

### INT8 (8-bit Integer)
* **Total Bits:** 8
* **Range:** Can represent 256 different values (e.g., -128 to +127).
* **Use:** A fast and reliable format for inference. A 7B model in INT8 takes only **7 GB** of memory.

---

### INT4 (4-bit Integer)
* **Total Bits:** 4
* **Range:** Can represent only 16 different values (e.g., -8 to +7).
* **Use:** Extreme compression, used for inference and as the base for **QLoRA**.

---

### NF4 (NormalFloat-4)
* **This is the most important 4-bit format.** The name "NormalFloat" is misleading; it is **still a 4-bit integer type**.
* **Key Insight:** Standard INT4 spaces its 16 values evenly (e.g., -8, -7, -6...). But model weights aren't evenly distributed; they are in a "normal" (Gaussian) distribution, clustered near zero.
* **NF4** is an optimized 4-bit format where the 16 available values are **not evenly spaced**. More values are "assigned" to be near zero, and fewer are out at the extremes, perfectly matching the distribution of model weights.
* **Benefit:** This results in a *much* lower precision loss compared to standard INT4.
* **Use:** This is the **default format used for QLoRA** to store the compressed 4-bit base model.

---

## Practical Memory Calculations

The formula for model size is simple:
**Memory (GB) = (Number of Parameters in Billions) $\times$ (Bytes per Parameter)**

* **FP32:** 4 bytes
* **BF16 / FP16:** 2 bytes
* **INT8:** 1 byte
* **INT4 / NF4:** 0.5 bytes

| Model Size | FP32 (4 bytes) | FP16/BF16 (2 bytes) | INT8 (1 byte) | INT4 (0.5 bytes) |
| :--- | :--- | :--- | :--- | :--- |
| **1B** | 4 GB | 2 GB | 1 GB | 0.5 GB |
| **7B** | 28 GB | 14 GB | 7 GB | 3.5 GB |
| **13B** | 52 GB | 26 GB | 13 GB | 6.5 GB |
| **70B** | 280 GB | 140 GB | 70 GB | 35 GB |

**Note:** This is *only* for the model weights. Training requires 3-4x this amount for gradients, optimizer states, and activations.

---

## Summary (Cheat Sheet)

| Phase | Recommended Precision | Reason |
| :--- | :--- | :--- |
| **Full Training** | **BF16** | Stable (no overflow) and efficient. |
| **Fine-Tuning** | **BF16** (or **FP16**) | Good balance of speed and precision. |
| **Inference** | **INT8** or **INT4 / NF4** | Small, fast, and cost-efficient. |

---

## Acknowledgements

These notes are based on the "Data Types in Deep Learning: FP32, FP16, BF16, INT8, INT4 Explained" lesson from the **LLM Engineering & Deployment Certification Program** by **Ready Tensor**.