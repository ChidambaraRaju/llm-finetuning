# Lesson Notes: Tokenization and Padding

This lesson covers the essential preprocessing steps that convert raw text into a format LLMs can understand: numbers. Models don't read words; they process numerical tensors. This process involves three key concepts: **Tokenization**, **Padding**, and **Attention Masking**.

---

## 1. Tokenization: From Text to Tokens

Language models can't process a string like "Hello". They must first convert it into a numerical ID. **Tokenization** is the process of breaking text down into small, meaningful units (tokens) and mapping them to unique integers.

### The Evolution of Tokenization

1.  **Word-Level:**
    * **Example:** `"The cat"` $\rightarrow$ `[1, 2]`
    * **Limitation:** The vocabulary size becomes enormous (hundreds of thousands of words), and it can't handle new or misspelled words ("ChatGPT").
2.  **Character-Level:**
    * **Example:** `"The cat"` $\rightarrow$ `[84, 104, 101, 32, 99, 97, 116]`
    * **Limitation:** Very inefficient. Sequences become extremely long, and it's harder for the model to learn meaningful concepts from individual letters.
3.  **Subword-Level (The Modern Standard):**
    * **Example:** `"unbelievable"` $\rightarrow$ `["un", "believ", "able"]`
    * **Strength:** This is the perfect balance. Common words ("The", "cat") are kept as single tokens, while complex or new words are broken into reusable subword pieces.
    * This allows the model to handle *any* new word (like "ChatGPT-4.5-turbo") by breaking it down into known fragments (`["Chat", "GPT", "-", "4", ".", "5", ...]`), keeping the vocabulary small and efficient (e.g., 30k-50k tokens).

### Special Tokens

Every tokenizer also includes special tokens to give the model structural information:
* `bos_token` (Begin Of Sequence): Marks the start of a text (e.g., `<|begin_of_text|>`).
* `eos_token` (End Of Sequence): Marks the end of a text (e.g., `<|end_of_text|>`).
* `pad_token` (Padding): A filler token (e.g., `<pad>`) used to make sequences the same length.
* `unk_token` (Unknown): Represents a token that is not in the vocabulary (rare in subword tokenizers).

---

## 2. Padding: Solving the Length Problem

Models train on data in "batches" (groups of sentences) for efficiency. The problem is that sentences have different lengths:
* `"Hi!"` (3 tokens)
* `"What's the weather like?"` (8 tokens)

GPU hardware requires all inputs in a batch to be a uniform shape (i.e., the same length). **Padding** solves this by adding a special `pad_token` to the shorter sequences until they all match the length of the longest one.

**Example (Batch padded to length 8):**
* **Original:** `[1, 2345, 3]`
* **Padded:** `[1, 2345, 3, 0, 0, 0, 0, 0]`
    *(Assuming `0` is the `pad_token` ID)*
* **Original:** `[1, 456, 78, 901, 234, 567, 8, 3]`
* **Padded:** `[1, 456, 78, 901, 234, 567, 8, 3]`

---

## 3. Attention Masks: Ignoring the Padding

Padding creates a new, serious problem: the model will try to learn from the filler tokens, treating them as real words. This is like trying to teach a student by giving them a book where half the pages are filled with the word "filler."

The **Attention Mask** solves this. It's a second list of 1s and 0s that tells the model exactly which tokens to pay attention to and which to ignore.

* `1` = Pay attention (real token)
* `0` = Ignore (padding token)

**Example (Batch padded to length 5):**
* **Input IDs:** `[1, 2345, 3, 0, 0]`
* **Attention Mask:** `[1, 1, 1, 0, 0]`

When the model receives this, it knows to process the first three tokens (`[1, 2345, 3]`) and completely ignore the last two (`[0, 0]`). This prevents the padding from "polluting" the learning process.

Hugging Face tokenizers can do all three steps for you automatically:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
batch = ["Hello world!", "How are you doing today?"]

# Tokenize, pad, and create attention masks all at once
inputs = tokenizer(
    batch,
    padding=True,
    return_tensors="pt", # Return PyTorch Tensors
    return_attention_mask=True
)

# inputs['input_ids'] -> The padded token IDs
# inputs['attention_mask'] -> The corresponding attention mask
```

---

## Best Practices & Pitfalls

1.  **Always use the *exact* same tokenizer** for fine-tuning as the one used for the base model. A mismatch will scramble your inputs.
2.  **Always define a `pad_token`**. Some models (like GPT-2) don't have one by default. A common trick is to set it to the `eos_token`:
    ```python
    if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token
    ```
3.  **Always pass the `attention_mask`** to the model during training. Forgetting this is a silent error that will degrade your model's performance.

---

## Acknowledgements

These notes are based on the "Tokenization and Padding: Preparing Text Data for LLM Training" lesson from the **LLM Engineering & Deployment Certification Program** by **Ready Tensor**.