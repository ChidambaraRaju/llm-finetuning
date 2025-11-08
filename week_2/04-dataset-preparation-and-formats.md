# Lesson Notes: Dataset Preparation for Fine-Tuning

This lesson covers the most critical component of any fine-tuning project: the **data**. The quality, structure, and format of your dataset will have the single biggest impact on your final model's behavior.

## Everything Starts with Data

In any LLM project, there are three main levers:
1.  **The Dataset:** What the model learns from.
2.  **The Model Architecture:** The "engine" (e.g., LLaMA, Mistral). We usually don't change this.
3.  **The Loss Function:** How the model is scored.

The **dataset is the most important lever** you have for shaping the model's outcome. The architecture is fixed, but the data is what you control. The old saying **"garbage in, garbage out"** is the most important rule in fine-tuning.

---

## Dataset Sources and Creation

Your dataset should look like the interactions you expect your model to handle in production.

### Dataset Sources

1.  **Internal Data:** Your company's documents, FAQs, transcripts, etc.
2.  **Public Datasets:** Open-source datasets used for prototyping or as a base (e.g., `Alpaca`, `OASST`).
3.  **Custom Data:** Created from scratch when the other sources aren't a good fit.

### Creation Approaches

1.  **Human-Labeled:** The "gold standard." Subject-matter experts write or verify each example. This is slow and expensive but produces highly reliable data.
2.  **Synthetic (LLM-Generated):** A large model (like GPT-4) is used to generate thousands of examples from a few "seed" prompts. This is fast and scalable but *must* be validated, as it can be repetitive or biased.
3.  **Hybrid:** The most common and effective approach. An LLM generates data, and a human reviews, filters, and ranks it. This balances the speed of synthetic generation with the quality of human review.

---

## Dataset Formats (Schemas)

Different training stages require different data formats. We are focused on the **Supervised Fine-Tuning (SFT)** stage.

| Format | Primary Use Case | Key Fields | Example Dataset |
| :--- | :--- | :--- | :--- |
| **Pretraining** | Base model training | `text` | The Pile, WikiText |
| **Instruction** | **SFT (Single-turn)** | `instruction`, `input`, `output` | Alpaca, FLAN |
| **Conversation** | **SFT (Multi-turn Chat)** | `messages` (list of `role`, `content`) | OASST, ShareGPT |
| **Preference** | Alignment (DPO/ORPO) | `prompt`, `chosen`, `rejected` | HH-RLHF |

### 1. Instruction Format (Alpaca-style)

This is the simplest format, ideal for single-turn tasks like Q&A, summarization, or translation.

**JSON Example:**
```json
{
  "instruction": "Translate to French",
  "input": "Hello, world!",
  "output": "Bonjour, le monde!"
}
```

### 2. Conversation Format (OASST-style)

This is the standard for building multi-turn chatbots. The dataset consists of a list of messages, each with a `role` (like `system`, `user`, or `assistant`).

**JSON Example:**
```json
{
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "What is 2+2?" },
    { "role": "assistant", "content": "4" }
  ]
}
```

---

## Format vs. Model Input: The "Chat Template"

This is a critical concept. The model **does not** see the JSON structure with fields like `instruction` or `messages`.

The model *only* sees **a single, long string of text**.

The structured formats (like JSON) are for our convenience. A **chat template** is used to automatically convert this structured data into the final string format the model was trained on, adding all the necessary special tokens.

**Example:**
The Alpaca-style JSON from before...
```json
{
  "instruction": "Summarize the text below.",
  "input": "Large language models are revolutionizing AI.",
  "output": "LLMs are transforming artificial intelligence."
}
```
...is rendered by the tokenizer's **chat template** into this single string:

```text
### Instruction:
Summarize the text below.

### Input:
Large language models are revolutionizing AI.

### Response:
LLMs are transforming artificial intelligence.<|end_of_text|>
```

Frameworks like Hugging Face `TRL` handle this conversion automatically, often with a single command:
`tokenizer.apply_chat_template(example, tokenize=False)`

---

## The Practical Workflow in Hugging Face

1.  **Find & Load Datasets:** You can find thousands of datasets on the Hugging Face Hub and load them instantly.
    ```python
    from datasets import load_dataset
    dataset = load_dataset("tatsu-lab/alpaca")
    print(dataset["train"][0])
    ```

2.  **Create Your Own Dataset:** You can create datasets programmatically. Libraries like `distilabel` are designed to help you use LLMs to generate synthetic data at scale.

3.  **Validate and Clean Data:** **Do not trust your data.** You must validate it. Wasting GPU time on bad data is the most common mistake.
    * Check for empty or truncated responses.
    * Remove duplicates.
    * Ensure the schema is consistent.
    * Check length distributions to find outliers.
    ```python
    # Example: Filter out rows with empty outputs
    dataset = dataset.filter(lambda example: example["output"].strip() != "")
    ```

4.  **Save and Publish:** Once your dataset is clean, save it to disk or push it to the Hugging Face Hub so you (and your team) can reuse it.
    ```python
    # Save to disk
    dataset.save_to_disk("data/alpaca_clean")

    # Or push to the Hub
    dataset.push_to_hub("your-username/alpaca-clean")
    ```

---

## Acknowledgements

These notes are based on the "Dataset Preparation for LLM Fine-Tuning" lesson from the **LLM Engineering & Deployment Certification Program** by **Ready Tensor**.