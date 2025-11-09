# Lesson Notes: Instruction Fine-Tuning & Assistant-Only Masking

This lesson covers one of the most critical concepts in Supervised Fine-Tuning (SFT): **Assistant-Only Masking** (also called Loss Masking).

This technique is the key to teaching a model how to be a helpful *assistant* (which replies to instructions) instead of just a *base model* (which completes text).

## The Selective Learning Challenge

In SFT, our dataset is made of "instruction" and "response" pairs:
`User: What's 2+2?`
`Assistant: 4`

A base model learns by default to predict *every* next token. If we fine-tune on this data without any changes, the model will learn to predict the user's prompt *and* the assistant's answer.

**This is the wrong behavior.**

At inference time, the user will provide the prompt. The model's *only* job is to generate the assistant's response. Training the model to predict the user's text teaches it to do the wrong thing (e.g., echo the prompt, or continue the user's question with another question).

## The Solution: Assistant-Only Masking

The solution is to tell the model to **only learn from the assistant's tokens**. We do this by "masking" all other tokens from the loss calculation.

### How it Works: The `–100` Label

We differentiate between what the model *sees* and what it *learns from*.

1.  **What the Model Sees (Attention Mask):** The model must *see* the entire conversation, including the user's prompt, to have context for its answer. The **Attention Mask** (from the previous lesson) is all `1`s for all real tokens (user and assistant).

2.  **What the Model Learns From (Loss Masking):** We provide a separate list of `labels` to the loss function. In PyTorch, the Cross-Entropy Loss function is hard-coded to **completely ignore** any token with the special label `–100`.

We use this to create our list of labels:
* Set the label for every `User` and `System` token to `–100`.
* Keep the original token ID for every `Assistant` token.

---

### Visualization of the Process

Imagine this simple, tokenized exchange:

* **Input Tokens:** `[User:, What's, 2+2?, Assistant:, 4]`

Here is what the model is given for training:

1.  **`input_ids` (What the model *sees*):**
    * `[34, 56, 90, 345, 12, 789]` (The token IDs for the full conversation)
2.  **`attention_mask` (What to *pay attention* to):**
    * `[1, 1, 1, 1, 1, 1]` (Pay attention to *all* tokens)
3.  **`labels` (What to *learn from*):**
    * `[-100, -100, -100, -100, -100, 789]`

**Result:**
The model sees the full context `[User:, What's, 2+2?, Assistant:]` to predict the next token. It predicts a probability distribution. The loss function then compares this prediction *only* to the one unmasked label, `[789]` (the token for "4").

All the `–100` tokens are skipped. The model's "error" (loss) is calculated *only* on its ability to generate the assistant's response.

### Example: Multi-Turn Conversation

This technique is essential for multi-turn chat. The model learns to *only* predict the assistant's part of the dialogue.

`System: You are a math tutor.`
`User: What's 2+2?`
`Assistant: It's 4.`
`User: And 3+3?`
`Assistant: That's 6.`

**How the model learns:**
* `[System: You are a math tutor.]` $\rightarrow$ **[MASKED, loss is ignored]**
* `[User: What's 2+2?]` $\rightarrow$ **[MASKED, loss is ignored]**
* `[Assistant: It's 4.]` $\rightarrow$ **[LEARNED, loss is calculated]**
* `[User: And 3+3?]` $\rightarrow$ **[MASKED, loss is ignored]**
* `[Assistant: That's 6.]` $\rightarrow$ **[LEARNED, loss is calculated]**

The model sees the full conversation for context but is *only* graded on its ability to generate "It's 4" and "That's 6" in the correct spots.

---

## How Masking is Implemented in Practice

**You usually don't have to do this manually.**

Modern fine-tuning frameworks like **Hugging Face TRL (SFTTrainer)** and **Axolotl** handle this **automatically**.

When you format your dataset using a **chat template** (as discussed in Lesson 4), the tokenizer and trainer work together to:
1.  Format the `[USER]`, `[ASSISTANT]` roles correctly.
2.  Automatically apply the `–100` mask to all tokens that are not part of the `[ASSISTANT]` role.

Your only job is to **ensure your dataset is formatted correctly** with the proper roles. The framework handles the masking.

## Debugging Masking Issues

If masking is set up incorrectly, it's the #1 cause of strange fine-tuning behavior.

* **Symptom:** The model echoes the user's input or asks questions back.
    * **Diagnosis:** The `User` tokens are *not* being masked. The model is learning to predict them.
    * **Fix:** Check your chat template and data format.

* **Symptom:** The training loss doesn't decrease (or stays at 0).
    * **Diagnosis:** *All* tokens are being masked (all `–100`). The model has nothing to learn from.
    * **Fix:** Check that your `Assistant` tokens are *not* masked and have the correct token IDs.

* **Symptom:** The model generates system prompts (e.g., "You are a helpful assistant...") in its answers.
    * **Diagnosis:** The `System` tokens are *not* being masked.
    * **Fix:** Ensure your system prompts are also being masked to `–100`.

---

## Acknowledgements

These notes are based on the "Instruction Fine-Tuning in LLMs: Assistant-Only Masking Explained" lesson from the **LLM Engineering & Deployment Certification Program** by **Ready Tensor**.