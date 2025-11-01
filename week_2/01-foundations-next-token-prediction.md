# Lesson Notes: Understanding Next-Token Prediction

This lesson covers the most fundamental concept in LLM training and fine-tuning: **next-token prediction**. Every function of a Large Language Model, from pretraining to fine-tuning, is built on this single mechanism.

## The Big Idea: LLMs are Massive Classifiers

At their core, language models are not mysterious reasoning engines; they are enormous **classification systems**.

When a model generates a word, it's not just picking from a few options. It is performing a massive classification task:
1.  It looks at the sequence of text so far (e.g., "The cat sat on the...").
2.  It then computes a probability score for **every single token** in its vocabulary (which can be 50,000+ tokens).
3.  This distribution includes likely words ("mat", "floor") and highly unlikely words ("chemistry", "mitochondrion").
4.  The model then selects the next token from this probability distribution.

This single decision—which of the 50,000+ tokens should come next—is the core of all LLM behavior.


| Candidate Token | Probability |
| :--- | :--- |
| mat | 0.78 |
| floor | 0.12 |
| chair | 0.05 |
| bed | 0.02 |
| table | 0.01 |

## From Classification to Generation: The Autoregressive Loop

A model generates entire paragraphs by running this classification task in a **self-feeding loop**. This process is called **autoregression**, which means the model uses its own previous output as the input for the next step.

Here is the step-by-step flow:
* **User:** "What's the capital of France?"
* **Model Step 1:**
    * Input: "What's the capital of France?"
    * Predicts: "The"
* **Model Step 2:**
    * Input: "What's the capital of France? The"
    * Predicts: "capital"
* **Model Step 3:**
    * Input: "What's the capital of France? The capital"
    * Predicts: "of"
* **Model Step 4:**
    * Input: "What's the capital of France? The capital of"
    * Predicts: "France"
* **Model Step 5:**
    * Input: "What's the capital of France? The capital of France"
    * Predicts: "is"
* **Model Step 6:**
    * Input: "What's the capital of France? The capital of France is"
    * Predicts: "Paris"

The model isn't planning a full sentence. It is only ever predicting **one token at a time**, based on the complete history of tokens that came before it.

## Why Model Outputs Vary: Probabilistic Sampling

If you ask an LLM the same question twice, you often get different answers. This is not an error; it's an intentional feature called **probabilistic sampling**.

Instead of *always* picking the single token with the highest probability, the model "samples" from the top-ranked tokens. This introduces controlled randomness, allowing the model to be more creative and human-like.

This behavior is controlled by **inference parameters** like:
* **`Temperature`:** Controls the "boldness" of the sampling. Low temperature makes the model safe and predictable; high temperature makes it more creative or random.
* **`Top-k` / `Top-p`:** These methods filter the vocabulary down to a smaller set of likely tokens to sample from, ignoring the long tail of unlikely options.

## Pattern Matching, Not Reasoning

This is the most critical concept to grasp: **LLMs match patterns; they do not reason.**

When a model "explains" a math problem step-by-step, it is *not* thinking through the logic. It is generating a "reasoning-like" sequence of text because it has been trained on millions of examples of math problems that *look like that*.

You are not teaching the model to "think." You are teaching it to reproduce specific, high-quality patterns.

## How Fine-Tuning Builds on This

Fine-tuning **does not** change the model's core algorithm. The model is still just predicting the next token.

The only things that change are the **data** and the **goal**:
* **Pretraining:** The model learns general language patterns from a massive, raw, and diverse dataset (e.g., the whole internet).
* **Fine-Tuning:** We continue the *exact same process* but on a smaller, curated, and specialized dataset (e.g., instruction-response pairs, medical notes, or chat dialogues).

Fine-tuning is just "training again," but with a more focused dataset to teach the model new, specific patterns.

---

## Acknowledgements

These notes are based on the "LLM Fine-Tuning Foundations: Understanding Next-Token Prediction" lesson from the **LLM Engineering & Deployment Certification Program** by **Ready Tensor**.