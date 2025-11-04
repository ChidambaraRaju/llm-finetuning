# Lesson Notes: How LLMs Learn (Loss, Masking)

This lesson covers the core mechanics of *how* a Large Language Model (LLM) learns. The process involves three key concepts: **next-token prediction** (the task), **cross-entropy loss** (the measurement), and **masking** (the focus).

## The Learning Feedback Loop

Models start out "terrible" at predicting text, guessing tokens almost at random. They learn through a simple feedback loop:
1.  **Predict:** Guess the next token.
2.  **Compare:** Compare the guess to the *correct* token from the training data.
3.  **Measure:** Quantify how "wrong" the guess was.
4.  **Adjust:** Update the model's internal parameters (weights) to make a slightly better prediction next time.

This cycle of "prediction, comparison, and adjustment" is what turns a random model into a capable one.

## The Training Objective

The model's goal is to predict the next token given all previous tokens. During training, a single sentence is broken down into multiple training steps.

For the sequence: "The cat sat on the mat"

| Input Sequence | Target (Correct Next Token) |
| :--- | :--- |
| "The" | "cat" |
| "The cat" | "sat" |
| "The cat sat" | "on" |
| "The cat sat on" | "the" |
| "The cat sat on the" | "mat" |

At each step, the model outputs a probability distribution over its entire vocabulary (e.g., 50,000+ tokens). The goal is to assign the highest possible probability to the *correct* target token.

---

## Measuring Wrongness: Cross-Entropy Loss

The **loss function** is what measures the "wrongness" of a prediction. For LLMs, this is almost always **Cross-Entropy Loss**.

It calculates the penalty based on the probability the model assigned to the one *correct* token.

The formal equation $L = -\sum y_i \log(p_i)$ simplifies to:

$L = -\log(p_{correct})$

Where $p_{correct}$ is the model's predicted probability for the single true token.

**Example:**
The model sees "The cat sat on the..." and must predict the next token. The correct answer is "mat".

| Candidate Token | Predicted Probability |
| :--- | :--- |
| **mat** | **0.78** (Correct) |
| floor | 0.12 |
| chair | 0.05 |
| bed | 0.03 |
| wall | 0.02 |

The loss is calculated *only* on the probability for "mat":
* **Loss** = $-\log(0.78) \approx 0.25$

This small loss value (0.25) is the "training signal" that gets sent back into the model to update its weights.
* If the model had predicted 0.99 for "mat", the loss would be tiny ($-\log(0.99) \approx 0.01$).
* If it had predicted 0.01 for "mat", the loss would be huge ($-\log(0.01) \approx 4.61$).

### Learning from Sequences: Teacher Forcing

During training, the model does *not* use its own predictions to guess the next token (that's inference).

Instead, it uses **Teacher Forcing**: for every step, it predicts the next token based on the *true* preceding tokens from the training data. This ensures the model learns from the correct context rather than compounding its own errors.

The final loss for the sequence is simply the average of all the individual token losses.

---

## Masking: Deciding Which Tokens Matter

We don't always want the model to learn from every single token in a sequence. **Masking** is the mechanism that controls what the model "sees" and which tokens contribute to the loss.

There are two key types of masking:

### 1. Causal Masking (Base Models)

This is the standard for autoregressive (left-to-right) models. It ensures the model can't "see the future." When predicting the token at position 4 (e.g., "on"), the model can *only* use tokens at positions 1-3 ("The cat sat"). This is fundamental to how base models are pretrained.


### 2. Assistant-Only Masking (Chat Models)

This is the critical technique used in **Supervised Fine-Tuning (SFT)** to create chat models.

When training on a chat example, the model still uses *causal masking* (it can't see future tokens), but we **only compute loss on the assistant's tokens.**

**Example:**
`User: What's the capital of France?`
`Assistant: The capital of France is Paris.`

The model predicts *every* token, but we instruct the loss function to **ignore** all the tokens in the User's prompt ("What's", "the", "capital", "of", "France", "?").

The loss is calculated **only** for the tokens: "The", "capital", "of", "France", "is", "Paris", "."

This teaches the model *not* to predict the user's input, but to learn *how to generate the appropriate reply* given that input. This selective scoring is what turns a text-completion model into an instruction-following model.

---

## The Learning Loop Summarized

All LLM training follows this repeating cycle:
1.  **Take a batch of text.**
2.  **Predict** the next token at every position.
3.  **Compute the Loss** (Cross-Entropy) by comparing the prediction to the correct token.
4.  **Apply Masking** so only the desired tokens (e.g., the assistant's reply) count toward the loss.
5.  **Update Weights** to reduce the loss.
6.  **Repeat** billions of times.

This same "prediction $\rightarrow$ loss $\rightarrow$ masking $\rightarrow$ update" engine powers both pretraining and fine-tuning. The only difference is the dataset and the masking strategy.

---

## Acknowledgements

These notes are based on the "How LLMs Learn: Loss, Masking, and Next-Token Prediction" lesson from the **LLM Engineering & Deployment Certification Program** by **Ready Tensor**.