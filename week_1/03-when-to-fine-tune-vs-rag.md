# Lesson Notes: When to Fine-Tune or Use RAG

This lesson covers the practical question of *how* to customize a Large Language Model (LLM) for a specific task. It breaks down the three primary methods available and explains how to choose the right one.

## The Three Ways to Adapt an LLM

When customizing an LLM, you have three main tools. They are best understood as layers that build on each other:

1.  **Prompt Engineering:** Guiding the model with instructions and examples. This *shapes the output*.
2.  **RAG (Retrieval-Augmented Generation):** Giving the model access to external knowledge. This *expands what the model knows*.
3.  **Fine-Tuning:** Retraining the model on your own data. This *changes what the model is*.

---

### 1. Prompt Engineering: Start Here, Always

This is the simplest and fastest way to customize an LLM. You craft detailed instructions, provide examples (few-shot learning), or define formats to guide the model's response.

* **Example:** Providing a persona, rules, and output format for a customer support bot.
* **Why Start Here?**
    * **Cheap:** You only pay for the API usage (inference).
    * **Instant:** Results appear immediately.
    * **Iterative:** You can quickly test and tweak the prompt.
* **Limitation:** It's like giving an intern new instructions for every single task. You must re-supply the context in every prompt, which is inefficient for deep, specialized knowledge.

### 2. RAG: When the Model Needs to Know More

**Retrieval-Augmented Generation (RAG)** connects your LLM to an external knowledge source (e.g., your company's documents, a database) and feeds it the right information at runtime.

The model isn't *taught* new facts; it *reads* them dynamically to answer a question.

**How RAG Works (The Flow):**
1.  **User Asks:** A user asks a question (e.g., "What is our Pro Plan's warranty?").
2.  **System Searches:** The system searches your private data (using embeddings and a vector database) for relevant snippets.
3.  **Snippets Inserted:** The relevant snippets are pulled and inserted into the prompt given to the LLM.
4.  **LLM Answers:** The LLM reads this new context and crafts an answer grounded in your data.

* **When to use RAG:**
    * You need accurate, **up-to-date answers**.
    * Your data **changes frequently** (e.g., updating a policy doc is easier than retraining a model).
    * You need **data privacy** (the retrieval can be local).
    * You want **transparency** (the model can cite its sources).
* **Limitation:** RAG only feeds the model context; it doesn't make the model smarter or teach it new *behaviors* or *styles*.

### 3. Fine-Tuning: When You Need True Adaptation

Fine-tuning involves **teaching the model new behaviors** by training it on your own dataset of examples. This process updates the model's internal weights, making the new behavior permanent. It's the difference between "telling" (prompting) and "teaching" (fine-tuning).

* **When to use Fine-Tuning:**
    * You need **consistent, repeatable output** in a specific format or style.
    * You need the model to understand **domain-specific language** or jargon.
    * You can't rely on long prompts or external retrieval (e.g., you need the model to run **offline**).
    * You have a **high-quality dataset** of at least a few hundred examples.

#### ⚠️ Caution: Data Quality is Everything

Fine-tuning is powerful, but it has one major risk: **it makes your data's flaws permanent.**
* A model can't learn useful behavior from inconsistent, noisy, or poorly labeled data.
* If you train on flawed data, the model will confidently produce flawed outputs.
* **There is no "undo" button.** You cannot easily "patch" or "unlearn" bad training. You must start the entire fine-tuning process over with a clean dataset.
* **Fine-tuning amplifies what's already in your data.** Start with data worth amplifying.

---

## How to Choose: Three Scenarios

| Persona | Goal | Method | Why? |
| :--- | :--- | :--- | :--- |
| **Maya (Startup Founder)** | Automate professional emails & support replies. | **Prompt Engineering** | It's cost-effective, fast, and good enough. No GPUs needed. |
| **Leo (Enterprise Architect)**| Securely answer internal Q&A about company policies. | **RAG Pipeline** | It provides control & transparency. Data stays secure and can be updated easily. |
| **Aisha (Healthcare Researcher)** | Generate consistent, structured clinical summaries. | **LLM Fine-Tuning** | She needs high precision, compliance, and a specific structure that RAG/prompts can't guarantee. |

## How These Methods Work Together

In the real world, the best systems often **layer all three methods**:

* **Fine-Tuning:** A model is fine-tuned on a legal corpus to learn legal language.
* **RAG:** The fine-tuned model is connected via RAG to fetch specific, relevant case laws or clauses.
* **Prompting:** A final prompt defines the task and formats the output (e.g., "Draft a motion...").

## When NOT to Fine-Tune

Fine-tuning is often overused. Pause and reconsider if:
* You have **limited data** (less than a few hundred examples).
* Your task is generic (e.g., simple summarization).
* You have tight budget or time constraints.
* You can get **90% of the way there with good prompting and RAG.**

**Rule of Thumb:** If your data fits in the context window and your results are already good, fine-tuning probably isn't worth the cost.

---

## Acknowledgements

These notes are based on the "When to Fine-Tune or Use RAG" lesson from the **LLM Engineering & Deployment Certification Program** by **Ready Tensor**.