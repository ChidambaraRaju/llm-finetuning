# Lesson Notes: Language Model Architectures

This lesson provides a mental model for the three core architectures of Large Language Models (LLMs). It explains the foundation they all share and details why one specific architecture has come to dominate modern AI.

## The Foundation: The Transformer

All modern language models are built upon the **Transformer** architecture, which was introduced by Google in the 2017 paper "Attention Is All You Need".

Its revolutionary feature was the **attention mechanism**. Unlike older models that processed words in sequence, attention allows the model to consider all tokens in a sentence at once and understand their relationships. This is how a model can determine if the word "bank" refers to a river or a financial institution.

The original Transformer design had two main parts:
* **Encoder:** Reads and understands the entire input text, looking both forwards and backwards to build context.
* **Decoder:** Generates the output text one word at a time, using the encoder's understanding and what it has already written.

This full design was originally created for machine translation (e.g., translating a sentence from English to French).

## The Three Transformer Architectures

Researchers discovered that for different tasks, you don't always need both parts. This led to three main families of models.

### 1. Encoder-Decoder Models (The Translator)

This is the original, full Transformer architecture. It uses the encoder to understand an input sequence and the decoder to transform it into a new output sequence.

* **Best For:** Sequence-to-sequence ($X \rightarrow Y$) tasks like translation, summarization, and paraphrasing.
* **Model Examples:** T5, BART, PEGASUS.

### 2. Encoder-Only Models (The Analyst)

These models use *only* the encoder stack. They are designed to look at the full input simultaneously (bidirectionally) to build a deep understanding.

* **Key Feature:** They are excellent at understanding and labeling text but *cannot* generate fluent new text.
* **Best For:** Classification, Named Entity Recognition (NER), topic detection, semantic similarity, and generating embeddings.
* **Model Examples:** BERT, RoBERTa, ALBERT, and DistilBERT.
* **Key Limitation:** You cannot fine-tune an encoder-only model like BERT into a chatbot.

### 3. Decoder-Only Models (The Author)

This is the **dominant architecture for modern LLMs**. These models generate text one token at a time, from left to right.

* **Key Feature:** These models are **autoregressive**. This means they predict the very next word based on all the words that came before it. Each new word they generate becomes part of the input for predicting the *next* word.
* **Best For:** Chat assistants, code generation, creative writing, email drafting, and general instruction-following.
* **Model Examples:**
    * GPT-3.5, GPT-4 (OpenAI)
    * Claude 3 (Anthropic)
    * Gemini (Google)
    * LLaMA 2, LLaMA 3 (Meta)
    * Mistral, Mixtral

This program focuses entirely on Decoder-Only models, as they are the standard for nearly all modern, real-world AI applications.

> **A Note on Chatbots:**
> Products like ChatGPT or Claude are *full systems* built on top of a core Decoder-Only model. These products include other components like memory, moderation layers, retrieval tools, and guardrails. This program focuses on fine-tuning and deploying the core language model layer itself.

## Architecture Comparison

| Aspect | Encoder-Decoder | Encoder-Only | Decoder-Only (GPT-Style) |
| :--- | :--- | :--- | :--- |
| **Text Processing** | Encoder is bidirectional, Decoder is unidirectional | Bidirectional (sees full context) | Unidirectional (left-to-right) |
| **Primary Strength**| Sequence transformation | Understanding & classification | Generation & conversation |
| **Best For** | Translation, summarization | Classification, NER, embeddings | Chat, code, creative writing |
| **Can Generate Text?**| Yes, but focused on transformation | No (or poorly) | Yes, fluently |
| **Examples** | T5, BART | BERT, RoBERTa | GPT, LLaMA, Claude, Mistral |
| **This Program** | Not covered | Not covered | **Primary focus** |


## Why Decoder-Only Models Dominate

The entire AI industry has shifted to Decoder-Only models for three main reasons:

1.  **Versatility:** A single model can act as a chatbot, code generator, translator, summarizer, and reasoning engine. It can handle tasks that once required specialized encoder or encoder-decoder models.
2.  **Scalability:** These models scale predictably. As model size and training data increase, their performance reliably improves.
3.  **Ecosystem Maturity:** The entire open-source ecosystem—from Hugging Face and PEFT to DeepSpeed and Axolotl—is consolidated around fine-tuning and serving decoder-only models.

When people say "LLM" today, they are almost always referring to a Decoder-Only model.

---

## Acknowledgements

These notes are based on the "Language Model Architectures" lesson from the **LLM Engineering & Deployment Certification Program** by **Ready Tensor**.