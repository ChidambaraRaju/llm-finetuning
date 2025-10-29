# Lesson Notes: Reproducing Hugging Face Leaderboard Benchmarks

This lesson provides a hands-on exercise to move from *interpreting* leaderboards to *reproducing* their benchmark scores. This is a critical skill for verifying published results and evaluating your own custom models.

The exercise uses the **`lm-evaluation-harness`**, which is the same open-source tool that Hugging Face uses to generate its Open LLM Leaderboard scores. This ensures the results are transparent, consistent, and reproducible.

* **Benchmark for this Exercise:** `tinyGSM8K`, a lightweight benchmark that measures math/reasoning capabilities.

---

## Step 1: Set Up Your Colab Environment

First, open a Google Colab notebook, set the runtime to GPU (`Runtime > Change runtime type > GPU`), and install the necessary libraries.

```bash
# Install the core evaluation framework and language detection
!pip install lm_eval langdetect -q

# Install the tinyBenchmarks package
!pip install git+[https://github.com/felipemaiapolo/tinyBenchmarks](https://github.com/felipemaiapolo/tinyBenchmarks)
```

You can verify the installation by checking the help command:
```bash
!lm_eval --help
```

## Step 2: Choose a Model and Task

For this exercise, we'll use a small, fast instruct-tuned model and the lightweight math benchmark.

* **Model:** `meta-llama/Llama-3.2-1B-Instruct`
* **Task:** `tinyGSM8K`

## Step 3: Run Your First Evaluation (from the Command Line)

The simplest way to run an evaluation is with the CLI.

```bash
!lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct \
    --tasks tinyGSM8K \
    --device auto \
    --batch_size auto
```
This command will:
1.  Download the model from Hugging Face.
2.  Load it onto the GPU.
3.  Run the `tinyGSM8K` benchmark using official prompts.
4.  Print the final accuracy and metrics, which should closely match the public leaderboard.

## Step 4: Run Evaluations (from Python)

For better integration into a workflow (like after a fine-tuning run), you can use the Python API. This allows you to save and track results.

```python
from lm_eval import evaluator
from joblib import dump

# Run the evaluation
results = evaluator.simple_evaluate(
    model="hf",
    model_args="pretrained=meta-llama/Llama-3.2-1B-Instruct,parallelize=True,trust_remote_code=True",
    tasks=["tinyGSM8K"],
    device="cuda",
    batch_size="auto"
)

# Print and save the results
print(results)
dump(results, "results.joblib")
```

This approach lets you save structured results (`results.joblib`) for later comparison and build automated performance tracking into your projects.

## Step 5: Interpret Your Results

The output will be a dictionary containing the scores.

**Example Result:**
```json
{
  "results": {
    "tinyGSM8k": {
      "exact_match,strict-match": 0.390149...,
      "exact_match_stderr,strict-match": "N/A",
      "exact_match,flexible-extract": 0.390149...,
      "exact_match_stderr,flexible-extract": "N/A"
    }
  }
}
```

* **Accuracy:** The model scored approximately **39%** on `tinyGSM8k`.
* **Metrics:**
    * `strict-match`: Counts only exact matches.
    * `flexible-extract`: Allows for small formatting differences.
    * In this case, both scores are the same.
* **`stderr` (Standard Error):** This is 'N/A' by default. You can enable it by adding the flag `--bootstrap_iters=1000`. This statistical method estimates how much the score might vary by re-running the test on different samples, giving you a confidence interval.


## Week 1 Complete!

You have now completed Week 1. You've learned how to:
* Understand LLM architectures (Encoder, Decoder, etc.) and ecosystems (Frontier vs. Open-Weight).
* Decide when to use Fine-Tuning vs. RAG.
* Read and interpret model leaderboards (Hugging Face, Chatbot Arena).
* Run an evaluation yourself to reproduce a benchmark score.

Next week, the course will cover the building blocks of fine-tuning: tokenization, dataset preparation, LoRA, and QLoRA.

## Acknowledgements

These notes are based on the "Evaluating LLMs: Reproducing Hugging Face Leaderboard Benchmarks" lesson from the **LLM Engineering & Deployment Certification Program** by **Ready Tensor**.