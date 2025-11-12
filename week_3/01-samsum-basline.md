# Lesson Notes: SAMSum Fine-Tuning Project Baseline

This lesson kicks off the first hands-on project of the course: fine-tuning a model for dialogue summarization. The first step in any fine-tuning experiment is to establish a **baseline**—a "before" snapshot of the model's performance on the task *before* any training.

This baseline score will be the reference point we try to beat in the upcoming lessons.


## The Project: Dialogue Summarization

* **Task:** Dialogue Summarization. The model will be given a chat conversation (a dialogue) and must generate a concise, factual summary.
* **Dataset:** **SAMSum**, a well-known benchmark dataset of real-world chat conversations paired with human-written summaries.
* **Metric:** **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation). This is the standard metric for summarization. It measures the overlap between the model's generated summary and the "ground truth" human summary.

### What is ROUGE?
* **ROUGE-1:** Measures the overlap of individual words (unigrams).
* **ROUGE-2:** Measures the overlap of word pairs (bigrams).
* **ROUGE-L:** Measures the longest common subsequence, reflecting sentence structure and order.
* **In short:** Higher ROUGE scores mean the generated summary is closer to the human-written one.

### The Base Model
For this project, we'll use **Llama 3.2 1B Instruct**.
* **Why?** It's small enough to run quickly in a Google Colab environment, but advanced enough to understand instructions and show measurable improvement from fine-tuning.

---

## Running the Baseline Evaluation

We will now walk through the 5 steps to get the baseline ROUGE score for the Llama 3.2 1B model on the SAMSum dataset.

### Step 1: Dataset Configuration

We define our dataset parameters in a `config.yaml` file to ensure reproducibility. We will only use 200 samples from each split (train, validation, test) to keep the evaluation fast.

```yaml
dataset:
  name: knkarthick/samsum
  cache_dir: ../data/datasets
  field_map:
    input: dialogue
    output: summary
  type: completion
  splits:
    train: 200
    validation: 200
    test: 200
  seed: 42
```

### Step 2: Load and Prepare the Dataset

We load the dataset from Hugging Face and select our 200-sample subset from the validation split.

```python
from datasets import load_dataset

def load_and_prepare_dataset(cfg):
    # ... (code to load based on config)
    dataset = load_dataset(cfg_dataset["name"])
    val_key = "validation" if "validation" in dataset else "val"
    
    # ... (select subsets)
    val = select_subset(dataset[val_key], cfg_dataset["splits"]["validation"], seed=42)
    
    print(f"Loaded {len(val)} val samples.")
    return val
```

### Step 3: Load the Model and Tokenizer

We load the Llama 3.2 1B model using `bfloat16` for memory efficiency. We also set the `pad_token` to be the same as the `eos_token`, a common practice.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def setup_model_and_tokenizer(cfg, use_4bit=False, use_lora=False):
    model_name = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token # Set pad token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    
    print("Using base model only (no LoRA).")
    return model, tokenizer
```

### Step 4: Generate Predictions

We loop through our validation dataset and use the model to generate a summary for each dialogue. We use a specific **prompt template** to instruct the model.

* **Task Instruction / Prompt:**
    ```
    You are a helpful assistant who writes concise, factual summaries of conversations.
    Summarize the following conversation into a single sentence.

    ## Dialogue:
    [Dialogue text goes here]
    ## Summary:
    ```

* **Generation Code:**
    ```python
    from transformers import pipeline
    from tqdm import tqdm

    def generate_predictions(model, tokenizer, dataset, task_instruction, batch_size=8):
        prompts = []
        for sample in dataset:
            user_prompt = (
                f"{task_instruction}\n\n"
                f"## Dialogue: \n{sample['dialogue']}\n## Summary:"
            )
            messages = [{"role": "user", "content": user_prompt}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        # Use a pipeline for efficient batch generation
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, do_sample=False)
        preds = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating summaries"):
            batch = prompts[i : i + batch_size]
            outputs = pipe(batch, max_new_tokens=256, return_full_text=False)
            preds.extend([o[0]["generated_text"].strip() for o in outputs])
        
        return preds
    ```

### Step 5: Compute ROUGE Scores

Finally, we use the `evaluate` library to compare our model's predictions against the human reference summaries.

```python
import evaluate

def compute_rouge(predictions, samples):
    rouge = evaluate.load("rouge")
    references = [s["summary"] for s in samples]
    return rouge.compute(predictions=predictions, references=references)
```

---

## Baseline Results

After running the full evaluation, the Llama 3.2 1B Instruct model **(with no fine-tuning)** achieved the following scores on the SAMSum validation set:

| Scenario | ROUGE-1 | ROUGE-2 | ROUGE-L |
| :--- | :--- | :--- | :--- |
| **Llama 3.2 1B Instruct Baseline** | **35.1%** | **13.0%** | **27.2%** |

This is our "before" snapshot. These scores represent the model's out-of-the-box performance. The goal of our fine-tuning in the next lessons will be to improve these numbers significantly.

---

## Acknowledgements

These notes are based on the "SAMSum Fine-Tuning Project: Establishing Your Baseline Performance" lesson from the **LLM Engineering & Deployment Certification Program** by **Ready Tensor**.