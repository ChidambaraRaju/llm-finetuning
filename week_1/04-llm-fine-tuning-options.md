# Lesson Notes: LLM Fine-Tuning Options

This lesson covers the three fundamental decisions every engineer must make *before* starting a fine-Tuning project. These choices define your workflow, cost, control, and scalability.

## The Three Decisions of Fine-Tuning

Every fine-tuning project is shaped by three key layers:

1.  **Model Access:** Are you using a provider's API or running the model yourself?
2.  **Compute Environment:** Will you train on your own machine or in the cloud?
3.  **Orchestration:** Will you write custom code or use a managed framework?

---

### Decision Layer 1: Model Access

This is the first and most important choice: what *kind* of model are you fine tuning?

#### Frontier Models (API-Based)

These are closed-source models (like GPT-4, Claude) where fine-tuning is offered as a managed service.

* **How it works:** You upload a dataset (usually a `.jsonl` file) to the provider's API. They handle the entire training process "behind the scenes" and give you a new endpoint to call. You never see or handle the model weights.
* **Pros:** Simple, scalable, and requires no infrastructure management.
* **Cons:**
    * You cannot inspect or modify the model.
    * You cannot reproduce or audit the training process.
    * You pay per request, often at a higher rate.

#### Open-Weight Models (Full Control)

These are the models you can download and run yourself (like LLaMA 3, Mistral, Phi-3).

* **How it works:** You have full control over the model weights. You can train them locally or on rented GPUs, modify the architecture, and manage the entire process.
* **Pros:**
    * Full transparency and independence.
    * Deep customization is possible.
* **Cons:** You are responsible for managing compute, tracking experiments, and ensuring reproducibility.

**This certification focuses on open-weight models** because they are essential for transparent, customizable, and independent AI engineering.

---

### Decision Layer 2: Compute Environment

Once you have a model, you decide *where* to run the training.

#### Local Training

This means training on your own workstation, laptop (with a GPU), or internal company servers.

* **Pros:** Full control and privacy; fast iteration cycles for small experiments.
* **Cons:** Limited by your own GPU capacity.
* **Use Case:** Ideal for initial experimentation and fine-tuning smaller models.

#### Cloud Training

This means renting GPU resources on demand from platforms like AWS EC2, RunPod, Vast.ai, or Google Colab Pro.

* **Pros:** Access to powerful, scalable hardware (e.g., A100s, H100s) that you don't have to own.
* **Cons:** Incurs costs based on usage.
* **Use Case:** The standard for most real-world projects, allowing you to start small and scale up as needed.

---

### Decision Layer 3: Orchestration

This is the *how* of fine-tuning. How will you manage the training process itself?

#### Custom Code Approach

Here, you work directly with foundational Python libraries like Hugging Face `transformers`, `peft` (Parameter-Efficient Fine-Tuning), `trl` (Transformer Reinforcement Learning), and `accelerate`.

* **How it works:** You write the Python scripts to load the model, configure parameters (like LoRA), and run the training loop.
* **Pros:** Gives you fine-grained control over every single parameter and allows you to integrate new techniques (like QLoRA) immediately.
* **Use Case:** Best for research, experimentation, and when you need maximum flexibility.

**Example (Custom Code with `trl`):**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from peft import LoraConfig

# 1. Load Model and Tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. Define LoRA Config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)

# 3. Set up Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=my_dataset,
    peft_config=lora_config,
    max_seq_length=2048,
)

# 4. Train
trainer.train()
```

#### Managed Framework Approach

Here, you use a higher-level framework that abstracts away the boilerplate code. You define your training in a configuration file (like YAML) and let the framework handle the orchestration.

* **Popular Frameworks:** Axolotl, AWS SageMaker, Together.ai.
* **Pros:** Focuses on data and configuration, not code. It's built for reliability, consistent results, and automated scaling.
* **Use Case:** Perfect for enterprise or production workflows.

**Example (Managed Framework with `Axolotl`):**
```yaml
# Axolotl configuration YAML
base_model: meta-llama/Llama-2-7b-hf
datasets:
  - path: /support_data.jsonl
    type: completion

adapter: lora
lora_r: 32
lora_alpha: 16

sequence_len: 2048
micro_batch_size: 2
num_epochs: 3
```

---

## Choosing Your Workflow

Your choice depends on your goals:

* **For Simplicity & Speed:** Use **Frontier (API)** models.
* **For Transparency & Control:** Use **Open-Weight** models on **Local** compute.
* **For Scale & Performance:** Use **Open-Weight** models on **Cloud** compute.
* **For Flexibility & Experimentation:** Use the **Custom Code** approach.
* **For Reliability & Automation:** Use a **Managed Framework**.

Many teams use a hybrid approach: experimenting with custom code locally, then scaling up in the cloud using a managed framework.

---

## Acknowledgements

These notes are based on the "LLM Fine-Tuning Options" lesson from the **LLM Engineering & Deployment Certification Program** by **Ready Tensor**.