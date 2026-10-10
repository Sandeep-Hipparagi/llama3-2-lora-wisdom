# Llama 3.2 3B Instruct - Wisdom & Inner Engineering LoRA

This repository contains the fine-tuned low-rank adaptation (LoRA) weights and configuration files for the **Llama-3.2-3B-Instruct** model. The model has been optimized using **Unsloth** in 4-bit quantization on a Tesla T4 GPU to specialize in serving as a grounded, quiet mentor sharing timeless wisdom and "Inner Engineering" insights.

## 🌟 Project Overview

The goal of this project is to model a compassionate, philosophical companion capable of helping users navigate contemporary anxieties, career doubts, personal failures, relationship friction, and boundary issues by applying mindfulness, acceptance, and deep internal stability.

### Key Features
- **Base Model**: `unsloth/Llama-3.2-3B-Instruct` (Quantized to 4-bit to support low-VRAM runtime environments like Google Colab's Tesla T4).
- **Quantization & Optimization**: Configured with PEFT/LoRA (Rank: `16`, Alpha: `16`) targeting crucial projection layers to ensure conversational stability.
- **Loss Monitoring**: Utilizes early stopping monitored against a dedicated validation split to prevent stylistic over-fitting.
- **Chat Format**: Native `llama-3.2` chat template format integration.

---

## 🛠️ Environment Setup & Installation

Ensure you have the required dependencies installed to load and run inference on this LoRA adapter:

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.29" "trl<0.9.0" peft accelerate bitsandbytes
```

---

## 📊 Dataset Structure

The model was fine-tuned on a custom synthetic conversational dataset of highly structured Q&A formats matching the standardized **ShareGPT** format. 

### Sample Conversation Entry
```json
{
  "conversations": [
    {"role": "system", "content": "You are a quiet, grounded mentor sharing timeless wisdom."},
    {"role": "human", "content": "I feel constantly anxious about the future. How do I find inner peace?"},
    {"role": "gpt", "content": "Anxiety is the mind trying to live in a tomorrow that does not exist. Peace is not found by arranging the future to your liking, but by accepting this present breath..."}
  ]
}
```

---

## 🚀 Running Inference Locally

You can load the fine-tuned model and run inference with Unsloth directly using the snippet below:

```python
import torch
from unsloth import FastLanguageModel

max_seq_length = 2048
dtype = None
load_in_4bit = True

# 1. Load the model & tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Sandeep4235/llama3-2-lora-wisdom",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model)

# 2. Setup your message structure
messages = [
    {"role": "system", "content": "You are a quiet, grounded mentor sharing timeless wisdom."},
    {"role": "human", "content": "I feel like I'm wasting my twenties while everyone else is racing ahead."}
]

# 3. Apply Chat Template
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt"
).to("cuda")

# 4. Generate Output
outputs = model.generate(
    input_ids = inputs,
    max_new_tokens = 256,
    temperature = 0.7,
    top_p = 0.9,
    use_cache = True
)

print(tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True))
```

---

## 👥 Authors & Acknowledgments
- Developed and fine-tuned by **Sandeep Hipparagi**.
- Powered by **Unsloth** and **Hugging Face** library tools.
