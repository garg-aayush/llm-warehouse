# load relevant libraries
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer

# get or add the Huggingface token key (a)
HF_TOKEN = os.environ["HF_TOKEN"]


dataset_name = "garg-aayush/mini-platypus-1K"
dataset = load_dataset(dataset_name, split="train")

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, # compute datatype float16 is GPU architecture >= Ampere else float16
    bnb_4bit_use_double_quant=True, # even quantization parameters are quantized
)

# LoRA configuration
peft_config = LoraConfig(
    r=16,               # rank of the matrix
    lora_alpha=32,      # strength of adapter (weight): standard = 32
    lora_dropout=0.05,  # 5% dropout ability
    bias="none",        
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj'] 
    # the more module -> the more parameters --> better performance
)


# Model
base_model = "NousResearch/Llama-2-7b-hf"

device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    attn_implementation="flash_attention_2", # set this to True if your GPU supports it (Flash Attention drastically speeds up model computations)
    use_cache=False, # set to False as we're going to use gradient checkpointing
    quantization_config=bnb_config,
    device_map=device_map,
)


# Cast the layernorm in fp32
# make output embedding layer require grads, add the upcasting of the lmhead to fp32
# take some layers and use them in highest available precision, helps to build the better model
model = prepare_model_for_kbit_training(model)


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
# unknown token, padding token has effect on the generation process
tokenizer.pad_token = tokenizer.unk_token 
tokenizer.padding_side = "right" # Load base moodel



## 4. Set Training and SFT arguments

# Set training arguments
training_arguments = TrainingArguments(
        output_dir="./results",
        num_train_epochs=4,             # 3-5 epochs good for Llama-2 model
        per_device_train_batch_size=8, # batch size per device during training
        gradient_accumulation_steps=3,
        evaluation_strategy="steps",
        eval_steps=2000,
        logging_steps=1,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,             # QLORA and model impect the learning rate
        lr_scheduler_type="linear",
        warmup_steps=10,
        report_to="wandb",
        fp16=True,
        # max_steps=2,  # Remove this line for a real fine-tuning
        push_to_hub=True,
        hub_model_id="llama-2-7b-miniplatypus-1K",
        hub_strategy="every_save",
        hub_token=HF_TOKEN 
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="instruction",
    max_seq_length=2048, # as in colab, VRAM is quite low
    tokenizer=tokenizer,
    args=training_arguments,
)

## 5. Train the model


# Train model
trainer.train()


# ## 6. Save the model


# Save trained model
new_model = "llama-2-7b-miniplatypus-1K"
trainer.model.save_pretrained(new_model)

# push the model to hub
trainer.push_to_hub()


# ## 7. Infer the trained model
# Run text generation pipeline with the trained model
prompt = "What is a large language model?"
instruction = f"### Instruction:\n{prompt}\n\n### Response:\n"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=256)
result = pipe(instruction)
print(result[0]['generated_text'][len(instruction):])


