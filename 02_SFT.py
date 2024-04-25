
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig
from transformers import TrainingArguments
from datasets import Dataset, DatasetDict
import os


############################################################################################################
# Input parameters
############################################################################################################
dataset_name = "garg-aayush/ultrachat-refined-100K-2048"
debug = False
dataset_text_field = "text"
debug_size = 1000
max_token_length = 2048

model_id = "NousResearch/Llama-2-7b-hf"

wandb_project_name = "llama2-7b-sft"
output_dir = "../results/train-llama2-7b-check"

attention_type = "flash_attention_2"

bits_and_bytes_config = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.float16,
    "bnb_4bit_use_double_quant": True,
}


training_args_dict = {
    "bf16": True,                   # specify bf16=True instead when training on GPUs that support bf16
    "do_eval": True,                # set to True to evaluate the model on the evaluation dataset
    "evaluation_strategy": "steps", # evaluate the model every epoch/steps
    "eval_steps": 200,                #
    "gradient_accumulation_steps": 32,  # number of gradient accumulation steps
    "gradient_checkpointing": True,     # set to True to use gradient checkpointing
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "learning_rate": 5.0e-05,
    "log_level": "info",
    "logging_steps": 1,             # log every 5 steps
    "logging_strategy": "steps",    
    "lr_scheduler_type": "cosine",  # set the learning rate scheduler to cosine decay
    "max_steps": -1,                # maximum number of training steps
    "num_train_epochs": 1,          # number of training epochs
    "output_dir": output_dir,       # path where the Trainer will save its checkpoints and logs
    "overwrite_output_dir": True,   # overwrite the content of the output directory
    "per_device_eval_batch_size": 4, # originally set to 8
    "per_device_train_batch_size": 2, # originally set to 8
    "push_to_hub": False,
    "hub_model_id": "llama2-7b-sft-qlora",
    "hub_strategy": "every_save",
    "report_to": "wandb",
    "save_strategy": "no",
    "save_total_limit": None,
    "seed": 100,
}

peft_config_dict = { "r":16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
}

############################################################################################################
# Load dataset
############################################################################################################
dataset = load_dataset(dataset_name)
print(dataset)

# select 100 examples (if debug)
debug = False
if debug:
  dataset = DatasetDict({
      "train": dataset["train"].select(range(debug_size)),
      "eval": dataset["eval"].select(range(debug_size)),
    })
  print(f"Debug mode: dataset has been reduced to {debug_size} examples")
  dataset

# print the first example
example = dataset["train"][0]
for k, v in example.items():
  print(f"{k}: {v}")


# split the dataset into training and evaluation sets
train_dataset = dataset["train"]
eval_dataset = dataset["eval"]


# print the number of examples in the training and evaluation sets
print("Train Dataset:\n", train_dataset)
print("Eval Dataset:\n", eval_dataset)


############################################################################################################
# Load tokenizer
############################################################################################################
tokenizer = AutoTokenizer.from_pretrained(model_id)

# set pad_token_id equal to the eos_token_id if not set
if tokenizer.pad_token_id is None:
  print("Setting pad_token_id to eos_token_id")
  tokenizer.pad_token_id = tokenizer.eos_token_id

# Set reasonable default for models without max length
if tokenizer.model_max_length > max_token_length:
  print("Setting model_max_length to max_token_length")
  tokenizer.model_max_length = max_token_length


############################################################################################################
# Define model configurations
############################################################################################################
# set wandb project name
os.environ["WANDB_PROJECT"] = wandb_project_name


# Define quantization configuration
quantization_config = BitsAndBytesConfig(**bits_and_bytes_config)
print("\nQuantization configuration:")
for k, v in bits_and_bytes_config.items():
    print(f"{k}: {v}")


# Device map
device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None
print(f"device_map: {device_map}")


# Define model configuration
model_kwargs = dict(
    attn_implementation=attention_type, # set this to True if your GPU supports it (Flash Attention drastically speeds up model computations)
    torch_dtype="auto",
    use_cache=False, # set to False as we're going to use gradient checkpointing
    device_map=device_map,
    quantization_config=quantization_config,
)
print("\nModel configuration:")
for k, v in model_kwargs.items():
  print(f"{k}: {v}")


# training arguments
print("\nTraining arguments:")
for k, v in training_args_dict.items():
  print(f"{k}: {v}")
training_args = TrainingArguments(**training_args_dict)


# peft configuration
print("\nPEFT configuration:")
for k, v in peft_config_dict.items():
  print(f"{k}: {v}")
peft_config = LoraConfig(**peft_config_dict)


############################################################################################################
# Set SFT Trainer
############################################################################################################
trainer = SFTTrainer(
        model=model_id,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field=dataset_text_field,
        tokenizer=tokenizer,
        packing=True,
        peft_config=peft_config,
        max_seq_length=tokenizer.model_max_length,
    )


############################################################################################################
# Train the model
############################################################################################################
train_result = trainer.train()



############################################################################################################
# Save metrics and model
############################################################################################################
metrics = train_result.metrics
metrics["train_samples"] =len(train_dataset)
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

# Save trained model
trainer.model.save_pretrained(output_dir)

