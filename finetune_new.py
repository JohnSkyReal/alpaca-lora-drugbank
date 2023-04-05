import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM, LlamaTokenizer
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--data_path", type=str, default="merge.json")
parser.add_argument("--output_path", type=str, default="lora-alpaca")
parser.add_argument("--model_path", type=str, default="decapoda-research/llama-7b-hf")
args = parser.parse_args()

if not args.wandb:
    os.environ["WANDB_MODE"] = "disable"

# Setting for A100 - For 3090
MICRO_BATCH_SIZE = 8  # change to 4 for 3090
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 3  # paper uses 3
LEARNING_RATE = 3e-4  # from the original paper
CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

DATA_PATH = args.data_path
OUTPUT_DIR = args.output_path

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

model = LlamaForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=True,
    device_map=device_map,
)

# config = AutoConfig.from_pretrained("decapoda-research/llama-7b-hf")
# # 修改config中的vocab_size参数
# config.vocab_size = 32003

# # 根据修改后的config创建模型
# model = LlamaForCausalLM.from_pretrained(
#     "decapoda-research/llama-7b-hf",
#     load_in_8bit=True,
#     device_map="auto",
#     config=config
# )

tokenizer = LlamaTokenizer.from_pretrained(
    "decapoda-research/llama-7b-hf", add_eos_token=True
)

tokenizer.pad_token_id = 0
tokenizer.unk_token_id = 0
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2

# DEFAULT_PAD_TOKEN = "[PAD]"
# DEFAULT_EOS_TOKEN = "[EOS]"
# DEFAULT_BOS_TOKEN = "[BOS]"
# DEFAULT_UNK_TOKEN = "[UNK]"

# tokenizer.add_special_tokens({'pad_token': DEFAULT_PAD_TOKEN})
# tokenizer.add_special_tokens({'unk_token': DEFAULT_UNK_TOKEN})
# tokenizer.add_special_tokens({'bos_token': DEFAULT_BOS_TOKEN})
# tokenizer.add_special_tokens({'eos_token': DEFAULT_EOS_TOKEN})
model = prepare_model_for_int8_training(model)


config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
# tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
data = load_dataset("json", data_files=DATA_PATH)


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""

# print(data)

data = data.shuffle().map(
    lambda data_point: tokenizer(
        generate_prompt(data_point),
        truncation=True,
        max_length=CUTOFF_LEN+1,
        padding="max_length",
    )
)

print(tokenizer.decode(data['train'][100]["input_ids"]))

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=20,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=1,
        output_dir=OUTPUT_DIR,
        save_total_limit=3,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False

# old_state_dict = model.state_dict
# model.state_dict = (
#     lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
# ).__get__(model, type(model))

# if torch.__version__ >= "2" and sys.platform != "win32":
#     model = torch.compile(model)

trainer.train(resume_from_checkpoint=False)

model.save_pretrained(OUTPUT_DIR)