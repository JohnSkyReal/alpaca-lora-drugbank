import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--data_path", type=str, default="merge.json")
parser.add_argument("--output_path", type=str, default="lora-alpaca")
parser.add_argument("--model_path", type=str, default="facebook/opt-6.7b")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--micro_batch_size", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--max_length", type=int, default=256)
parser.add_argument("--warmup_steps", type=int, default=20)
parser.add_argument("--logging_steps", type=int, default=1)
parser.add_argument("--eval_steps", type=int, default=200)
parser.add_argument("--save_steps", type=int, default=20)
parser.add_argument("--save_total_limit", type=int, default=3)
parser.add_argument("--test_size", type=int, default=0)   # 该参数暂停使用
parser.add_argument("--resume_from_checkpoint", type=str, default=None)
parser.add_argument("--ignore_data_skip", type=str, default="False")
args = parser.parse_args()

if not args.wandb:
    os.environ["WANDB_MODE"] = "disabled"

MICRO_BATCH_SIZE = args.micro_batch_size  # change to 4 for 3090
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = args.epochs  # paper uses 3
LEARNING_RATE = 3e-4  # from the original paper
CUTOFF_LEN = args.max_length  # 256 accounts for about 96% of the data
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = args.test_size #2000




model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-6.7b",
    load_in_8bit=True,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b")

# # 虽说显式声明了四种特殊字符，但tokenizer
# tokenizer.pad_token_id = 1
# tokenizer.unk_token_id = 3
# tokenizer.bos_token_id = 0  # 模型中这里同样设定为2
# tokenizer.eos_token_id = 2

# print(tokenizer.unk_token, tokenizer.unk_token_id)
# print(tokenizer.bos_token, tokenizer.bos_token_id)
# print(tokenizer.eos_token, tokenizer.eos_token_id)
# print(tokenizer.pad_token, tokenizer.pad_token_id)

model = prepare_model_for_int8_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

print_trainable_parameters(model)

# 由于OPTtokenizer无法自动添加eos token，因此手动在text末尾添加eos token：</s>

data = load_dataset("json", data_files="DrugBank_train_prepared.jsonl")  # 与GPT3 finetune格式相同

'''
Example:
{"prompt":"Cocaine sometimes proves to be fatal when used in combination with heroin.\n\n###\n\n","completion":" <Cocaine, effect, heroin> END"}
'''

def generate_prompt(data_point):
    text = data_point["prompt"] + data_point["completion"] + '</s>'
    return text
#     # sorry about the formatting disaster gotta move fast
#     if data_point["input"]:
#         return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
#
# ### Instruction:
# {data_point["instruction"]}
#
# ### Input:
# {data_point["input"]}
#
# ### Response:
# {data_point["output"]}</s>"""
#     else:
#         return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
#
# ### Instruction:
# {data_point["instruction"]}
#
# ### Response:
# {data_point["output"]}</s>"""

# print(data)

data = data.shuffle().map(
    lambda data_point: tokenizer(
        generate_prompt(data_point),
        truncation=True,
        max_length=256,
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
        warmup_ratio=0.1,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="opt-outputs",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
model.save_pretrained("opt-outputs")