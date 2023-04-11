import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
import argparse
import warnings


parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true", default=False)  # 改了这里
# parser.add_argument("--wandb", action="store_true", default=True)
parser.add_argument("--data_path", type=str, default="DrugBank_train_prepared.jsonl")
parser.add_argument("--test_path", type=str, default="merge.json")
parser.add_argument("--output_path", type=str, default="opt-outputs")
parser.add_argument("--model_path", type=str, default="facebook/opt-6.7b")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--micro_batch_size", type=int, default=16)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--max_length", type=int, default=64)  # 改了这里
# parser.add_argument("--max_length", type=int, default=256)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--logging_steps", type=int, default=1)
parser.add_argument("--eval_steps", type=int, default=20)
parser.add_argument("--save_steps", type=int, default=20)
parser.add_argument("--save_total_limit", type=int, default=30)
parser.add_argument("--do_test", type=int, default=0)   # 0: 不用测试集，1：用测试集
parser.add_argument("--resume_from_checkpoint", type=str, default=None)
parser.add_argument("--ignore_data_skip", type=str, default="False")
args = parser.parse_args()

if not args.wandb:
    os.environ["WANDB_MODE"] = "disabled"

MICRO_BATCH_SIZE = args.micro_batch_size
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = args.epochs
LEARNING_RATE = 2e-4
CUTOFF_LEN = args.max_length
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
VAL_SET_SIZE = args.do_test

DATA_PATH = args.data_path
TEST_DATA_PATH = args.test_path
OUTPUT_DIR = args.output_path

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

print(args.model_path)


model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    load_in_8bit=True,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(args.model_path)

# # 虽说显式声明了四种特殊字符，但tokenizer
# tokenizer.pad_token_id = 1
# tokenizer.unk_token_id = 3
# tokenizer.bos_token_id = 0  # 模型中这里同样设定为2
# tokenizer.eos_token_id = 2

# print(tokenizer.unk_token, tokenizer.unk_token_id)
# print(tokenizer.bos_token, tokenizer.bos_token_id)
# print(tokenizer.eos_token, tokenizer.eos_token_id)
# print(tokenizer.pad_token, tokenizer.pad_token_id)

# model = prepare_model_for_int8_training(model)

for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)


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
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

print_trainable_parameters(model)

# 由于OPTtokenizer无法自动添加eos token，因此手动在text末尾添加eos token：</s>

data = load_dataset("json", data_files=DATA_PATH)  # 与GPT3 finetune格式相同

# '''
# Example:
# {"prompt":"Cocaine sometimes proves to be fatal when used in combination with heroin.\n\n###\n\n","completion":" <Cocaine, effect, heroin> END"}
# '''
#
# def generate_prompt(data_point):
#     text = data_point["prompt"] + data_point["completion"] + '</s>'
#     return text


def generate_prompt(data_point):
    if "instruction" in data_point.keys():
        # sorry about the formatting disaster gotta move fast
        if data_point["input"]:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}</s>"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}</s>"""
    else:
        return data_point["prompt"] + data_point["completion"] + '</s>'


data = data.shuffle().map(
    lambda data_point: tokenizer(
        generate_prompt(data_point),
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length",
    )
)

if VAL_SET_SIZE > 0:
    test_data = load_dataset("json", data_files=TEST_DATA_PATH)
    test_data = test_data.shuffle().map(
        lambda data_point: tokenizer(
            generate_prompt(data_point),
            truncation=True,
            max_length=CUTOFF_LEN,
            padding="max_length",
        )
    )
else:
    test_data = None

print("train data length: {}".format(len(data['train'])))
print("dev/test data length: {}".format(len(test_data['train']) if test_data else 0))
print(data)
print(test_data)
print(tokenizer.decode(data['train'][0]["input_ids"]))
print(tokenizer.decode(test_data['train'][0]["input_ids"]) if test_data else None)


trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    eval_dataset=test_data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
        save_strategy="steps",
        eval_steps=args.eval_steps if VAL_SET_SIZE > 0 else None,
        save_steps=args.save_steps,
        output_dir=OUTPUT_DIR,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        report_to="wandb" if args.wandb else [],
        ignore_data_skip=args.ignore_data_skip,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# old_state_dict = model.state_dict
# model.state_dict = (
#     lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
# ).__get__(model, type(model))
#
# if torch.__version__ >= "2" and sys.platform != "win32":
#     model = torch.compile(model)

trainer.train()
model.save_pretrained(OUTPUT_DIR)