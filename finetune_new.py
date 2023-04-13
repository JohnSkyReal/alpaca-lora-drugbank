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
import warnings


parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true", default=False) # 如果想激活wandb，就传入--wandb即可，不用添加数值
parser.add_argument("--data_path", type=str, default="merge.json")
parser.add_argument("--test_path", type=str, default="merge.json")
parser.add_argument("--output_path", type=str, default="lora-alpaca")
parser.add_argument("--model_path", type=str, default="decapoda-research/llama-7b-hf")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--micro_batch_size", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--max_length", type=int, default=256)
# parser.add_argument("--warmup_steps", type=int, default=20)  # 换用下面 warmup_ratio
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--logging_steps", type=int, default=1)
parser.add_argument("--eval_steps", type=int, default=20)
parser.add_argument("--save_steps", type=int, default=20)
parser.add_argument("--save_total_limit", type=int, default=30)
parser.add_argument("--do_test", type=int, default=0)   # 0: 不用测试集，1：用测试集
parser.add_argument("--resume_from_checkpoint", type=str, default=None)  # 暂停使用
parser.add_argument("--ignore_data_skip", type=str, default="False")  # False为从断点处数据继续训练，True为从头开始  # 暂停使用
args = parser.parse_args()

if not args.wandb:
    os.environ["WANDB_MODE"] = "disabled"

# Setting for A100 - For 3090
MICRO_BATCH_SIZE = args.micro_batch_size  # change to 4 for 3090
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = args.epochs  # paper uses 3
LEARNING_RATE = 3e-4  # from the original paper
CUTOFF_LEN = args.max_length  # 256 accounts for about 96% of the data
LORA_R = 8  # 8
LORA_ALPHA = 16  # 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = args.do_test #2000

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

model = LlamaForCausalLM.from_pretrained(
    args.model_path,
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
    args.model_path, add_eos_token=True
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

if args.resume_from_checkpoint:
# Check the available weights and load them
    checkpoint_name = os.path.join(args.resume_from_checkpoint, "pytorch_model.bin")  # Full checkpoint
    if not os.path.exists(checkpoint_name):
        pytorch_bin_path = checkpoint_name
        checkpoint_name = os.path.join(args.resume_from_checkpoint, "adapter_model.bin")  # only LoRA model - LoRA config above has to fit
        if os.path.exists(checkpoint_name):
            os.rename(checkpoint_name, pytorch_bin_path)
            warnings.warn("The file name of the lora checkpoint 'adapter_model.bin' is replaced with 'pytorch_model.bin'")
        else:
            args.resume_from_checkpoint = None  # So the trainer won't try loading its state
            warnings.warn(f"Checkpoint {checkpoint_name} not found, now training from scratch!")
    # The two files above have a different name depending on how they were saved, but are actually the same.
    if os.path.exists(checkpoint_name):  # 找到有该checkpoint目录
        print(f"Restarting from {checkpoint_name}")
        adapters_weights = torch.load(checkpoint_name)
        model = set_peft_model_state_dict(model, adapters_weights)
    else:
        print(f"Checkpoint {checkpoint_name} not found")
    print("checkpoint_name", checkpoint_name)


data = load_dataset("json", data_files=DATA_PATH)

# print('following is model structure -----------------')

# print(model)
model.print_trainable_parameters()


# def generate_prompt(data_point):
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
# {data_point["output"]}"""
#     else:
#         return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
#
# ### Instruction:
# {data_point["instruction"]}
#
# ### Response:
# {data_point["output"]}"""

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
{data_point["output"]}"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""
    else:
        return data_point["prompt"] + data_point["completion"]

# print(data)

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
    eval_dataset=test_data["train"] if test_data else None,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        # warmup_steps=args.warmup_steps,
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
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

model.save_pretrained(OUTPUT_DIR)