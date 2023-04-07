import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-6.7b",
    load_in_8bit=True,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b")

# 虽说显式声明了四种特殊字符，但tokenizer
tokenizer.pad_token_id = 1
tokenizer.unk_token_id = 3
tokenizer.bos_token_id = 0  # 模型中这里同样设定为2
tokenizer.eos_token_id = 2

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

print_trainable_parameters(model)

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

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
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        warmup_steps=20,
        num_train_epochs=3,
        max_steps=150,
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