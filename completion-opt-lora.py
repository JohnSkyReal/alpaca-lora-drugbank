import os
import xlrd
import json
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="DrugBank_test.xls")
parser.add_argument("--output_path", type=str, default="prediction_drugbank_opt-lora-drug-ft10-gpt3_form")
parser.add_argument("--model_path", type=str, default="facebook/opt-6.7b")
parser.add_argument("--lora_path", type=str, default="opt-lora-ft10-gpt3_form")
parser.add_argument("--has_input", type=int, default=0)
args = parser.parse_args()

# 加载model与tokenizer
peft_model_id = args.lora_path
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)


# dump:写入json文件
def dump_json(path, data):
    with open(path, 'w', encoding='utf-8')as f:
        json.dump(data, f)


def readxls(path):
    data = []
    xl = xlrd.open_workbook(path)
    sheet = xl.sheets()[0]  # 0 表示读取第一个工作表
    for i in range(sheet.nrows):
        data.append(list(sheet.row_values(i)))
    # print(data)
    return data


def auto_completion(text, has_input):
    if has_input == 1:
        text = text.split('\n\n###\n\n')[0].strip()
        input_text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Extract drug entities and their relationships from the following biomedical text.

### Input:
{text}

### Response:"""
    else:
        input_text = text

    batch = tokenizer(input_text, return_tensors="pt")
    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**batch, max_new_tokens=256)
        response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        print(response)
        print('-' * 20)
    return response


def extract_prompt(lines):
    all_pairs = []
    for line in lines:
        pair = {
            'prompt': line[0],
            'label': line[1],
            'response': ''
        }
        all_pairs.append(pair)
    return all_pairs


def main():
    save_path = args.output_path
    has_input = args.has_input  # 1:有，0:没有
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    path = args.data_path
    lines = readxls(path)
    all_pairs = extract_prompt(lines[1:])
    print(len(all_pairs))
    i = 1
    for pair in all_pairs:
        prompt = pair['prompt']
        response = auto_completion(prompt, has_input)
        pair['response'] = response
        json_path = save_path + r'/response_{}.json'.format(i)
        dump_json(json_path, pair)
        i += 1
        # time.sleep(2)


if __name__ == '__main__':
    main()
