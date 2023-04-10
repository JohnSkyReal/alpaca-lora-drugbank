import os
import xlrd
import json
import time
from peft import PeftModel
from transformers import AutoConfig, LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="DrugBank_test.xlsx")
parser.add_argument("--output_path", type=str, default="prediction_drugbank_eos-lora-alpaca-drug-ft3-no-ins")
parser.add_argument("--model_path", type=str, default="decapoda-research/llama-7b-hf")
parser.add_argument("--lora_path", type=str, default="eos-lora-alpaca-drug-ft10-no-ins")
parser.add_argument("--has_input", type=int, default=0)
args = parser.parse_args()

# 加载model与tokenizer
tokenizer = LlamaTokenizer.from_pretrained(args.model_path)

model = LlamaForCausalLM.from_pretrained(
    args.model_path,
    load_in_8bit=True,
    device_map="auto",
)

model = PeftModel.from_pretrained(model, args.lora_path)
# model = PeftModel.from_pretrained(model, "eos-lora-alpaca-drug-ft10-k_v")


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
        input_text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
Extract drug entities and their relationships from the following biomedical text.

### Input:
{text}

### Response:"""
    elif has_input == 0:
        input_text = f'''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{text}

### Response:'''
    else:
        input_text = text + '\n\n###\n\n'

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].cuda()

    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        repetition_penalty=1.2,
        top_k=40,
        do_sample=False,
        num_beams=4,

    )
    #     generation_config = GenerationConfig(
    #         temperature=0,
    #         top_p=1,
    #         top_k=50,
    #         num_beams=1

    #     )
    print("Generating...")

    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=256,
    )
    for s in generation_output.sequences:
        response = tokenizer.decode(s, skip_special_tokens=True)  # 新增参数，来自opt
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
        prompt = pair['prompt'].split('\n\n###\n\n')[0].strip()
        response = auto_completion(prompt, has_input)
        pair['response'] = response
        json_path = save_path + r'/response_{}.json'.format(i)
        dump_json(json_path, pair)
        i += 1
        # time.sleep(2)


if __name__ == '__main__':
    main()

# # 无input的情况
# input_text ='''Below is an instruction that describes a task. Write a response that appropriately completes the request.
#
# ### Instruction:
# What are Alpacas and how are they different to Lamas?
#
# ### Response:
# '''
#
#
# # 有input的情况
#
# input_text ="""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
# ### Instruction:
# Extract drug entities and their relationships from the following biomedical text.
#
# ### Input:
# The concomitant use of oxybutynin with other anticholinergic drugs or with other agents which produce dry mouth, constipation, somnolence (drowsiness), and/or other anticholinergic-like effects may increase the frequency and/or severity of such effects.
#
# ### Response:
# """


## 输出示例
'''
 ⁇  Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
For patients receiving ketoconazole or other potent CYP3A4 inhibitors such as other azole antifungals (eg, itraconazole, miconazole) or macrolide antibiotics (eg, erythromycin, clarithromycin) or cyclosporine or vinblastine, the recommended dose of DETROL LA is 2 mg daily.

### Response:
<ketoconazole, advise, DETROL LA>
<macrolide antibiotics, advise, DETROL LA>
<cyclosporine, advise, DETROL LA>
<vinblastine, advise, DETROL LA>

'''


'''
 ⁇  Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
Extract drug entities and their relationships from the following biomedical text.

### Input:
CYP3A4 Inhibitors: Ketoconazole, an inhibitor of the drug metabolizing enzyme CYP3A4, significantly increased plasma concentrations of tolterodine when coadministered to subjects who were poor metabolizers (see CLINICAL PHARMACOLOGY, Variability in Metabolism and Drug-Drug Interactions).

### Response:
Entities: Ketoconazole, tolterodine; Relationship: mechanism
Entities: CYP3A4, tolterodine; Relationship: mechanism

'''