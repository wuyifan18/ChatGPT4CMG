import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import tiktoken
from tqdm import tqdm

from utils import gpt_call, num_tokens_from_messages

lan_list = ["java", "csharp", "cpp", "python", "javascript"]
prompt_template = "Generate a concise commit message that summarizes the content of code changes.\n{}code change:\n{}\ncommit message:"
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


def get_sample(lan, number):
    random.seed(0)
    with open(f"dataset/MCMD/{lan}/test.jsonl", "r") as fr:
        data_samples = [json.loads(d) for d in fr]

    if number > 0:
        with open(f"dataset/MCMD/{lan}/train.jsonl", "r") as fr:
            corpus = [json.loads(d) for d in fr]
        print(f"corpus: {len(corpus)}")

        for d in tqdm(data_samples, total=len(data_samples)):
            d["example"] = ""
            examples = random.sample(corpus, number)

            d["example"] = "".join([f"code change:\n{example['diff']}\ncommit message: {example['msg']}\n\n" for example in examples])
            prompt = prompt_template.format(d["example"], d["diff"])
            if num_tokens_from_messages(prompt) > 16384:
                used_length = num_tokens_from_messages(
                    prompt_template + "".join([f"code change:\n\ncommit message: {example['msg']}\n\n" for example in examples]))
                available_length = (16200 - used_length) // (number + 1)
                d["diff"] = encoding.decode(encoding.encode(d["diff"], disallowed_special=())[:available_length])
                for example in examples:
                    example['diff'] = encoding.decode(encoding.encode(example['diff'], disallowed_special=())[:available_length])
                d["example"] = "".join([f"code change:\n{example['diff']}\ncommit message: {example['msg']}\n\n" for example in examples])
                prompt = prompt_template.format(d["example"], d["diff"])
            assert num_tokens_from_messages(prompt) <= 16384
    else:
        used_length = num_tokens_from_messages(prompt_template)
        for d in tqdm(data_samples, total=len(data_samples)):
            d["diff"] = encoding.decode(encoding.encode(d["diff"], disallowed_special=())[:(16384 - used_length)])
    return data_samples


def gpt_predict(sample: dict, lan: str, number: int):
    if number == 0:
        prompt = prompt_template.format("", sample["diff"])
    else:
        prompt = prompt_template.format(sample["example"], sample["diff"])
    result = gpt_call(prompt, model="gpt-3.5-turbo-1106", temperature=0, seed=0)
    pred = result["choices"][0]["message"]["content"]
    result = {"diff_id": sample["diff_id"], "repo": sample["repo"], "sha": sample["sha"], "ref": sample["msg"], "pred": pred.split("\n")[0]}
    file_path = f"RQ1/{lan}/{number}.txt"
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a") as fw:
        fw.write(json.dumps(result) + "\n")


if __name__ == '__main__':
    for lan in lan_list:
        for number in [0, 1, 2, 4, 8, 16, 32, 64]:
            print(lan, number)
            data_samples = get_sample(lan, number)
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                list(tqdm(executor.map(partial(gpt_predict, lan=lan, number=number), data_samples), total=len(data_samples)))
