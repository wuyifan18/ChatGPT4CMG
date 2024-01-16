import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import tiktoken
from tqdm import tqdm

from utils import gpt_call, num_tokens_from_messages, jaccard_similarity, text_to_split

lan_list = ["java", "csharp", "cpp", "python", "javascript"]
prompt_template = "Generate a concise commit message that summarizes the content of code changes.\n{}code change:\n{}\ncommit message:"
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


def get_sample(lan):
    with open(f"dataset/MCMD/{lan}/test.jsonl", "r") as fr:
        data_samples = [json.loads(d) for d in fr]

    with open(f"dataset/MCMD/{lan}/train.jsonl", "r") as fr:
        corpus = [json.loads(d) for d in fr]
    corpus_sets = [set(text_to_split(doc["diff"])) for doc in corpus]
    print(f"corpus: {len(corpus)}")

    file_path = f"RQ2/{lan}_data.txt"
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    for d in tqdm(data_samples, total=len(data_samples)):
        top_similarities = jaccard_similarity(d["diff"], corpus_sets, 64)
        d["examples"] = [corpus[index] for index, _ in top_similarities]
        with open(file_path, 'a') as fw:
            fw.write(json.dumps(d) + '\n')


def gpt_predict(sample: dict, lan: str, number: int):
    prompt = prompt_template.format(sample["example"], sample["diff"].strip())
    result = gpt_call(prompt, model="gpt-3.5-turbo-1106", temperature=0, seed=0)
    pred = result["choices"][0]["message"]["content"]
    result = {"diff_id": sample["diff_id"], "repo": sample["repo"], "sha": sample["sha"], "ref": sample["msg"], "pred": pred.split("\n")[0]}
    file_path = f"RQ2/{lan}/{number}.txt"
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a") as fw:
        fw.write(json.dumps(result) + "\n")


if __name__ == '__main__':
    for lan in lan_list:
        if not os.path.exists(f"RQ2/{lan}_data.txt"):
            get_sample(lan)
        for number in [1, 2, 4, 8, 16, 32, 64]:
            print(lan, number)
            with open(f"RQ2/{lan}_data.txt") as fr:
                data_samples = [json.loads(d) for d in fr.readlines()]
                for d in tqdm(data_samples, total=len(data_samples)):
                    examples = d["examples"][-number:]
                    d["example"] = "".join([f"code change:\n{example['diff'].strip()}\ncommit message: {example['msg'].strip()}\n\n" for example in examples])
                    prompt = prompt_template.format(d["example"], d["diff"])
                    if num_tokens_from_messages(prompt) > 16384:
                        used_length = num_tokens_from_messages(
                            prompt_template + "".join([f"code change:\n\ncommit message: {example['msg']}\n\n" for example in examples]))
                        available_length = (16200 - used_length) // (number + 1)
                        d["diff"] = encoding.decode(encoding.encode(d["diff"], disallowed_special=())[:available_length])
                        for example in examples:
                            example['diff'] = encoding.decode(encoding.encode(example['diff'], disallowed_special=())[:available_length])
                        d["example"] = "".join([f"code change:\n{example['diff'].strip()}\ncommit message: {example['msg'].strip()}\n\n" for example in examples])
                        prompt = prompt_template.format(d["example"], d["diff"])
                    assert num_tokens_from_messages(prompt) <= 16384

            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                list(tqdm(executor.map(partial(gpt_predict, lan=lan, number=number), data_samples), total=len(data_samples)))
