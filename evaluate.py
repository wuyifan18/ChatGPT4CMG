#!/usr/bin/env python
# !-*-coding:utf-8 -*-
import json
import re
import sys

import pandas as pd

sys.path.append("metric")
from metric.smooth_bleu import codenn_smooth_bleu
from metric.meteor.meteor import Meteor
from metric.rouge.rouge import Rouge
from metric.cider.cider import Cider

import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(format='[%(asctime)s - %(levelname)s - %(name)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def Commitbleus(refs, preds):
    r_str_list = []
    p_str_list = []
    for r, p in zip(refs, preds):
        r_str_list.append([" ".join([str(token_id) for token_id in r[0]])])
        p_str_list.append(" ".join([str(token_id) for token_id in p]))
    bleu_list, bleu_lists = codenn_smooth_bleu(r_str_list, p_str_list)
    codenn_bleu = bleu_list[0]
    B_Norm = round(codenn_bleu, 2)
    print("BLEU: ", B_Norm)
    scores = [bleu_list[0] for bleu_list in bleu_lists]
    return B_Norm, scores


def read_to_list(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        data = [json.loads(d) for d in f.readlines()]
    refs, preds = [], []
    for row in data:
        ref = [x for x in re.split(r'(\W)', row["ref"].lower()) if x.strip()]
        pred = [x for x in re.split(r'(\W)', row["pred"].lower()) if x.strip()]
        refs.append(ref)
        preds.append(pred)
    return refs, preds


def metetor_rouge_cider(refs, preds):
    refs_dict = {}
    preds_dict = {}
    for i in range(len(preds)):
        preds_dict[i] = [" ".join(preds[i])]
        refs_dict[i] = [" ".join(refs[i][0])]

    score_Meteor, scores_Meteor = Meteor().compute_score(refs_dict, preds_dict)
    print("Meteor: ", round(score_Meteor * 100, 2))

    score_Rouge, scores_Rouge = Rouge().compute_score(refs_dict, preds_dict)
    print("Rouge-L: ", round(score_Rouge * 100, 2))

    score_Cider, scores_Cider = Cider().compute_score(refs_dict, preds_dict)
    print("Cider: ", round(score_Cider, 2))

    return round(score_Meteor * 100, 2), round(score_Rouge * 100, 2), round(score_Cider, 2), scores_Meteor, scores_Rouge, scores_Cider


def compute(result_path):
    refs, preds = read_to_list(result_path)
    refs = [[t] for t in refs]
    bleu_score, scores_bleu = Commitbleus(refs, preds)
    meteor, rouge, cider, scores_Meteor, scores_Rouge, scores_Cider = metetor_rouge_cider(refs, preds)
    print()
    return bleu_score, meteor, rouge, cider, scores_bleu, scores_Meteor, scores_Rouge, scores_Cider


def evaluate(RQ):
    results = []
    lan_list = ["java", "csharp", "cpp", "python", "javascript"]
    for number in [0, 1, 2, 4, 8, 16, 32, 64]:
        tmp = {"number": number}
        avg_bleu_score, avg_meteor, avg_rouge, avg_cider = 0.0, 0.0, 0.0, 0.0
        for lan in lan_list:
            print(number, lan)
            result_path = f"{RQ}/{lan}/{number}.txt"
            bleu_score, meteor, rouge, cider, _, _, _, _ = compute(result_path)
            tmp.update({f"{lan}_BLEU": bleu_score,
                        f"{lan}_METEOR": meteor,
                        f"{lan}_ROUGE-L": rouge,
                        f"{lan}_Cider": cider})
            avg_bleu_score += bleu_score
            avg_meteor += meteor
            avg_rouge += rouge
            avg_cider += cider
        tmp.update({"avg_BLEU": round(avg_bleu_score / len(lan_list), 2),
                    "avg_METEOR": round(avg_meteor / len(lan_list), 2),
                    "avg_ROUGE-L": round(avg_rouge / len(lan_list), 2),
                    "avg_Cider": round(avg_cider / len(lan_list), 2)})
        results.append(tmp)

    columns = ["number"]
    for lan in lan_list:
        columns.extend([f"{lan}_BLEU", f"{lan}_METEOR", f"{lan}_ROUGE-L", f"{lan}_Cider"])
    columns.extend(["avg_BLEU", "avg_METEOR", "avg_ROUGE-L", "avg_Cider"])
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(f"{RQ}/result.csv", index=False)


if __name__ == '__main__':
    evaluate("RQ1")
    evaluate("RQ2")
