from collections import defaultdict
import json, random, codecs
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import pdb
import os, sys
from tqdm import tqdm
import argparse


def normalize_text(utt):
    utt_ = []
    numbers = {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six"}
    for i in utt.split():
        utt_.append(numbers.get(i, i.lower()))
    return " ".join(utt_)


def jsonl_to_df(jsonl):
    df_ = defaultdict(list, [])
    for inst in jsonl:
        df_["utt_id"].append(inst["identifier"])
        df_["sentence"].append(inst["sentence"])   
        df_["utt"].append(normalize_text(inst["sentence"]))
        df_["worlds"].append(inst["worlds"])
    df = pd.DataFrame(df_)
    return df


def get_data_with_similar_utt(utt_id, df, num_proxy_data):
    inst = df[(df.utt_id==utt_id)]
    utt = inst.utt.to_list()[0]
    df_ = df.copy()
    df_["bleu_score"] = df_.apply(lambda x: get_pairwise_bleu(x.utt, utt), axis=1)
    df_ = df_.sort_values(["bleu_score"], ascending=False)
    output = []
    for idx, row in df_.iterrows():
        if row.utt_id != utt_id:
            output.append(row)
        if len(output) >= num_proxy_data:
            return output


def get_data_with_similar_utt_by_sentence(sentence, df, num_proxy_data):
    utt = normalize_text(sentence)
    df_ = df.copy()
    df_["bleu_score"] = df_.apply(lambda x: get_pairwise_bleu(x.utt, utt), axis=1)
    df_ = df_.sort_values(["bleu_score"], ascending=False)
    output = []
    for idx, row in df_.iterrows():
        output.append(row)
        if len(output) >= num_proxy_data:
            return output


def get_pairwise_bleu(utt1, utt2):
    reference = [utt1.split()]
    candidate = utt2.split() 
    return sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))


def generate_proxy_utt_dataset(input_file: str, output_file: str) -> None:
    import warnings
    warnings.filterwarnings(action='ignore')

    NUM_PROXY_UTT = 50

    train_grouped = [json.loads(i) for i in open(input_file)]
    dataset = jsonl_to_df(train_grouped)

    with open(output_file, "w") as outfile:
        for data in tqdm(train_grouped):
            utt_id = data['identifier']
            sentence = data['sentence']
            proxy_data = get_data_with_similar_utt(utt_id, dataset, NUM_PROXY_UTT)
            """
            proxy_data: List[Dict{utt_id, sentence, utt, worlds, bleu_score},
                             Dict{utt_id, sentence, utt, worlds, bleu_score},
                             ...
                             Dict{utt_id, sentence, utt, worlds, bleu_score}]
            len(proxy_data) == NUM_PROXY_UTT
            """
            data['proxy_data'] = [json.loads(data_point.to_json()) for data_point in proxy_data]
        
            json.dump(data, outfile)
            outfile.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="grouped NLVR data file in json format")
    parser.add_argument("output_file", type=str, help="Output file with proxy utterances in json format")
    args = parser.parse_args()
    generate_proxy_utt_dataset(args.input_file, args.output_file)
    