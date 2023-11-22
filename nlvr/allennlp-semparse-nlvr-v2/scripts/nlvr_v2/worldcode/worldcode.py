from pathlib import Path
from collections import defaultdict
import json, random, codecs
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import pdb
import os, sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))))
)

from allennlp_semparse.domain_languages import NlvrLanguageFuncComposition
from allennlp_semparse.domain_languages.nlvr_language_v2 import Box
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def get_world_code(production_rule_seq, proxy_data):
    world_code = []
    for instance in proxy_data:
        structured_reps = instance['worlds']
        for structured_rep in structured_reps:
            boxes = {
                Box(object_list, box_id)
                for box_id, object_list in enumerate(structured_rep)
            }
            world = NlvrLanguageFuncComposition(boxes)
            world_code.append(world.execute_action_sequence(production_rule_seq))
    return world_code

def get_vote_score(world_codes, *args):
    world_codes = np.array(world_codes)
    code_vote = np.mean(world_codes, axis=0)
    gold_code = code_vote.round().astype('int')
    return np.mean(world_codes == gold_code, axis=1).tolist()

def get_weighted_vote_score_soft_type(world_codes, probabilities):
    world_codes = np.array(world_codes)
    score_true = np.sum(world_codes * probabilities, axis=0)
    score_false = np.sum((1-world_codes) * probabilities, axis=0)
    scores = np.sum(world_codes * score_true + (1-world_codes) * score_false, axis=1)
    normalized_scores = scores / max(scores)
    return normalized_scores.tolist()

def get_weighted_vote_score(world_codes, probabilities): # hard type
    world_codes = np.array(world_codes)
    code_vote = np.sum((world_codes-0.5) * probabilities, axis=0) / np.sum(probabilities)
    gold_code = code_vote > 0
    return np.mean(world_codes == gold_code, axis=1).tolist()

def set_recall(tgt, pred):
    if len(tgt) == 0:
        return 0
    score = 0
    for item in tgt:
        if item in pred:
            score += 1
    return score / len(tgt)

def set_precision(tgt, pred):
    if len(pred) == 0:
        return 0
    score = 0
    for item in pred:
        if item in tgt:
            score += 1
    return score / len(pred)

def get_agenda_recall_score(agendas, sequences):
    agenda_scores = []
    for sequence in sequences:
        recall = set_recall(agendas, sequence)
        agenda_scores.append(recall)
    return agenda_scores

def get_agenda_f1_score(agendas, sequences):
    agenda_scores = []
    for sequence in sequences:
        recall = set_recall(agendas, sequence)
        precision = set_precision(agendas, sequence)
        f1_score = 2 * precision * recall / (precision + recall + 1e-9)
        agenda_scores.append(f1_score)
    return agenda_scores


def write_examples(output_file, sentence, correct_logical_forms, scores=None, world_codes=None):
    if scores == None:
        scores = [0.0] * len(correct_logical_forms)
    with open(output_file, 'a') as f:
        f.write('sentence: ' + sentence + '\n')
        for idx, score in enumerate(scores):
            f.write(f'score: {score:.6f} \t\t logical form: {correct_logical_forms[idx]}\n')
            if world_codes:
                f.write(f'world code: {world_codes[idx]}\n')
        f.write('\n\n')


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings(action='ignore')

    NUM_PROXY_UTT = 20
    output_file = "../resources/data/nlvr/processed/agendav6_SORT_ML11/train_grouped_proxy_utt.json"

    train_grouped = [json.loads(i) for i in open("../resources/data/nlvr/processed/agendav6_SORT_ML11/train_grouped.json")]
    dataset = jsonl_to_df(train_grouped)

    count = 0
    for data in train_grouped:
        
        if len(data['correct_sequences']) > 0:
            count += 1
            utt_id = data['identifier']
            sentence = data['sentence']
            proxy_data = get_data_with_similar_utt(utt_id, dataset, NUM_PROXY_UTT)
            world_codes = []
            correct_logical_forms = []
            for correct_seq in data['correct_sequences']:
                world_code, logical_form = get_world_code(correct_seq, proxy_data)
                world_codes.append(world_code)
                correct_logical_forms.append(logical_form)
            scores = get_vote_score(world_codes)

            write_examples()

            majority_correct_sequences = []
            for idx, seq in enumerate(data['correct_sequences']):
                if scores[idx] == 1:
                    majority_correct_sequences.append(seq)
                    # majority_correct_logical_forms.append(correct_logical_forms[idx])

            # data['proxy_data'] = [json.loads(data_point.to_json()) for data_point in proxy_data]
            data['majority_correct_sequences'] = majority_correct_sequences

            # pdb.set_trace()
        
            # json.dump(data, outfile)
            # outfile.write("\n")

            # pdb.set_trace()
            if count > 20:
                break

    

