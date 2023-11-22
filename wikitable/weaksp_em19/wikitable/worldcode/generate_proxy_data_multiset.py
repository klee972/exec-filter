from collections import defaultdict
import json, random, codecs
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import pdb
import os, sys
from tqdm import tqdm
import pickle
import argparse

from wikitable.sempar.context.table_question_context import TableQuestionContext
from allennlp.data.tokenizers.token import Token
from multiset import Multiset
from copy import deepcopy


class ReaderUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "wikitable.reader.reader"
        return super().find_class(module, name)


def normalize_text(utt):
    utt_ = []
    numbers = {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six"}
    for i in utt.split():
        utt_.append(numbers.get(i, i.lower()))
    return " ".join(utt_)


def build_dataframe(examples):
    df_ = defaultdict(list, [])
    for inst in examples:
        df_["id"].append(inst["id"])
        df_["utt"].append(inst["question"])
        df_["context"].append(inst["context"])
        # df_["normalized_question"].append(normalize_text(inst["question"]))
    df = pd.DataFrame(df_)
    return df


def find_column_types(lf):
    bracket_separator = lambda x: x.replace('(', '( ').replace(')', ' )')
    sep_lf = bracket_separator(lf)
    sep_lf = sep_lf.split(' ')
    types = [token for token in sep_lf if 'column:' in token]
    # types = [token.split(':')[0] for token in sep_lf if 'column:' in token]
    return types


def preprocess_tables(table_id_to_context):
    '''
    eliminate tables with "None"s or ""s
    '''
    table_id_to_context_ = deepcopy(table_id_to_context)
    for table_id, (ctx, ctx_multiset) in table_id_to_context.items():
        for column in ctx.table_data[0].keys():
            for row in ctx.table_data:
                entity = row[column]
                if entity in [None, '']:
                    try: del table_id_to_context_[table_id]
                    except: pass

    return table_id_to_context_


def get_tables_with_similar_column_types(target_lfs, example_context_id, table_id_to_context, table_id_to_context_preprocessed):
    ex_ctx, ex_ctx_multiset = table_id_to_context[example_context_id]
    try: del table_id_to_context_preprocessed[example_context_id]
    except: pass

    lf_type_set = set()
    for sketch, lf_list in target_lfs.items():
        for lf in lf_list:
            types = find_column_types(lf)
            lf_type_set.update(types)

    lf_type_list = [lf_type.split(':')[0] for lf_type in lf_type_set]
    lf_type_multiset = Multiset(lf_type_list)

    output = []
    for table_id, (ctx, ctx_multiset) in table_id_to_context_preprocessed.items():
        intersection = ex_ctx_multiset & ctx_multiset
        score = len(intersection) / len(ex_ctx_multiset)

        if lf_type_multiset <= ctx_multiset:
            output.append({
                'id': table_id,
                # 'context': ctx,
                'context_type_multiset': ctx_multiset,
                'similarity_score': score
            })

    sorted_output = sorted(output, key=lambda x: x['similarity_score'], reverse=True)

    # pdb.set_trace()
    return sorted_output



def generate_proxy_tables(train_examples, tables, example_dict, output_file):
    import warnings
    warnings.filterwarnings(action='ignore')

    NUM_PROXY_UTT = 40

    dataset = build_dataframe(train_examples)

    dummy_tokenized_question = [
        Token(token, pos_=pos) for token,pos in zip(train_examples[0]["tokens"], train_examples[0]["pos_tags"])
    ]

    table_id_to_context = {}
    for table_id, table in tables.items():
        table_lines = table['raw_lines']
        context = TableQuestionContext.read_from_lines(table_lines, dummy_tokenized_question)
        context.take_corenlp_entities([])

        ctx_col_types = [ctx_col.split(':')[0] for ctx_col in context.table_data[0]]
        ctx_multiset = Multiset(ctx_col_types)

        table_id_to_context[table_id] = (context, ctx_multiset)

    table_id_to_context_ = preprocess_tables(table_id_to_context)

    output_dict = {}

    for example in tqdm(train_examples):
        if (example["id"], example["context"]) not in example_dict:
            continue

        target_lfs = example_dict[(example["id"], example["context"])]

        proxy_data = get_tables_with_similar_column_types(
            target_lfs, example["context"], table_id_to_context, table_id_to_context_
        )

        # data['proxy_data'] = proxy_data
        output_dict[example["id"]] = proxy_data

    with open(output_file, "wb") as f:
        pickle.dump(output_dict, f)


if __name__ == "__main__":
    output_file = "./id_to_proxy_data_multiset_preproc.pkl"

    with open("processed/wikitable_glove_42B_minfreq_3.pkl", 'rb') as f:
        unpickler = ReaderUnpickler(f)
        wt_reader = unpickler.load()

    with open("processed/train.pkl", 'rb') as f:
        example_dict = pickle.load(f)[1]

    train_examples = wt_reader.train_examples
    tables = wt_reader.table_dict
    
    generate_proxy_tables(train_examples, tables, example_dict, output_file)
