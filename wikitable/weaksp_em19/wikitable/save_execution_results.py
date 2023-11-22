from sempar.context.table_question_context import TableQuestionContext
from sempar.domain_languages.wikitable_abstract_language import WikiTableAbstractLanguage
from allennlp.semparse.domain_languages import ParsingError, ExecutionError
from allennlp.data.tokenizers.token import Token

import sys, os
sys.path.insert(
    0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
)

from wikitable.model.baseline import Programmer
from wikitable.model.seq import SeqProgrammer
from wikitable.model.struct import StructProgrammer
from wikitable.reader.reader import WTReader
from wikitable.reader.util import load_jsonl, load_jsonl_table, load_actions
from wikitable.train_config.train_seq_config import config
from wikitable.trainer.util import get_sketch_prod, filter_sketches, create_opt, clip_model_grad, weight_init, set_seed
from worldcode.worldcode_multiset_v2 import WorldcodeGenerator, get_vote_score, linearize_lfs
from multiset import Multiset
from copy import deepcopy
import torch
import sys
import pickle
import copy
from tqdm import tqdm
from pathlib import Path
from typing import List
from collections import defaultdict
import pdb



# hdlr = logging.FileHandler('/tmp/train_baseline.log')
# logger.addHandler(hdlr)

class ReaderUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "wikitable.reader.reader"
        return super().find_class(module, name)


if __name__ == "__main__":
    NUM_MAX_TABLES, NUM_MAX_SAMPLING, MIN_LF_NUM = 40, 10, 10

    # load raw data
    with open(config.reader_pkl, 'rb') as f:
        unpickler = ReaderUnpickler(f)
        wt_reader = unpickler.load()
    with open(config.sketch_pkl, 'rb') as f:
        example_dict = pickle.load(f)[1]
    with open(config.sketch_test_pkl, 'rb') as f:
        test_example_dict = pickle.load(f)[1]
    sketch_lf_actions = load_actions(config.sketch_action_file)
    
    with open('./id_to_proxy_data_multiset_preproc.pkl', 'rb') as f:
        id2pr = pickle.load(f)

    # load data
    train_examples = wt_reader.train_examples
    dev_examples = wt_reader.dev_examples
    test_examples = wt_reader.test_examples
    tables = wt_reader.table_dict
    pretrained_embeds = wt_reader.wordvec
    id2prod, prod2id= get_sketch_prod(train_examples, tables)

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



    counter = 0
    # target_lfs_dict = {}
    worldcode_dict = {}
    for example in tqdm(train_examples):
        # if it does not trigger any programs, then no need to train it
        if (example["id"], example["context"]) not in example_dict:
            continue
        if len(example["tokens"]) > 30:
            continue

        target_lfs = example_dict[(example["id"], example["context"])]
        linearized_target_lfs = linearize_lfs(target_lfs)

        if len(linearized_target_lfs) > MIN_LF_NUM: # perform filtering only if there are enough logical forms available

            table_id = example["context"]
            table_lines = tables[table_id]["raw_lines"]

            target_value, target_can = example["answer"] # (targeValue, targetCan)
            tokenized_question = [ Token(token, pos_=pos) for token,pos in  zip(example["tokens"], example["pos_tags"])]
            if len(tokenized_question) == 1: continue # ignore the single-token one

            context = TableQuestionContext.read_from_lines(table_lines, tokenized_question)
            context.take_corenlp_entities(example["entities"])
            context.take_features(example["features"], example["prop_features"])
            context.anonymized_tokens = example["tmp_tokens"]

            '''
            id2pr is a dictionary that maps sentence id to a list of proxy examples
            {'nt-0': [{
                'id': 't_204_515', 
                'context_type_multiset': Multiset({'number_column': 5, 'string_column': 8, 'date_column': 2, 'num2_column': 1}), 
                'similarity_score': 1.0
            }...]}
            '''
            proxy_examples = id2pr[example['id']]

            # world_code = get_world_code(target_lfs, context, proxy_examples, tables, tokenized_question, table_id_to_context)
            wc_generator = WorldcodeGenerator(NUM_MAX_TABLES, NUM_MAX_SAMPLING, target_lfs, context)
            world_code = wc_generator.get_world_code(proxy_examples, tables, tokenized_question, table_id_to_context)
            # print(len(world_code))

            if len(world_code) > 10:

                # scores = get_vote_score(world_code)

                # target_lfs_ = {}
                # i = 0
                # threshold = max(scores)/2
                # for sketch, lf_list in target_lfs.items():
                #     new_lf_list = [lf for j, lf in enumerate(lf_list) if scores[i+j] > threshold]
                #     i += len(lf_list)
                #     if len(new_lf_list) > 0:
                #         target_lfs_[sketch] = new_lf_list
                # target_lfs = target_lfs_
                counter += 1
                
            worldcode_dict[(example["id"], example["context"])] = world_code

        # target_lfs_dict[(example["id"], example["context"])] = target_lfs

    with open("execution_results_resample.pkl", "wb") as f:
        pickle.dump(worldcode_dict, f)

    print('number of successful cases:', counter) # v1: 4243, v2: 4996, resample: 4065. 6hrs (out of 6207)