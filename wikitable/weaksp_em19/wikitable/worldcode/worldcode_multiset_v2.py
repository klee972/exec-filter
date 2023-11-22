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

from wikitable.sempar.context.table_question_context import TableQuestionContext
from wikitable.sempar.domain_languages.wikitable_abstract_language import WikiTableAbstractLanguage

import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def bracket_separator(lf):
    return lf.replace('(', '( ').replace(')', ' )')

def bracket_merger(lf):
    return lf.replace('( ', '(').replace(' )', ')')

def find_columns(lf):
    sep_lf = bracket_separator(lf)
    sep_lf = sep_lf.split(' ')
    tokens = [token for token in sep_lf if 'column:' in token]
    return tokens

def find_entities(lf):
    sep_lf = bracket_separator(lf)
    sep_lf = sep_lf.split(' ')
    tokens = []
    for idx, token in enumerate(sep_lf):
        if 'filter' in token:
            tokens.append((sep_lf[idx + 3], sep_lf[idx + 2])) # key is entity and value is corresponding column
    return tokens

def linearize_lfs(dict_lfs):
    list_lfs = []
    for sk, lfs in dict_lfs.items():
        for lf in lfs:
            list_lfs.append(lf)
    return list_lfs


class WorldcodeGenerator(object):
    def __init__(self, num_max_tables, num_max_samling, target_lfs, query_context):
        self.num_max_tables = num_max_tables
        self.num_max_samling = num_max_samling
        self.target_lfs = target_lfs
        self.query_context = query_context

        self.lf_columns_set = set()
        self.lf_entities_set = set()
        self.lf_entities_dict = defaultdict(set)
        for sketch, lf_list in self.target_lfs.items():
            for lf in lf_list:
                columns = find_columns(lf)
                entities = find_entities(lf)
                self.lf_columns_set.update(columns)
                self.lf_entities_set.update(entities)
                for entity, column in entities:
                    self.lf_entities_dict[column].add(entity)

        for column in self.lf_entities_dict.keys():
            self.lf_columns_set.discard(column)

    def _sample_columns_and_entities_to_replace(self, table_column_dict, table_data, world):
        column_and_entity_replacement_dict = {}
        column_replacement_dict = {}

        for column, entities in self.lf_entities_dict.items():
            column_type = column.split(':')[0]
            columns_with_same_type = table_column_dict[column_type]
            random.shuffle(columns_with_same_type)
            sampled_column = columns_with_same_type.pop()
            entity_sampling_pool = [row[sampled_column] for row in table_data if row[sampled_column] not in [None, '']]
            column_replacement_dict[column] = sampled_column

            if len(entities) <= len(entity_sampling_pool):
                sampled_entities = random.sample(entity_sampling_pool, len(entities))
            else: 
                sampled_entities = random.choices(entity_sampling_pool, k=len(entities))

            for i, entity in enumerate(entities):
                if column_type == 'string_column':
                    sampled_entity_ = 'string:' + sampled_entities[i]
                    world.add_constant(sampled_entity_, sampled_entity_)
                else:
                    sampled_entity_ = str(sampled_entities[i])
                    world.add_constant(sampled_entity_, sampled_entities[i])

                column_and_entity_replacement_dict[column + ' ' + entity] = (sampled_column, sampled_entity_)

        for column in self.lf_columns_set:
            column_type = column.split(':')[0]
            columns_with_same_type = table_column_dict[column_type]
            random.shuffle(columns_with_same_type)
            sampled_column = columns_with_same_type.pop()
            column_replacement_dict[column] = sampled_column
        
        return column_and_entity_replacement_dict, column_replacement_dict


    def get_world_code(self, proxy_data, tables, tokenized_question, table_id_to_context):

        worldcode = []
        for instance in proxy_data:
            instance_table_id = instance['id']
            context, context_type = table_id_to_context[instance_table_id]

            # table_lines = tables[instance_table_id]['raw_lines']
            # context = TableQuestionContext.read_from_lines(table_lines, tokenized_question)
            # context.take_corenlp_entities([])

            world = WikiTableAbstractLanguage(context)
            for column, entities in self.lf_entities_dict.items(): # prepare world
                for entity in entities:
                    column_type = column.split(':')[0]
                    if column_type == 'string_column':
                        world.add_constant(entity, entity)
                    else:
                        world.add_constant(str(entity), entity)

            
            table_data = context.table_data
            table_column_types = table_data[0].keys()

            table_column_dict = defaultdict(list)
            for typ in table_column_types:
                table_column_dict[typ.split(':')[0]].append(typ)

            sampling_iter = 0
            while True:
                self.column_and_entity_replacement_dict, self.column_replacement_dict = \
                    self._sample_columns_and_entities_to_replace(deepcopy(table_column_dict), table_data, world)

                entity_and_column_replaced_lfs = []
                for sketch, lf_list in self.target_lfs.items():
                    for lf in lf_list:
                        sep_lf = bracket_separator(lf)
                        sep_lf = sep_lf.split(' ')
                        lf_replaced = sep_lf[:]
                        
                        for i in range(len(lf_replaced) - 1):
                            if lf_replaced[i] + ' ' + lf_replaced[i+1] in self.column_and_entity_replacement_dict.keys():
                                clause = self.column_and_entity_replacement_dict[lf_replaced[i] + ' ' + lf_replaced[i+1]]
                                lf_replaced[i], lf_replaced[i+1] = clause
                                continue
                            if lf_replaced[i] in self.column_replacement_dict.keys():
                                lf_replaced[i] = self.column_replacement_dict[lf_replaced[i]]
                        
                        lf_replaced = bracket_merger(' '.join(lf_replaced))
                        entity_and_column_replaced_lfs.append(lf_replaced)

                execution_results = []
                error_count, success_count = 0, 0
                error_indices = []
                for lf_idx, lf in enumerate(entity_and_column_replaced_lfs):
                    try:
                        denotation = world.execute(lf)
                        if type(denotation) == list:
                            denotation_ = ''
                            for item in denotation:
                                denotation_ += item
                            denotation = denotation_
                        success_count+=1
                    except:
                        error_count+=1
                        denotation = 'EXECUTION_ERROR'
                        err_table = instance_table_id
                        err_lf = lf
                        error_indices.append(lf_idx)
                    execution_results.append(denotation)

                if error_count / (error_count + success_count) < 0.1: # sampling was good enough
                    worldcode.append(execution_results)
                    break

                if sampling_iter > self.num_max_samling: # there was no good sampling. just skip this one
                    break
                sampling_iter += 1

            if len(worldcode) >= self.num_max_tables:
                break

        return worldcode


def get_vote_score(world_codes, probabilities=None):
    # "probabilities" are normalized probabilities of logical forms
    if probabilities is None:
        probabilities = [1/len(world_codes[0])] * len(world_codes[0])
    scores = [0] * len(world_codes[0]) # number of lfs
    for execution_results in world_codes:
        denotation_scores = defaultdict(float)
        for i, denotation in enumerate(execution_results):
            if denotation == 'EXECUTION_ERROR':
                score = 0
            else:
                score = probabilities[i]
            denotation_scores[denotation] += score
        for i, denotation in enumerate(execution_results):
            if denotation == 'EXECUTION_ERROR':
                scores[i] += probabilities[i] # when EXECUTION_ERROR, consider it as unique value
            else:
                scores[i] += denotation_scores[denotation]
    return scores

def get_vote_score_hard_type(world_codes, probabilities=None):
    # "probabilities" are normalized probabilities of logical forms
    if probabilities is None:
        probabilities = [1/len(world_codes[0])] * len(world_codes[0])
    scores = [0] * len(world_codes[0]) # number of lfs
    for execution_results in world_codes:
        denotation_scores = defaultdict(float)
        for i, denotation in enumerate(execution_results):
            if denotation == 'EXECUTION_ERROR':
                denotation_scores[f'EXECUTION_ERROR_{i}'] += probabilities[i]
            else:
                denotation_scores[denotation] += probabilities[i]

        majority_denotation = max(denotation_scores, key=denotation_scores.get)

        if type(majority_denotation) is str and 'EXECUTION_ERROR' in majority_denotation:
            maj_i = int(majority_denotation.split('_')[-1])
        
            for i, denotation in enumerate(execution_results):
                if denotation == 'EXECUTION_ERROR':
                    if i == majority_denotation.split('_')[-1]:
                        scores[i] += 1/len(world_codes)
                        break
        else:
            for i, denotation in enumerate(execution_results):
                if denotation == majority_denotation:
                    scores[i] += 1/len(world_codes)
    return scores

def write_examples(output_file, sentence_tokens, target_lfs, scores=None, world_codes=None, probabilities=None, table_id=None):
    lfs = linearize_lfs(target_lfs)
    sentence = ' '.join(sentence_tokens)
    if scores == None:
        scores = [0.0] * len(lfs)
    with open(output_file, 'a') as f:
        f.write('table_id: ' + table_id + '\n')
        f.write('sentence: ' + sentence + '\n')
        for idx, score in enumerate(scores):
            f.write(f'probability: {probabilities[idx]:.6f} \t\t score: {score:.6f} \t\t logical form: {lfs[idx]}\n')
            if world_codes:
                f.write(f'world code: {world_codes[idx]}\n')
        f.write('\n\n')

