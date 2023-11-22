#! /usr/bin/env python


import sys
import os
import json
import argparse

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
)

from allennlp.models.archival import load_archive
from allennlp_semparse.dataset_readers import NlvrV2PairedDatasetReader
from allennlp_semparse.models import NlvrPairedSemanticParser
from allennlp_semparse.domain_languages import NlvrLanguageFuncComposition
from allennlp_semparse.domain_languages.nlvr_language_v2 import Box
from allennlp_semparse.common import ParsingError, ExecutionError
from scripts.nlvr_v2.worldcode.worldcode import (
    get_world_code, get_vote_score, get_weighted_vote_score, get_weighted_vote_score_soft_type,
    write_examples,
    get_agenda_recall_score
)
from tqdm import tqdm
import pdb
import torch
import numpy as np



def make_data(
    input_file: str,
    output_file: str,
    archived_model_file: str,
    max_num_decoded_sequences: int,
    cuda_device: int,
    prune_data: bool,
    selection_threshold: float,
) -> None:
    reader = NlvrV2PairedDatasetReader(output_agendas=False)
    model = load_archive(archived_model_file, cuda_device=cuda_device).model
    if not isinstance(model, NlvrPairedSemanticParser):
        model_type = type(model)
        raise RuntimeError(
            f"Expected an archived NlvrMMLSemanticParser, but found {model_type} instead"
        )
    # Tweaking the decoder trainer to coerce the it to generate a k-best list. Setting k to 100
    # here, so that we can filter out the inconsistent ones later.
    # model._decoder_beam_search == allennlp_semparse.state_machines.beam_search.BeamSearch
    model._beam_search._beam_size = 100
    model._beam_search._per_node_beam_size = 100
    model.training = False

    ########################## CONFIGS ###########################
    # model._decoder_step.cbs_mode = True

    is_worldcode = True
    is_agenda_score_weighting = True

    is_write_examples = False
    example_path = 'wc_examples/iwbs5_aww_erm.txt'

    is_iwbs_by_agenda = True

    selection_scheme_key = 'weighted_voting_soft'
    selection_scheme_map = {
        'voting': get_vote_score, 
        'weighted_voting': get_weighted_vote_score,
        'weighted_voting_soft': get_weighted_vote_score_soft_type
        }
    selection_scheme = selection_scheme_map[selection_scheme_key]
    num_proxy_data = 20
    ##############################################################
    
    num_outputs = 0
    num_w_candidates = 0
    num_sentences = 0
    num_correct, num_correct_after_pruning = 0, 0
    with open(output_file, "w") as outfile:
        for line in tqdm(open(input_file)):
            num_sentences += 1
            input_data = json.loads(line)
            sentence = input_data["sentence"]

            model._beam_search._beam_size = 100

            structured_representations = input_data["worlds"]
            labels = input_data["labels"]
            instance = reader.text_to_instance(sentence, structured_representations)

            worlds = []
            for structured_representation in structured_representations:
                boxes = {
                    Box(object_list, box_id)
                    for box_id, object_list in enumerate(structured_representation)
                }
                worlds.append(NlvrLanguageFuncComposition(boxes))

            agenda_fulfilled = False
            best_agenda_score = 0

            while not agenda_fulfilled:
                outputs = model.forward_on_instance(instance)
                action_strings = outputs["best_action_strings"]
                logical_forms = outputs["logical_form"]
                log_probs = outputs["batch_action_scores"]

                correct_sequences = []
                correct_logical_forms = []
                log_probs_of_correct_sequences = []

                # Checking for consistency
                # worlds = [NlvrLanguageFuncComposition(structure) for structure in structured_representations]
                for sequence, logical_form, log_prob in zip(action_strings, logical_forms, log_probs):
                    try:
                        denotations = [world.execute(logical_form) for world in worlds]
                    except (ParsingError, ExecutionError, TypeError):
                        print("Error executing: {}\n".format(logical_form))
                        denotations = [False for world in worlds]
                    denotations_are_correct = [
                        label.lower() == str(denotation).lower()
                        for label, denotation in zip(labels, denotations)
                    ]
                    if all(denotations_are_correct):
                        correct_sequences.append(sequence)
                        correct_logical_forms.append(logical_form)
                        log_probs_of_correct_sequences.append(log_prob)
                num_correct += len(correct_sequences)

                scores = None

                if is_iwbs_by_agenda:
                    agendas = worlds[0].get_agenda_for_sentence(sentence)
                    agenda_scores = get_agenda_recall_score(agendas, correct_sequences)

                    if len(agenda_scores) > 0 and max(agenda_scores) > best_agenda_score:
                        best_agenda_score = max(agenda_scores)
                        best_outputs = outputs

                        best_agenda_scores = agenda_scores
                        best_correct_sequences = correct_sequences
                        best_correct_logical_forms = correct_logical_forms
                        best_log_probs_of_correct_sequences = log_probs_of_correct_sequences

                    if (len(agenda_scores) < 1 or best_agenda_score < 1) \
                        and model._beam_search._beam_size < 100 * (2 ** 5):
                        model._beam_search._beam_size *= 2
                        continue
                    else:
                        agenda_fulfilled = True
                
                elif is_agenda_score_weighting:
                    agendas = worlds[0].get_agenda_for_sentence(sentence)
                    agenda_scores = get_agenda_recall_score(agendas, correct_sequences)
                    agenda_fulfilled = True

                else:
                    agenda_scores = 0.0
                    agenda_fulfilled = True

            if is_iwbs_by_agenda:
                agenda_scores = best_agenda_scores
                correct_sequences = best_correct_sequences
                correct_logical_forms = best_correct_logical_forms
                log_probs_of_correct_sequences = best_log_probs_of_correct_sequences


            if len(correct_sequences) > 0 and is_worldcode:
                
                proxy_data = input_data["proxy_data"][:num_proxy_data]
                world_codes = []
                for sequence in correct_sequences:
                    world_code = get_world_code(sequence, proxy_data)
                    world_codes.append(world_code)

                if is_agenda_score_weighting:
                    probabilities = np.array(agenda_scores).reshape(-1, 1)
                else:
                    probabilities = np.exp(torch.Tensor(
                        log_probs_of_correct_sequences).cpu().numpy()).reshape(-1, 1)
                scores = selection_scheme(world_codes, probabilities)

                best_score = 0
                selected_action_sequences = []
                world_codes_after_pruning = []
                new_scores = []
                for idx, score in enumerate(scores):
                    if score >= selection_threshold:
                        selected_action_sequences.append(correct_sequences[idx])
                        world_codes_after_pruning.append(world_codes[idx])
                        new_scores.append(score)

                if len(selected_action_sequences) == 0:
                    for idx, score in enumerate(scores):
                        if score > best_score:
                            best_score = score
                            selected_action_sequences = [correct_sequences[idx]]
                            world_codes_after_pruning = [world_codes[idx]]
                            new_scores = [best_score]
                        elif score == best_score:
                            selected_action_sequences.append(correct_sequences[idx])
                            world_codes_after_pruning.append(world_codes[idx])
                            new_scores.append(score)

                correct_sequences = sorted(selected_action_sequences, 
                    key=lambda x: new_scores[selected_action_sequences.index(x)], reverse=True)
                
                correct_logical_forms = sorted(correct_logical_forms, 
                    key=lambda x: scores[correct_logical_forms.index(x)], reverse=True)

                scores = sorted(scores, reverse=True)

            elif is_iwbs_by_agenda:
                correct_sequences = sorted(correct_sequences, 
                    key=lambda x: agenda_scores[correct_sequences.index(x)], reverse=True)

                correct_logical_forms = sorted(correct_logical_forms, 
                    key=lambda x: agenda_scores[correct_logical_forms.index(x)], reverse=True)

                scores = sorted(agenda_scores, reverse=True)

            if is_write_examples:
                if num_sentences == 1:
                    with open(example_path, 'a') as exf:
                        exf.write('\n\n\nNEXT ITER START\n\n\n')
                # if num_sentences <= 500:
                write_examples(example_path, sentence, correct_logical_forms, scores)

            correct_sequences = correct_sequences[:max_num_decoded_sequences]
            num_correct_after_pruning += len(correct_sequences)
            output_data = {
                "id": input_data["identifier"] if "identifier" in input_data else input_data["id"],
                "sentence": sentence,
                "worlds": structured_representations,
                "labels": input_data["labels"],
            }
            if correct_sequences:
                output_data.update(
                    {"correct_sequences": correct_sequences}
                )
                num_w_candidates += 1

            if prune_data:
                # Don't write instances without consistent candidates
                continue
            json.dump(output_data, outfile)
            outfile.write("\n")
            num_outputs += 1
        outfile.close()
    print(f"Total input sentences: {num_sentences}")
    print(f"{num_w_candidates} have candidates out of {num_outputs} sentences written.")
    avg_correct = float(num_correct) / num_w_candidates
    avg_correct_after_pruning = float(num_correct_after_pruning) / num_w_candidates
    print("Num candidates per example: {}".format(avg_correct))
    print("Num candidates per example after pruning: {}".format(avg_correct_after_pruning))
    print(f"Output written to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input data file")
    parser.add_argument("output", type=str, help="Output data file")
    parser.add_argument(
        "archived_model", type=str, help="Path to archived model.tar.gz to use for decoding"
    )
    parser.add_argument(
        "--max-num-sequences",
        type=int,
        dest="max_num_sequences",
        help="Maximum number of sequences per instance to output",
        default=20,
    )
    parser.add_argument("--cuda-device", dest="cuda_device", type=int, default=-1)
    parser.add_argument(
        "--prune-data",
        dest="prune_data",
        help="Should we only keep examples for which at least one correct logical-form is found?",
        action="store_true",
    )
    args = parser.parse_args()
    make_data(
        args.input, args.output, args.archived_model, args.max_num_sequences, args.cuda_device,
        args.prune_data, 0.8
    )
