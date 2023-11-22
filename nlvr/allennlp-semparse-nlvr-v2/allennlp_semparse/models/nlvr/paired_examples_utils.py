from typing import List, Tuple, Dict
import torch
import allennlp
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp_semparse.fields.production_rule_field import ProductionRule
from allennlp_semparse.state_machines.states import GrammarBasedState

from allennlp_semparse.domain_languages.nlvr_language_v2 import NlvrLanguageFuncComposition
# from allennlp_semparse.models.nlvr.nlvr_paired_semantic_parser import NlvrPairedSemanticParser

def get_programs_for_paired_examples(
        parser, # : NlvrPairedSemanticParser,
        actions: List[List[ProductionRule]],
        paired_masks: List[List[int]] = None,
        paired_identifiers: List[List[str]] = None,
        paired_sentences: Dict[str, torch.LongTensor] = None,
        paired_worlds: List[List[List[NlvrLanguageFuncComposition]]] = None,
        paired_labels: List[List[List[str]]] = None,
        original_tokenoffsets: List[List[Tuple[int, int]]] = None,
        paired_tokenoffsets: List[List[Tuple[int, int]]] = None,
        paired_nt_matches: List[List[bool]] = None,
        metadata: List[Dict] = None):

    """Start-to-end parsing of paired examples to output decoded programs.

    Each instance in the batch can be paired with multiple examples. Each instance can have varying number of paired
    examples, ranging from zero (0) to P.

    The inputs are batched in a nested manner; we will flatten this into a single effective batch which can be parsed.
    """
    group_idxs: List[int] = []  # mapping to original batch_idx for this paired example
    grpidx2batch_paired_idx = []
    flat_actions: List[List[ProductionRule]] = []
    flat_sentences: TextFieldTensors = {"tokens": {"tokens": []}}   # we are only using single token-indexer
    flat_identifiers: List[str] = []
    flat_worlds: List[List[NlvrLanguageFuncComposition]] = []
    flat_label_strings: List[List[str]] = []
    flat_original_tokenoffsets: List[Tuple[int, int]] = []
    flat_paired_tokenoffsets: List[Tuple[int, int]] = []
    flat_nt_matches: List[bool] = []

    for batch_idx, masks in enumerate(paired_masks):
        for paired_idx, mask in enumerate(masks):
            # paired_idx -- is the index of paired-example for this particular instance
            if mask == 0:
                continue
            group_idxs.append(batch_idx)
            grpidx2batch_paired_idx.append((batch_idx, paired_idx))
            flat_actions.append(actions[batch_idx])
            flat_identifiers.append(paired_identifiers[batch_idx][paired_idx])
            flat_worlds.append(paired_worlds[batch_idx][paired_idx])
            flat_label_strings.append(paired_labels[batch_idx][paired_idx])
            flat_original_tokenoffsets.append(original_tokenoffsets[batch_idx][paired_idx])
            flat_paired_tokenoffsets.append(paired_tokenoffsets[batch_idx][paired_idx])
            flat_nt_matches.append(paired_nt_matches[batch_idx][paired_idx])

            # paired_sentences["tokens"]["tokens"] is a (batch_size, num_pairings, length) sized tensor
            # Pull out the (batch_idx, paired_idx, :) tensor and append. Unsqueeze(0) to make concat easier later
            flat_sentences["tokens"]["tokens"].append(
                paired_sentences["tokens"]["tokens"][batch_idx, paired_idx, :].unsqueeze(0))

    if not group_idxs:
        return None

    flat_sentences["tokens"]["tokens"] = torch.cat(flat_sentences["tokens"]["tokens"], dim=0)

    paired_initial_state = parser.get_initial_grammar_state(
        sentence=flat_sentences,
        worlds=flat_worlds,
        label_strings=flat_label_strings,
        actions=flat_actions,
    )

    group_size = len(group_idxs)
    paired_initial_state.debug_info = [[] for _ in range(group_size)]
    best_paired_final_states = parser._beam_search.search(
        num_steps=parser._max_decoding_steps,
        initial_state=paired_initial_state,
        transition_function=parser._decoder_step,
        keep_final_unfinished_states=False,
    )

    # Create a batchidx: List[paired_program] where paired_program are consistent programs from all paired examples
    # This is a collection of consistent paired programs, irrespective of the example they come from.
    # Therefore, all paired_program could even come from a single paired_example as well
    # Here each paired_program is a dict
    # {
    #   "action_sequence": List[int],
    #   "score": torch.Tensor,
    #   "debug_info": List[Dict] debug_info
    #   "original_tokenoffset": Tuple[int, int],
    #   "paired_tokenoffset": Tuple[int, int],
    #   "paired_idx": the idx of the paired_example (for this batch_idx) for which this program is a parse
    # }
    # Caveat: since we're merging programs from all paired examples, "score" is no longer a prob-dist over this list
    batchidx2paired_programs: Dict[int, List[Dict]] = {}

    for group_idx in range(len(group_idxs)):
        batch_idx = group_idxs[group_idx]
        # Decoding may not have terminated with any completed logical forms, if `num_steps`
        # isn't long enough (or if the model is not trained enough and gets into an
        # infinite action loop).
        if group_idx in best_paired_final_states:
            # For this paired example, going through all states in the beam
            for state in best_paired_final_states[group_idx]:
                indices: List[int] = state.action_history[0]
                all_actions = state.possible_actions[0]
                sequence: List[str] = [all_actions[action][0] for action in indices]
                is_consistent = parser.is_consistent_program(sequence,
                                                             flat_worlds[group_idx],
                                                             flat_label_strings[group_idx])
                if not is_consistent:
                    continue

                # This paired_program is consistent, add to batchidx2paired_programs
                if batch_idx not in batchidx2paired_programs:
                    batchidx2paired_programs[batch_idx] = []

                paired_program_dict = {
                    "action_indices": indices,
                    "action_sequence": sequence,
                    "score": state.score[0],
                    "debug_info":  state.debug_info[0],
                    "original_tokenoffset": flat_original_tokenoffsets[group_idx],
                    "paired_tokenoffset": flat_paired_tokenoffsets[group_idx],
                    "paired_idx": grpidx2batch_paired_idx[group_idx][1],
                    "nt_match": flat_nt_matches[group_idx]
                }
                batchidx2paired_programs[batch_idx].append(paired_program_dict)

    return batchidx2paired_programs