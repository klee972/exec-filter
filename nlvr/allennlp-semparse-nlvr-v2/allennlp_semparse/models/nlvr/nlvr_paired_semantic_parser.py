import logging
from functools import partial
from typing import Any, List, Dict, Tuple, Callable
from overrides import overrides
from collections import defaultdict

import os
import torch

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.archival import load_archive, Archive
from allennlp.models.model import Model
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn import Activation, util

from allennlp_semparse.domain_languages import NlvrLanguageFuncComposition
from allennlp_semparse.fields.production_rule_field import ProductionRule
from allennlp_semparse.models.nlvr.nlvr_semantic_parser import NlvrSemanticParser
from allennlp_semparse.state_machines import BeamSearch
from allennlp_semparse.state_machines.states import GrammarBasedState
from allennlp_semparse.state_machines.trainers import (
    DecoderTrainer,
    ExpectedRiskMinimization,
    MaximumMarginalLikelihood,
)
from allennlp_semparse.common import ParsingError, ExecutionError
from allennlp_semparse.state_machines.transition_functions import BasicTransitionFunction
from allennlp_semparse.models.nlvr.paired_examples_utils import get_programs_for_paired_examples

logger = logging.getLogger(__name__)


@Model.register("nlvr_paired_parser")
class NlvrPairedSemanticParser(NlvrSemanticParser):
    """
    ``NlvrDirectSemanticParser`` is an ``NlvrSemanticParser`` that gets around the problem of lack
    of logical form annotations by maximizing the marginal likelihood of an approximate set of target
    sequences that yield the correct denotation. The main difference between this parser and
    ``NlvrCoverageSemanticParser`` is that while this parser takes the output of an offline search
    process as the set of target sequences for training, the latter performs search during training.

    Parameters
    ----------
    vocab : ``Vocabulary``
        Passed to super-class.
    sentence_embedder : ``TextFieldEmbedder``
        Passed to super-class.
    action_embedding_dim : ``int``
        Passed to super-class.
    encoder : ``Seq2SeqEncoder``
        Passed to super-class.
    attention : ``Attention``
        We compute an attention over the input question at each step of the decoder, using the
        decoder hidden state as the query.  Passed to the TransitionFunction.
    decoder_beam_search : ``BeamSearch``
        Beam search used to retrieve best sequences after training.
    max_decoding_steps : ``int``
        Maximum number of steps for beam search after training.
    dropout : ``float``, optional (default=0.0)
        Probability of dropout to apply on encoder outputs, decoder outputs and predicted actions.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        sentence_embedder: TextFieldEmbedder,
        action_embedding_dim: int,
        encoder: Seq2SeqEncoder,
        attention: Attention,
        beam_size: int,
        # decoder_beam_search: BeamSearch,
        max_decoding_steps: int,
        normalize_beam_score_by_length: bool = False,
        max_num_finished_states: int = None,
        dropout: float = 0.0,
        paired_treshold: float = 0.7,
        initial_mml_model_file: str = None,
    ) -> None:
        super(NlvrPairedSemanticParser, self).__init__(
            vocab=vocab,
            sentence_embedder=sentence_embedder,
            action_embedding_dim=action_embedding_dim,
            encoder=encoder,
            dropout=dropout,
        )
        # self._decoder_trainer = MaximumMarginalLikelihood()
        self._decoder_trainer: DecoderTrainer[
            Callable[[GrammarBasedState], torch.Tensor]
        ] = ExpectedRiskMinimization(
            beam_size=beam_size,
            normalize_by_length=normalize_beam_score_by_length,
            max_decoding_steps=max_decoding_steps,
            max_num_finished_states=max_num_finished_states,
        )

        self._decoder_step = BasicTransitionFunction(
            encoder_output_dim=self._encoder.get_output_dim(),
            action_embedding_dim=action_embedding_dim,
            input_attention=attention,
            activation=Activation.by_name("tanh")(),
            add_action_bias=False,
            dropout=dropout,
        )

        self._beam_search = BeamSearch(beam_size=beam_size)

        # self._decoder_beam_search = decoder_beam_search
        self._max_decoding_steps = max_decoding_steps
        self._action_padding_index = -1

        self._paired_treshold = paired_treshold

        # TODO (pradeep): Checking whether file exists here to avoid raising an error when we've
        # copied a trained ERM model from a different machine and the original MML model that was
        # used to initialize it does not exist on the current machine. This may not be the best
        # solution for the problem.
        if initial_mml_model_file is not None:
            if os.path.isfile(initial_mml_model_file):
                archive = load_archive(initial_mml_model_file)
                logger.info("MML File: {}".format(initial_mml_model_file))
                self._initialize_weights_from_archive(archive)
            else:
                # A model file is passed, but it does not exist. This is expected to happen when
                # you're using a trained ERM model to decode. But it may also happen if the path to
                # the file is really just incorrect. So throwing a warning.
                logger.warning(
                    "MML model file for initializing weights is passed, but does not exist."
                    " This is fine if you're just decoding."
                )

        # Making empty world for parsing utils
        self.world = NlvrLanguageFuncComposition({})

    def _initialize_weights_from_archive(self, archive: Archive) -> None:
        logger.info("Initializing weights from MML model.")
        model_parameters = dict(self.named_parameters())
        archived_parameters = dict(archive.model.named_parameters())
        sentence_embedder_weight = "_sentence_embedder.token_embedder_tokens.weight"
        if (
            sentence_embedder_weight not in archived_parameters
            or sentence_embedder_weight not in model_parameters
        ):
            raise RuntimeError(
                "When initializing model weights from an MML model, we need "
                "the sentence embedder to be a TokenEmbedder using namespace called "
                "tokens."
            )
        for name, weights in archived_parameters.items():
            if name in model_parameters:
                if name == "_sentence_embedder.token_embedder_tokens.weight":
                    # The shapes of embedding weights will most likely differ between the two models
                    # because the vocabularies will most likely be different. We will get a mapping
                    # of indices from this model's token indices to the archived model's and copy
                    # the tensor accordingly.
                    vocab_index_mapping = self._get_vocab_index_mapping(archive.model.vocab)
                    archived_embedding_weights = weights.data
                    new_weights = model_parameters[name].data.clone()
                    for index, archived_index in vocab_index_mapping:
                        new_weights[index] = archived_embedding_weights[archived_index]
                    logger.info(
                        "Copied embeddings of %d out of %d tokens",
                        len(vocab_index_mapping),
                        new_weights.size()[0],
                    )
                else:
                    new_weights = weights.data
                logger.info("Copying parameter %s", name)
                model_parameters[name].data.copy_(new_weights)

    def _get_vocab_index_mapping(self, archived_vocab: Vocabulary) -> List[Tuple[int, int]]:
        vocab_index_mapping: List[Tuple[int, int]] = []
        for index in range(self.vocab.get_vocab_size(namespace="tokens")):
            token = self.vocab.get_token_from_index(index=index, namespace="tokens")
            archived_token_index = archived_vocab.get_token_index(token, namespace="tokens")
            # Checking if we got the UNK token index, because we don't want all new token
            # representations initialized to UNK token's representation. We do that by checking if
            # the two tokens are the same. They will not be if the token at the archived index is
            # UNK.
            if (
                archived_vocab.get_token_from_index(archived_token_index, namespace="tokens")
                == token
            ):
                vocab_index_mapping.append((index, archived_token_index))
        return vocab_index_mapping

    @overrides
    def forward(
        self,  # type: ignore
        sentence: Dict[str, torch.LongTensor],
        worlds: List[List[NlvrLanguageFuncComposition]],
        actions: List[List[ProductionRule]],
        identifier: List[str] = None,
        target_action_sequences: torch.LongTensor = None,
        labels: torch.LongTensor = None,
        paired_masks: List[List[int]] = None,
        paired_identifiers: List[List[str]] = None,
        paired_sentences: Dict[str, torch.LongTensor] = None,
        paired_worlds: List[List[List[NlvrLanguageFuncComposition]]] = None,
        paired_labels: List[List[List[str]]] = None,
        original_tokenoffsets: List[List[Tuple[int, int]]] = None,
        paired_tokenoffsets: List[List[Tuple[int, int]]] = None,
        paired_nt_matches: List[List[bool]] = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Decoder logic for producing type constrained target sequences, trained to maximize marginal
        likelihod over a set of approximate logical forms.
        """
        batch_size = len(worlds)

        initial_rnn_state = self._get_initial_rnn_state(sentence)
        token_ids = util.get_token_ids_from_text_field_tensors(sentence)
        initial_score_list = [token_ids.new_zeros(1, dtype=torch.float) for i in range(batch_size)]
        label_strings = self._get_label_strings(labels) if labels is not None else None
        # TODO (pradeep): Assuming all worlds give the same set of valid actions.
        initial_grammar_state = [
            self._create_grammar_state(worlds[i][0], actions[i]) for i in range(batch_size)
        ]

        initial_state = GrammarBasedState(
            batch_indices=list(range(batch_size)),
            action_history=[[] for _ in range(batch_size)],
            score=initial_score_list,
            rnn_state=initial_rnn_state,
            grammar_state=initial_grammar_state,
            possible_actions=actions,
            extras=label_strings,
        )

        batchidx2paired_programs = None
        if self.training:
            # batchidx: List[paired_program]
            # Here each paired_program is a dict
            # {
            #   "action_sequence": List[int],
            #   "score": torch.Tensor,
            #   "debug_info": List[Dict] debug_info
            #   "original_tokenoffset": Tuple[int, int],
            #   "paired_tokenoffset": Tuple[int, int],
            #   "paired_idx": the idx of the paired_example (for this batch_idx) for which this program is a parse
            # }
            # Caveat: since we're merging programs from all paired examples, "score" is no longer a prob-dist over this
            # list
            batchidx2paired_programs = get_programs_for_paired_examples(
                self,
                actions,
                paired_masks,
                paired_identifiers,
                paired_sentences,
                paired_worlds,
                paired_labels,
                original_tokenoffsets,
                paired_tokenoffsets,
                paired_nt_matches,
                metadata,
            )

        #if not self.training:
            # might be needed for estimating alignment in paired training
        initial_state.debug_info = [[] for _ in range(batch_size)]
        best_final_states = self._beam_search.search(
            num_steps=self._max_decoding_steps,
            initial_state=initial_state,
            transition_function=self._decoder_step,
            keep_final_unfinished_states=False,
        )
        original_outputs = self._get_action_sequences_and_scores_from_final_states(
            batch_size=batch_size, best_final_states=best_final_states
        )
        best_action_sequences: Dict[int, List[List[int]]] = original_outputs[0]
        best_action_scores: Dict[int, List[torch.Tensor]] = original_outputs[1]
        best_debug_infos: Dict[int, List[List[Dict]]] = original_outputs[2]

        finished_costs: Dict[int, List[torch.Tensor]] = self._get_costs_by_batch(
            best_final_states, partial(self._get_state_cost, worlds, batchidx2paired_programs, metadata,
                                       self._paired_treshold)
        )

        loss = initial_state.score[0].new_zeros(1)
        for batch_index in best_action_sequences:
            # Finished model scores are log-probabilities of the predicted sequences. We convert
            # log probabilities into probabilities and re-normalize them to compute expected cost under
            # the distribution approximated by the beam search.
            costs = torch.cat([tensor.view(-1) for tensor in finished_costs[batch_index]])
            logprobs = torch.cat([tensor.view(-1) for tensor in best_action_scores[batch_index]])
            # Unmasked softmax of log probabilities will convert them into probabilities and
            # renormalize them.
            renormalized_probs = util.masked_softmax(logprobs, None)
            loss += renormalized_probs.dot(costs)
        mean_loss = loss / len(best_action_sequences)

        outputs = {"loss": mean_loss}

        if identifier is not None:
            outputs["identifier"] = identifier

        if not self.training:
            # We're testing
            # Action-string sequence, for each program, for each instance. If no program is decoded for an instance,
            # the corresponding instance's List[List[str]] would just be an empty []
            batch_action_strings: List[List[List[str]]] = self._get_action_strings(
                actions, best_action_sequences
            )
            # Denotation, for each world, for each program, for each instance. Similarly, if no program is decoded,
            # the corresponding instance's List[List[str]] would just be an empty []
            batch_denotations: List[List[List[str]]] = self._get_denotations(
                batch_action_strings, worlds
            )
            outputs["best_action_strings"] = batch_action_strings
            outputs["denotations"] = batch_denotations
            batch_action_scores = [
                best_action_scores[i] if i in best_action_scores else [] for i in range(batch_size)
            ]
            outputs["batch_action_scores"] = batch_action_scores

            batch_sequence_is_correct = None
            if label_strings is not None:
                # Prediction correct/incorrect, for each world, for each instance
                batch_sequence_is_correct: List[List[bool]] = self._update_metrics(
                    action_strings=batch_action_strings, worlds=worlds, label_strings=label_strings
                )
                outputs["label_strings"] = label_strings
                outputs["sequence_is_correct"] = batch_sequence_is_correct
                # consistent_programs = []
                # for i in range(batch_size):
                #     cps = []
                #     for (p, s) in zip(batch_action_strings[i], batch_action_scores[i]):
                #         if self.is_consistent_program(p, worlds[i], label_strings[i]):
                #             cps.append((p, s))
                #     import pdb
                #     pdb.set_trace()
                #     consistent_programs.append(cps)

                consistent_programs = [
                    [(worlds[i][0].action_sequence_to_logical_form(p), s) for (p, s) in zip(batch_action_strings[i],
                                                                                            batch_action_scores[i])
                     if self.is_consistent_program(p, worlds[i], label_strings[i])] for i in range(batch_size)
                ]
                outputs["consistent_programs"] = consistent_programs

            if metadata is not None:
                outputs["sentence"] = [x["sentence"] for x in metadata]
                outputs["sentence_tokens"] = [x["sentence_tokens"] for x in metadata]
            outputs["debug_info"] = []
            for i in range(batch_size):
                if i in best_final_states:
                    outputs["debug_info"].append(
                        best_final_states[i][0].debug_info[0]
                    )  # type: ignore
                else:
                    outputs["debug_info"].append([])
            action_mapping = {}
            for batch_index, batch_actions in enumerate(actions):
                for action_index, action in enumerate(batch_actions):
                    action_mapping[(batch_index, action_index)] = action[0]
            outputs["action_mapping"] = action_mapping

        return outputs

    def get_initial_grammar_state(
        self,
        sentence: Dict[str, torch.LongTensor],
        worlds: List[List[NlvrLanguageFuncComposition]],
        label_strings: List[List[str]],
        actions: List[List[ProductionRule]],
    ) -> GrammarBasedState:
        batch_size = len(worlds)
        initial_rnn_state = self._get_initial_rnn_state(sentence)
        token_ids = util.get_token_ids_from_text_field_tensors(sentence)
        initial_score_list = [token_ids.new_zeros(1, dtype=torch.float) for i in range(batch_size)]
        # label_strings = self._get_label_strings(labels) if labels is not None else None
        # Assuming all worlds give the same set of valid actions. Doesn't matter even if worlds[i][0] is padded
        initial_grammar_state = [
            self._create_grammar_state(worlds[i][0], actions[i]) for i in range(batch_size)
        ]

        initial_state = GrammarBasedState(
            batch_indices=list(range(batch_size)),
            action_history=[[] for _ in range(batch_size)],
            score=initial_score_list,
            rnn_state=initial_rnn_state,
            grammar_state=initial_grammar_state,
            possible_actions=actions,
            extras=label_strings,  # This is used later when computing cost-function for ERM
        )
        return initial_state

    def _get_action_sequences_and_scores_from_final_states(
        self,
        batch_size: int,
        best_final_states: Dict[int, List[GrammarBasedState]],
        instance_mask: torch.Tensor = None,
        all_worlds: List[List[NlvrLanguageFuncComposition]] = None,
        all_label_strings: List[List[str]] = None,
        keep_only_consistent_programs: bool = False,
    ) -> Tuple[Dict[int, List[List[int]]], Dict[int, List[torch.Tensor]]]:
        if keep_only_consistent_programs and (all_worlds is None or all_label_strings is None):
            raise NotImplementedError("Need worlds and label_strings to measure consistency")

        instance2action_sequences: Dict[int, List[List[int]]] = {}
        instance2action_scores: Dict[int, List[torch.Tensor]] = {}
        instance2debug_infos: Dict[int, List[List[Dict]]] = {}
        for i in range(batch_size):
            # Decoding may not have terminated with any completed logical forms, if `num_steps`
            # isn't long enough (or if the model is not trained enough and gets into an
            # infinite action loop).
            if i in best_final_states:
                if instance_mask is not None and instance_mask[i] == 0:
                    # most likely a paired instance is masked
                    continue
                action_indices: List[List[int]] = []
                action_scores: List[torch.Tensor] = []
                debug_infos: List[List[Dict]] = []
                for state in best_final_states[i]:
                    if keep_only_consistent_programs:
                        indices: List[str] = state.action_history[0]
                        all_actions = state.possible_actions[0]
                        sequence: List[str] = [all_actions[action][0] for action in indices]
                        is_consistent = self.is_consistent_program(sequence,
                                                                   all_worlds[i],
                                                                   all_label_strings[i])
                        if not is_consistent:
                            continue
                    # Going over all states for an instance to get all finished-programs in the beam
                    # since search is finished only get the top-action history
                    action_indices.append(state.action_history[0])
                    action_scores.append(state.score[0])
                    debug_infos.append(state.debug_info[0])
                if action_indices:
                    instance2action_sequences[i] = action_indices
                    instance2action_scores[i] = action_scores
                    instance2debug_infos[i] = debug_infos
        return instance2action_sequences, instance2action_scores, instance2debug_infos

    def _update_metrics(
        self,
        action_strings: List[List[List[str]]],
        worlds: List[List[NlvrLanguageFuncComposition]],
        label_strings: List[List[str]],
    ) -> None:
        # TODO(pradeep): Move this to the base class.
        # TODO(pradeep): Using only the best decoded sequence. Define metrics for top-k sequences?
        batch_size = len(worlds)
        batch_sequence_is_correct = []
        for i in range(batch_size):
            instance_action_strings = action_strings[i]
            instance_worlds = worlds[i]
            sequence_is_correct = [False]
            topk_correct = False
            if instance_action_strings:
                instance_label_strings = label_strings[i]
                # Taking only the best sequence.
                sequence_is_correct = self._check_denotation(
                    instance_action_strings[0], instance_label_strings, instance_worlds
                )
                topk_programs_correctness = [
                    self._check_denotation(action_strings, instance_label_strings, instance_worlds)
                    for action_strings in instance_action_strings
                ]
                if any(all(x) for x in topk_programs_correctness):
                    topk_correct = True
            else:
                # No program was decoded for this instance, therefore, mark denotation is incorrect for all worlds
                # world can be none in case of padding
                num_worlds = sum([1 for world in instance_worlds if world is not None])
                sequence_is_correct = [False] * num_worlds
            for correct_in_world in sequence_is_correct:
                self._denotation_accuracy(1 if correct_in_world else 0)
            self._consistency(1 if all(sequence_is_correct) else 0)
            self._topk_consistency(1 if topk_correct else 0)
            batch_sequence_is_correct.append(sequence_is_correct)
        return batch_sequence_is_correct

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        k = self._beam_search._beam_size
        return {
            "denotation_accuracy": self._denotation_accuracy.get_metric(reset),
            "consistency": self._consistency.get_metric(reset),
            f"top{k}_const": self._topk_consistency.get_metric(reset),
        }

    def _get_state_cost(
        self,
        batch_worlds: List[List[NlvrLanguageFuncComposition]],
        batchidx2paired_programs: Dict[int, List[Dict]],
        metadata: List[Dict],
        paired_treshold: float,
        state: GrammarBasedState,
    ) -> torch.Tensor:
        """
        Return the cost of a finished state. Since it is a finished state, the group size will be
        1, and hence we'll return just one cost.

        The ``batch_worlds`` parameter here is because we need the world to check the denotation
        accuracy of the action sequence in the finished state.  Instead of adding a field to the
        ``State`` object just for this method, we take the ``World`` as a parameter here.

        ``paired_examples_output`` is a Dict with keys:
        "mask": `torch.Tensor` batch sized tensor masking instances without a paired example
        "action_sequences": Dict[int, List[List[int]]] map from batch_index to all action-sequences decoded in the beam
        "action_scores": Dict[int, List[torch.Tensor]] map from batch_index to logprob of action-sequences in the beam
        """
        if not state.is_finished():
            raise RuntimeError("_get_state_cost() is not defined for unfinished states!")

        batch_index = state.batch_indices[0]
        instance_worlds: List[NlvrLanguageFuncComposition] = batch_worlds[batch_index]
        # Using any tensor (state.score[0]) to create a ones of size=1
        denotation_cost = state.score[0].new_ones(1)
        # TODO(nitish) this shouldn't be an issue; probably some issue with function-composition execution
        try:
            sequence_is_correct: List[bool] = self._check_state_denotations(state, instance_worlds)
        except: # (ParsingError, ExecutionError, TypeError):
            # logger.warning("Failed to execute program")
            num_worlds = sum(world is not None for world in instance_worlds)
            sequence_is_correct = [False] * num_worlds

        if state.extras is None or all(sequence_is_correct):
            # first condition is illogical; copied from Coverage-based parser.
            # Zero-cost if sequence is consistent
            cost = denotation_cost - 1.0
        else:
            # Cost since sequence is not consistent
            cost = denotation_cost   # + 1.0

        if batchidx2paired_programs is None:
            # In non-training mode
            return cost

        if batch_index not in batchidx2paired_programs:
            return cost

        # if paired_mask == 0 or batch_index not in paired_examples_output["action_sequences"]:
        #     # This training instance doesn't have paired example OR
        #     # no programs were decoded for the paired example
        #     return cost

        # Assuming global actions only shared by all
        all_actions = state.possible_actions[0]

        original_action_sequence: List[int] = state.action_history[0]
        original_actions: List[str] = [all_actions[action][0] for action in original_action_sequence]
        original_debug_info: List[Dict] = state.debug_info[0]

        # All these paired programs ARE CONSISTENT
        paired_action_sequences: List[List[int]] = [p["action_indices"] for p in batchidx2paired_programs[batch_index]]
        paired_action_scores: List[torch.Tensor] = [p["score"] for p in batchidx2paired_programs[batch_index]]
        logprobs = torch.cat([tensor.view(-1) for tensor in paired_action_scores])
        # TODO(nitish): Caveat: programs from different paired_examples are merged. we are normalizing probabilities
        #  across examples below; which is not really probability anymore. E.g. logprobs could be [log(0.9), log(0.9)]
        #  which will get converted to [0.5, 0.5].
        #  Another caveat: if all consistent paired programs have very prediction proabability (logprobs = [log(0.01]),
        #  they'll get re-normalized to ([1.0]) high values which might make paired-cost less trustworthy
        normalized_paired_action_probs = util.masked_softmax(logprobs, None)
        # These are raw program probabilites as predicted by decoder for the original paired example. This tensor is not
        # a distribution.
        # paired_action_probs = torch.exp(logprobs)

        paired_debug_infos: List[List[Dict]] = [p["debug_info"] for p in batchidx2paired_programs[batch_index]]
        # (start, end) both _inclusive_
        original_tokenoffsets: List[Tuple[int, int]] = [p["original_tokenoffset"] for p in
                                                        batchidx2paired_programs[batch_index]]
        # TODO(nitish): Taking [0] original_tokenoffset since all of them should be equal. add check?
        original_tokenoffset = original_tokenoffsets[0]
        paired_tokenoffsets: List[Tuple[int, int]] = [p["paired_tokenoffset"] for p in
                                                      batchidx2paired_programs[batch_index]]
        paired_nt_matches: List[bool] = [p["nt_match"] for p in batchidx2paired_programs[batch_index]]

        num_paired_progs = len(batchidx2paired_programs[batch_index])
        # metadata: Dict = paired_examples_output["metadata"][batch_index]

        original_relevant_decoding_steps, orig_alignment_scores = self.get_relevant_decoding_steps(
            original_tokenoffset, original_debug_info, threshold=paired_treshold)
        relevant_actions = [original_actions[x] for x in original_relevant_decoding_steps]
        # import pdb
        # print("\n--------")
        # print(metadata[batch_index])
        # print(batch_worlds[0][0].action_sequence_to_logical_form(original_actions))
        # print(original_relevant_decoding_steps)
        # print(relevant_actions)
        # print("\npaired action .... ")
        relevant_nonterminal_actions = [x for x in relevant_actions
                                        if x not in instance_worlds[0].colornumsize_productions]

        share_ratios = [0.0 for _ in range(num_paired_progs)]
        if relevant_actions:
            for pidx in range(num_paired_progs):
                p_action_seq = paired_action_sequences[pidx]
                p_actions: List[str] = [all_actions[action][0] for action in p_action_seq]
                p_debug_info = paired_debug_infos[pidx]
                nt_match: bool = paired_nt_matches[pidx]
                p_relevant_decoding_steps, p_alignment_scores = self.get_relevant_decoding_steps(
                    paired_tokenoffsets[pidx], p_debug_info, threshold=paired_treshold)
                p_relevant_actions = []
                for stepnum in p_relevant_decoding_steps:
                    p_relevant_actions.append(p_actions[stepnum])

                p_relevant_nonterminal_actions = [x for x in p_relevant_actions
                                                  if x not in instance_worlds[0].colornumsize_productions]

                if nt_match:
                    share_ratio = self.compute_action_share_f1(relevant_nonterminal_actions,
                                                               p_relevant_nonterminal_actions)
                else:
                    share_ratio = self.compute_action_share_f1(relevant_actions, p_relevant_actions)

                # # Score between [0, 1], ratio of original relevant actions found in paired relevant actions
                # num_common_actions = len(self.common_elements(relevant_actions, p_relevant_actions))
                # share_ratio_precision = 0.0
                # if len(relevant_actions) > 0:
                #     share_ratio_precision = float(num_common_actions)/len(relevant_actions)
                # share_ratio_recall = 0.0
                # if len(p_relevant_actions) > 0:
                #     share_ratio_recall = float(num_common_actions)/len(p_relevant_actions)
                # share_ratio = 0.0
                # if (share_ratio_precision + share_ratio_recall) > 0.0:
                #     share_ratio = (2.0*share_ratio_precision*share_ratio_recall)/(share_ratio_precision +
                #                                                                   share_ratio_recall)
                share_ratios[pidx] = share_ratio
                # print(batch_worlds[0][0].action_sequence_to_logical_form(p_actions))
                # print(p_relevant_decoding_steps)
                # print(p_relevant_actions)
                # print(share_ratio)
                # pdb.set_trace()

                # print("Paired relevant: {}".format(p_relevant_actions))
                # print(share_ratio)
            share_ratios = normalized_paired_action_probs.new_tensor(share_ratios)

            # TODO(nitish): TYPE 1 - use normalized program probs and compute expected share-score
            # Within range [0, 1], 1 indicating maximum sharing between original and paired program
            share_score = normalized_paired_action_probs.dot(share_ratios)
            # print(share_ratios)
            # print(normalized_paired_action_probs)
            # print()

            paired_cost = 1.0 - share_score

            # print(share_score)
            # print(paired_cost)
            # pdb.set_trace()
        else:
            paired_cost = state.score[0].new_zeros(1)

        consistency_cost = cost
        cost = cost + paired_cost
        # print("consistency cost: {}".format(consistency_cost))
        # print("pairedalign cost: {}".format(paired_cost))
        # print("totalprogrm cost: {}".format(cost))
        # print("---------------")

        """
        # This is the number of items on the agenda that we want to see in the decoded sequence.
        # We use this as the denotation cost if the path is incorrect.
        # Note: If we are penalizing the model for producing non-agenda actions, this is not the
        # upper limit on the checklist cost. That would be the number of terminal actions.
        denotation_cost = torch.sum(state.checklist_state[0].checklist_target.float())
        checklist_cost = self._checklist_cost_weight * checklist_cost
        # TODO (pradeep): The denotation based cost below is strict. May be define a cost based on
        # how many worlds the logical form is correct in?
        # extras being None happens when we are testing. We do not care about the cost
        # then.  TODO (pradeep): Make this cleaner.
        if state.extras is None or all(self._check_state_denotations(state, instance_worlds)):
            cost = checklist_cost
        else:
            cost = checklist_cost + (1 - self._checklist_cost_weight) * denotation_cost
        """
        return cost

    @staticmethod
    def compute_action_share_f1(orig_actions, paired_actions):
        num_common_actions = len(NlvrPairedSemanticParser.common_elements(orig_actions, paired_actions))
        share_ratio_precision = 0.0
        if len(orig_actions) > 0:
            share_ratio_precision = float(num_common_actions) / len(orig_actions)
        share_ratio_recall = 0.0
        if len(paired_actions) > 0:
            share_ratio_recall = float(num_common_actions) / len(paired_actions)
        share_f1 = 0.0
        if (share_ratio_precision + share_ratio_recall) > 0.0:
            share_f1 = (2.0 * share_ratio_precision * share_ratio_recall) / (share_ratio_precision +
                                                                             share_ratio_recall)
        return share_f1

    def get_relevant_decoding_steps(self,
        tokenoffsets: Tuple[int, int],
        debug_info: List[Dict],
        threshold: float = 0.4
    ):
        """Given a span token offset, return a list of action indices that are generated by those tokens.

        An action is estimated to be generated by a span if the total attention mass from the tokens to that action
        is above a certain threshold:
            s(a_i) = sum(qattn[i][s:e])
        where, a_i is the action-index in question, qattn is a list of question-attentions for the actions, (s,e) is
        the token_offset
        """
        relevant_decoding_steps = []
        scores = []
        start, end = tokenoffsets   # _inclusive_
        for i, info_dict in enumerate(debug_info):
            span_to_action_attention = info_dict["question_attention"][start:end+1]
            score = torch.sum(span_to_action_attention)
            scores.append(score)
            if score > threshold:
                relevant_decoding_steps.append(i)
        return relevant_decoding_steps, scores

    @staticmethod
    def _get_costs_by_batch(
        # states: List[GrammarBasedState],
        final_states: Dict[int, List[GrammarBasedState]],
        cost_function: Callable[[GrammarBasedState], torch.Tensor],
    ) -> Dict[int, List[torch.Tensor]]:
        batch_costs: Dict[int, List[torch.Tensor]] = defaultdict(list)

        for batch_index, states in final_states.items():
            for state in states:
                cost = cost_function(state)
                batch_costs[batch_index].append(cost)
        return batch_costs

    # @staticmethod
    # def common_elements(l1, l2):
    #     """Returns elements in l1 that appear in l2. Multiple copies of same element are considered separately;
    #        each copy in l1 needs to be aligned in l2 to be considered a match.
    #        https://stackoverflow.com/a/22542432
    #     """
    #     return [e for e in l1 if e in l2 and (l2.pop(l2.index(e)) or True)]

    @staticmethod
    def common_elements(l1, l2):
        """Returns elements in l1 that appear in l2. Multiple copies of same element are considered separately;
           each copy in l1 needs to be aligned in l2 to be considered a match.
           https://stackoverflow.com/a/22542432
        """
        dict1 = {}
        for action in l1:
            if action not in dict1:
                dict1[action] = 0
            dict1[action] += 1
        dict2 = {}
        for action in l2:
            if action not in dict2:
                dict2[action] = 0
            dict2[action] += 1

        common_actions = []
        for action, count1 in dict1.items():
            count2 = dict2.get(action, 0)
            for i in range(min(count1, count2)):
                common_actions.append(action)

        return common_actions

    @staticmethod
    def is_consistent_program(action_sequence: List[str],
                              worlds: List[NlvrLanguageFuncComposition],
                              labels: List[str]):
        is_correct: List[bool] = NlvrSemanticParser._check_denotation(
            action_sequence=action_sequence, labels=labels, worlds=worlds)
        return all(is_correct)
