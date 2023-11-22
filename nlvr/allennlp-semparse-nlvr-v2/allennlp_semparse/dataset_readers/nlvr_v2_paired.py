from typing import Any, Dict, List, Tuple
import json
import logging
import numpy as np

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader
from allennlp.data.fields import Field, TextField, ListField, IndexField, LabelField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer, Token

from allennlp_semparse.domain_languages import NlvrLanguageFuncComposition
from allennlp_semparse.domain_languages.nlvr_language_v2 import Box
from allennlp_semparse.fields import ProductionRuleField


logger = logging.getLogger(__name__)


@DatasetReader.register("nlvr_v2_paired")
class NlvrV2PairedDatasetReader(DatasetReader):
    """
    ``DatasetReader`` for the NLVR domain. In addition to the usual methods for reading files and
    instances from text, this class contains a method for creating an agenda of actions that each
    sentence triggers, if needed. Note that we deal with the version of the dataset with structured
    representations of the synthetic images instead of the actual images themselves.

    We support multiple data formats here:
    1) The original json version of the NLVR dataset (http://lic.nlp.cornell.edu/nlvr/) where the
    format of each line in the jsonl file is
    ```
    "sentence": <sentence>,
    "label": <true/false>,
    "identifier": <id>,
    "evals": <dict containing all annotations>,
    "structured_rep": <list of three box representations, where each box is a list of object
    representation dicts, containing fields "x_loc", "y_loc", "color", "type", "size">
    ```

    2) A grouped version (constructed using ``scripts/nlvr/group_nlvr_worlds.py``) where we group
    all the worlds that a sentence appears in. We use the fields ``sentence``, ``label`` and
    ``structured_rep``.  And the format of the grouped files is
    ```
    "sentence": <sentence>,
    "labels": <list of labels corresponding to worlds the sentence appears in>
    "identifier": <id that is only the prefix from the original data>
    "worlds": <list of structured representations>
    ```

    3) A processed version that contains action sequences that lead to the correct denotations (or
    not), using some search. This format is very similar to the grouped format, and has the
    following extra field

    ```
    "correct_sequences": <list of lists of action sequences corresponding to logical forms that
    evaluate to the correct denotations>
    ```

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.
    tokenizer : ``Tokenizer`` (optional)
        The tokenizer used for sentences in NLVR. Default is ``SpacyTokenizer``
    sentence_token_indexers : ``Dict[str, TokenIndexer]`` (optional)
        Token indexers for tokens in input sentences.
        Default is ``{"tokens": SingleIdTokenIndexer()}``
    nonterminal_indexers : ``Dict[str, TokenIndexer]`` (optional)
        Indexers for non-terminals in production rules. The default is to index terminals and
        non-terminals in the same way, but you may want to change it.
        Default is ``{"tokens": SingleIdTokenIndexer("rule_labels")}``
    terminal_indexers : ``Dict[str, TokenIndexer]`` (optional)
        Indexers for terminals in production rules. The default is to index terminals and
        non-terminals in the same way, but you may want to change it.
        Default is ``{"tokens": SingleIdTokenIndexer("rule_labels")}``
    output_agendas : ``bool`` (optional)
        If preparing data for a trainer that uses agendas, set this flag and the datset reader will
        output agendas.
    """

    def __init__(
        self,
        lazy: bool = False,
        tokenizer: Tokenizer = None,
        sentence_token_indexers: Dict[str, TokenIndexer] = None,
        nonterminal_indexers: Dict[str, TokenIndexer] = None,
        terminal_indexers: Dict[str, TokenIndexer] = None,
        output_agendas: bool = False,
        mode: str = "test",
        **kwargs,
    ) -> None:
        super().__init__(lazy, **kwargs)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._sentence_token_indexers = sentence_token_indexers or {
            "tokens": SingleIdTokenIndexer()
        }
        self._nonterminal_indexers = nonterminal_indexers or {
            "tokens": SingleIdTokenIndexer("rule_labels")
        }
        self._terminal_indexers = terminal_indexers or {
            "tokens": SingleIdTokenIndexer("rule_labels")
        }
        self._output_agendas = output_agendas
        self._mode = mode
        assert self._mode in ["train", "test"]
        self.num_paired_examples = 0

    @overrides
    def _read(self, file_path: str):
        self.num_paired_examples = 0
        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                data = json.loads(line)
                sentence = data["sentence"]
                identifier = data["identifier"] if "identifier" in data else data["id"]
                if "worlds" in data:
                    # This means that we are reading grouped nlvr data. There will be multiple
                    # worlds and corresponding labels per sentence.
                    labels = data["labels"]
                    structured_representations = data["worlds"]
                else:
                    # We will make lists of labels and structured representations, each with just
                    # one element for consistency.
                    labels = [data["label"]]
                    structured_representations = [data["structured_rep"]]

                # Not all training instances would have this
                target_sequences: List[List[str]] = data.get("correct_sequences", None)

                # Not all training instances would have this
                paired_examples: List[Dict] = data.get("paired_examples", None)

                instance = self.text_to_instance(
                    sentence,
                    structured_representations,
                    paired_examples,
                    labels,
                    target_sequences,
                    identifier,
                )
                if instance is not None:
                    yield instance
        logger.info("Number of paired examples: {}".format(self.num_paired_examples))

    @overrides
    def text_to_instance(
        self,  # type: ignore
        sentence: str,
        structured_representations: List[List[List[JsonDict]]],
        paired_examples: List[Dict] = None,
        labels: List[str] = None,
        target_sequences: List[List[str]] = None,
        identifier: str = None,
    ) -> Instance:
        """
        Parameters
        ----------
        sentence : ``str``
            The query sentence.
        structured_representations : ``List[List[List[JsonDict]]]``
            A list of Json representations of all the worlds. See expected format in this class' docstring.
        labels : ``List[str]`` (optional)
            List of string representations of the labels (true or false) corresponding to the
            ``structured_representations``. Not required while testing.
        target_sequences : ``List[List[str]]`` (optional)
            List of target action sequences for each element which lead to the correct denotation in
            worlds corresponding to the structured representations.
        identifier : ``str`` (optional)
            The identifier from the dataset if available.
        """
        worlds = []
        for structured_representation in structured_representations:
            boxes = {
                Box(object_list, box_id)
                for box_id, object_list in enumerate(structured_representation)
            }
            worlds.append(NlvrLanguageFuncComposition(boxes))
        tokenized_sentence: List[Token] = self._tokenizer.tokenize(sentence)
        sentence_field = TextField(tokenized_sentence, self._sentence_token_indexers)
        production_rule_fields: List[Field] = []
        instance_action_ids: Dict[str, int] = {}
        # TODO(pradeep): Assuming that possible actions are the same in all worlds. This may change
        # later.
        for production_rule in worlds[0].all_possible_productions():
            instance_action_ids[production_rule] = len(instance_action_ids)
            field = ProductionRuleField(production_rule, is_global_rule=True)
            production_rule_fields.append(field)
        action_field = ListField(production_rule_fields)
        worlds_field = ListField([MetadataField(world) for world in worlds])
        metadata: Dict[str, Any] = {
            "sentence_tokens": [x.text for x in tokenized_sentence],
            "sentence": sentence,
        }
        fields: Dict[str, Field] = {
            "sentence": sentence_field,
            "worlds": worlds_field,
            "actions": action_field,
            "metadata": MetadataField(metadata),
        }
        if identifier is not None:
            fields["identifier"] = MetadataField(identifier)
            metadata["identifier"] = identifier
        # Depending on the type of supervision used for training the parser, we may want either
        # target action sequences or an agenda in our instance. We check if target sequences are
        # provided, and include them if they are. If not, we'll get an agenda for the sentence, and
        # include that in the instance.
        if target_sequences:
            action_sequence_fields: List[Field] = []
            for target_sequence in target_sequences:
                index_fields = ListField(
                    [
                        IndexField(instance_action_ids[action], action_field)
                        for action in target_sequence
                    ]
                )
                action_sequence_fields.append(index_fields)
                # TODO(pradeep): Define a max length for this field.
            fields["target_action_sequences"] = ListField(action_sequence_fields)
        else:
            # Make a single padded target sequence
            index_fields = ListField([IndexField(-1, action_field)])   # padded sequence of len=1
            action_sequence_fields: List[Field] = [index_fields]  # List of padded sequence
            fields["target_action_sequences"] = ListField(action_sequence_fields)

        # elif self._output_agendas:
        #     # TODO(pradeep): Assuming every world gives the same agenda for a sentence. This is true
        #     # now, but may change later too.
        #     agenda = worlds[0].get_agenda_for_sentence(sentence)
        #     assert agenda, "No agenda found for sentence: %s" % sentence
        #     # agenda_field contains indices into actions.
        #     agenda_field = ListField(
        #         [IndexField(instance_action_ids[action], action_field) for action in agenda]
        #     )
        #     fields["agenda"] = agenda_field

        if labels:
            labels_field = ListField(
                [LabelField(label, label_namespace="denotations") for label in labels]
            )
            fields["labels"] = labels_field

        paired_identifiers: List[str] = []
        paired_sentences: List[str] = []
        paired_structured_representations: List[List[Dict]] = []
        paired_labels: List[List[str]] = []
        original_charoffsets: List[Tuple[int, int]] = []
        paired_charoffsets: List[Tuple[int, int]] = []
        paired_nt_matches: List[bool] = []
        paired_masks: List[int] = []
        if paired_examples is not None and paired_examples:
            # Process paired example:
            for paired_example in paired_examples:
                paired_identifiers.append(paired_example["identifier"])
                paired_sentences.append(paired_example["sentence"])
                paired_structured_representations.append(paired_example["structured_representations"])
                paired_labels.append(paired_example["labels"])
                original_charoffsets.append(paired_example["orig_charoffsets"])
                paired_charoffsets.append(paired_example["paired_charoffsets"])
                paired_nt_matches.append(paired_example.get("nt_match", False))
                paired_masks.append(1)
                self.num_paired_examples += 1
        else:
            paired_identifiers.append("N/A")
            paired_sentences.append("NONE")
            paired_structured_representations.append([])
            paired_labels.append([])
            original_charoffsets.append((-1, -1))
            paired_charoffsets.append((-1, -1))
            paired_nt_matches.append(False)
            paired_masks.append(0)

        # paired_mask_field = ArrayField(np.array(paired_masks))
        paired_masks_field = MetadataField(paired_masks)
        paired_sentences_fieldlist = []
        tokenized_paired_sentences = []     # needed later to convert char-offsets to token-offsets

        for paired_sentence in paired_sentences:
            tokenized_paired_sentence: List[Token] = self._tokenizer.tokenize(paired_sentence)
            paired_sentence_field = TextField(
                tokenized_paired_sentence, self._sentence_token_indexers
            )
            tokenized_paired_sentences.append(tokenized_paired_sentence)
            paired_sentences_fieldlist.append(paired_sentence_field)
        paired_sentences_field = ListField(paired_sentences_fieldlist)

        paired_worlds: List[List[NlvrLanguageFuncComposition]] = []
        for ps_structured_representations in paired_structured_representations:
            # ps_structured_representations is a List[Dict]: List of worlds for one paired sentence
            ps_worlds: List[NlvrLanguageFuncComposition] = structured_representations_to_worlds(
                ps_structured_representations)
            paired_worlds.append(ps_worlds)
            # if ps_worlds:
            #     paired_worlds.append(ps_worlds)
            # else:
            #     # if ps_worlds is empty, means ps_structured_representations is empty =>
        paired_worlds_field = MetadataField(paired_worlds)

        # if not paired_worlds:
        #     # Pad with an empty world
        #     paired_worlds.append(NlvrLanguageFuncComposition(set({})))
        #     paired_labels.append("false")
        # paired_worlds_field = ListField([MetadataField(world) for world in paired_worlds])

        # paired_labels_listfield = []
        # for ps_labels in paired_labels:
        #     ps_labels_field = ListField(
        #         [LabelField(label, label_namespace="denotations") for label in ps_labels]
        #     )
        #     print(ps_labels_field)
        #     paired_labels_listfield.append(ps_labels_field)
        # paired_labels_field = ListField(paired_labels_listfield)
        paired_labels_field = MetadataField(paired_labels)

        # Token offset end is _inclusive_
        original_tokenidxs: List[Tuple[int, int]] = [
            (charidx_to_tokenidx(tokenized_sentence, charoffsets[0]),
             charidx_to_tokenidx(tokenized_sentence, charoffsets[1]))
            for charoffsets in original_charoffsets
        ]
        original_tokenoffsets_field = MetadataField(original_tokenidxs)
        paired_tokenidxs: List[Tuple[int, int]] = [
            (charidx_to_tokenidx(tokenized_paired_sentences[i], charoffsets[0]),
             charidx_to_tokenidx(tokenized_paired_sentences[i], charoffsets[1]))
            for i, charoffsets in enumerate(paired_charoffsets)
        ]
        paired_tokenoffsets_field = MetadataField(paired_tokenidxs)

        paired_fields: Dict[str, Field] = {
            "paired_masks": paired_masks_field,
            "paired_identifiers": MetadataField(paired_identifiers),
            "paired_sentences": paired_sentences_field,
            "paired_worlds": paired_worlds_field,
            "paired_labels": paired_labels_field,
            "original_tokenoffsets": original_tokenoffsets_field,
            "paired_tokenoffsets": paired_tokenoffsets_field,
            "paired_nt_matches": MetadataField(paired_nt_matches)
        }
        fields.update(paired_fields)
        # Even though metadata has been added to fields, since it's dict,
        # it should still get updated
        metadata["paired_sentences"] = paired_sentences
        # metadata["paired_sentence_tokens"] = [x.text for x in tokenized_paired_sentence]

        return Instance(fields)


def charidx_to_tokenidx(tokens: List[Token], charidx):
    """Get token idx from char idx (if char end, it is _exclusive_).
    If charidx falls between two tokens' char indices, we will choose the later token
    """
    if charidx == -1:
        # for a padded instance
        return -1
    starts = [t.idx for t in tokens]
    ends = [t.idx + len(t.text) for t in tokens]  # _exclusive_

    token_idx = -1
    for tidx, (s, e) in enumerate(zip(starts, ends)):
        if e < charidx:
            # The current token ends before the charidx
            continue
        else:
            # The token ends at the charidx or ends after it, so choose this token
            token_idx = tidx
            break
    if token_idx == -1:
        token_idx = len(tokens) - 1
    return token_idx


def structured_representations_to_worlds(
    structured_representations: List[Dict],
) -> List[NlvrLanguageFuncComposition]:
    worlds = []
    for structured_representation in structured_representations:
        boxes = {
            Box(object_list, box_id) for box_id, object_list in enumerate(structured_representation)
        }
        worlds.append(NlvrLanguageFuncComposition(boxes))
    return worlds


def candidate_sequences_to_action_sequences_field(
    target_sequences: List[List[str]], instance_action_ids: Dict[str, int], action_field
) -> List[Field]:
    # Each field would be a target_sequence
    action_sequence_fields: List[Field] = []
    for target_sequence in target_sequences:
        # List field representing the actions in the target sequence
        index_fields = ListField(
            [IndexField(instance_action_ids[action], action_field) for action in target_sequence]
        )
        action_sequence_fields.append(index_fields)
    return action_sequence_fields
