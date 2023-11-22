import json

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


def round_all(stuff, prec):
    """ Round all the number elems in nested stuff. """
    if isinstance(stuff, list):
        return [round_all(x, prec) for x in stuff]
    if isinstance(stuff, tuple):
        return tuple(round_all(x, prec) for x in stuff)
    if isinstance(stuff, float):
        return round(float(stuff), prec)
    if isinstance(stuff, dict):
        d = {}
        for k, v in stuff.items():
            d[k] = round(v, prec)
        return d
    else:
        return stuff


@Predictor.register("nlvr-parser")
class NlvrParserPredictor(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        if "worlds" in json_dict:
            # This is grouped data
            worlds = json_dict["worlds"]
            if isinstance(worlds, str):
                worlds = json.loads(worlds)
        else:
            structured_rep = json_dict["structured_rep"]
            if isinstance(structured_rep, str):
                structured_rep = json.loads(structured_rep)
            worlds = [structured_rep]
        identifier = json_dict["identifier"] if "identifier" in json_dict else None
        instance = self._dataset_reader.text_to_instance(
            sentence=sentence,  # type: ignore
            structured_representations=worlds,
            identifier=identifier,
        )
        return instance

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        if "identifier" in outputs:
            # Returning CSV lines for official evaluation
            identifier = outputs["identifier"]
            # Denotation, for each program, for each world --- List[List[str]]
            # Note: if no programs were decoded for this example, this list would be empty
            denotations = outputs["denotations"]
            if denotations:
                denotation = denotations[0][0]
            else:
                denotation = "NULL"
            return f"{identifier},{denotation}\n"
        else:
            return json.dumps(outputs) + "\n"


def get_token_attentions_string(tokens, attentions):
    """Make a human-readable string of tokens and their predicted attentions."""
    output_str = ""
    attentions = attentions[: len(tokens)]  # attention can be padded to a longer length
    for token, attn in zip(tokens, attentions):
        output_str += f"{token}({str(attn)}) "
    return output_str.strip()


def get_influential_tokens(tokens, attentions, threshold):
    """Return a list of tokens that are influential, i.e., attention >= threshold.
    The returned tokens list is the same size as input tokens, only that irrelevant tokens are _ (masked)
    """
    output_tokens = []
    attentions = attentions[: len(tokens)]  # attention can be padded to a longer length
    for token, attn in zip(tokens, attentions):
        if attn >= threshold:
            output_tokens.append(token)
        else:
            output_tokens.append("_")
    return output_tokens


@Predictor.register("nlvr-parser-visualize")
class NlvrVisualizePredictor(NlvrParserPredictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        if "worlds" in json_dict:
            # This is grouped data
            worlds = json_dict["worlds"]
            labels = json_dict["labels"]
            if isinstance(worlds, str):
                worlds = json.loads(worlds)
        else:
            structured_rep = json_dict["structured_rep"]
            labels = [json_dict["label"]]
            if isinstance(structured_rep, str):
                structured_rep = json.loads(structured_rep)
            worlds = [structured_rep]
        identifier = json_dict["identifier"] if "identifier" in json_dict else None
        instance = self._dataset_reader.text_to_instance(
            sentence=sentence,  # type: ignore
            structured_representations=worlds,
            identifier=identifier,
            labels=labels,
        )
        return instance

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        identifier = outputs.get("identifier", "N/A")
        sentence = outputs["sentence"]
        sentence_tokens = outputs["sentence_tokens"]
        best_action_strings = outputs["best_action_strings"]
        if best_action_strings:
            best_action_strings = best_action_strings[0]
        debug_info = outputs["debug_info"]
        # List of actions; each is a dict with keys: "predicted_action", "considered_actions", "action_probabilities",
        # "question_attention". List is empty in case a parse is not decoded
        predicted_actions = outputs["predicted_actions"]
        action_strings = [a["predicted_action"] for a in predicted_actions]
        question_attentions = [a["question_attention"] for a in predicted_actions]
        question_attentions = round_all(question_attentions, 3)

        # List of "{action} {attended_tokens}" strings -- constructed by selecting
        # influential tokens for an action by thresholding the attention
        actions_w_attended_tokens = []
        for action, attentions in zip(action_strings, question_attentions):
            imp_tokens = get_influential_tokens(sentence_tokens, attentions, threshold=0.1)
            imp_tokens_string = " ".join(imp_tokens)
            actions_w_attended_tokens.append(f"[{action}] --  {imp_tokens_string}")
            # utterance_attention_string = get_token_attentions_string(sentence_tokens, attentions)
            # action_w_attention.append(action)
            # action_w_attention.append(utterance_attention_string)

        label_strings = outputs["label_strings"]
        # All logical-forms in the beam
        logical_forms = outputs["logical_form"]
        best_logical_form = logical_forms[0] if logical_forms else ""
        denotations = outputs["denotations"]
        # Taking denotations for top-scoring program
        if denotations:
            denotations = denotations[0]
        sequence_is_correct = outputs.get("sequence_is_correct", None)

        consistent = None
        if sequence_is_correct is not None:
            consistent = all(sequence_is_correct)

        output_dict = {
            "identifier": identifier,
            "sentence": sentence,
            # "best_action_strings": best_action_strings,
            "best_action_strings": actions_w_attended_tokens,
            "best_logical_form": best_logical_form,
            # "best_logical_forms": logical_form,
            "label_strings": label_strings,
            "denotations": denotations,
            "sequence_is_correct": sequence_is_correct,
            "consistent": consistent,
            "consistent_programs": outputs["consistent_programs"]
        }

        output_str = json.dumps(output_dict, indent=2) + "\n"

        return output_str


@Predictor.register("nlvr-parser-predictions")
class NlvrPredictionPredictor(NlvrVisualizePredictor):
    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        identifier = outputs.get("identifier", "N/A")
        sentence = outputs["sentence"]
        best_action_strings = outputs["best_action_strings"]
        if best_action_strings:
            best_action_strings = best_action_strings[0]

        label_strings = outputs["label_strings"]
        # All logical-forms in the beam
        logical_form = outputs["logical_form"]
        denotations = outputs["denotations"]
        # Taking denotations for top-scoring program
        if denotations:
            denotations = denotations[0]
        sequence_is_correct = outputs.get("sequence_is_correct", None)

        consistent = None
        if sequence_is_correct is not None:
            consistent = all(sequence_is_correct)

        output_dict = {
            "identifier": identifier,
            "sentence": sentence,
            "best_action_strings": best_action_strings,
            # "action_w_attention": action_w_attention,
            "best_logical_forms": logical_form,
            "label_strings": label_strings,
            "denotations": denotations,
            "sequence_is_correct": sequence_is_correct,
            "consistent": consistent,
            "consistent_programs": outputs["consistent_programs"]
        }

        output_str = json.dumps(output_dict) + "\n"

        return output_str
