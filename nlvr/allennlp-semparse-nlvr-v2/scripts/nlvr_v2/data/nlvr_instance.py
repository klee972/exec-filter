from typing import List, Dict, Tuple, Union
import json
import os

from allennlp_semparse.domain_languages.nlvr_language_v2 import NlvrLanguageFuncComposition, Box


class NlvrInstance:
    def __init__(self, instance_dict: Dict):
        self.identifier: str = instance_dict["identifier"]
        self.sentence: str = instance_dict["sentence"]

        if "worlds" in instance_dict:
            # This means that we are reading grouped nlvr data. There will be multiple
            # worlds and corresponding labels per sentence.
            labels = instance_dict["labels"]
            structured_representations = instance_dict["worlds"]
        else:
            print("Cannot work with un-grouped NLVR data")
            raise NotImplementedError
            # # We will make lists of labels and structured representations, each with just
            # # one element for consistency.
            # labels = [instance_dict["label"]]
            # structured_representations = [instance_dict["structured_rep"]]

        self.labels = labels
        self.structured_representations: List[Dict] = structured_representations
        self.worlds: List[NlvrLanguageFuncComposition] = None

        # if "correct_sequences" in instance_dict:
        self.correct_candidate_sequences = instance_dict.get("correct_sequences", None)

        # Should have keys: identifier, sentence, structured_representations, labels,
        # orig_charoffsets, paired_charoffsets
        self.paired_examples: List[Dict] = instance_dict.get("paired_examples", None)

        # Store extra stuff from the instance not covered above
        self.extras = {}
        for key in instance_dict:
            if key not in ["identifier", "sentence", "worlds", "labels", "correct_sequences"]:
                self.extras[key] = instance_dict[key]

    def to_dict(self):
        output_dict = {
            "identifier": self.identifier,
            "sentence": self.sentence,
            "worlds": self.structured_representations,
            "labels": self.labels,
        }
        output_dict.update(self.extras)
        if self.correct_candidate_sequences is not None:
            output_dict["correct_sequences"] = self.correct_candidate_sequences
        if self.paired_examples is not None:
            output_dict["paired_examples"] = self.paired_examples

        return output_dict

    def convert_structured_to_worlds(self):
        self.worlds = []
        for structured_representation in self.structured_representations:
            boxes = {
                Box(object_list, box_id)
                for box_id, object_list in enumerate(structured_representation)
            }
            self.worlds.append(NlvrLanguageFuncComposition(boxes))


def print_dataset_stats(instances: List[NlvrInstance]):
    num_instances = len(instances)
    num_w_correct_sequences, num_pairings = 0, 0
    for instance in instances:
        if instance.correct_candidate_sequences is not None:
            if len(instance.correct_candidate_sequences) > 0:
                num_w_correct_sequences += 1
        if instance.paired_examples is not None:
            num_pairings += len(instance.paired_examples)

    print("Total instances: {}  Num w/ program-candidates: {}  Num w/ paired example pairings: {}".format(
        num_instances, num_w_correct_sequences, num_pairings
    ))


def read_nlvr_data(input_jsonl: str) -> List[NlvrInstance]:
    print("Reading instances from: {}".format(input_jsonl))
    instances: List[NlvrInstance] = []
    with open(input_jsonl) as data_file:
        for line in data_file:
            line = line.strip("\n")
            if not line:
                continue
            data = json.loads(line)
            instance: NlvrInstance = NlvrInstance(data)
            instances.append(instance)
    print("Num instances read: {}".format(len(instances)))
    return instances


def write_nlvr_data(instances: List[NlvrInstance], output_jsonl: str):
    print("Writing {} data to: {}".format(len(instances), output_jsonl))
    output_dir = os.path.split(output_jsonl)[0]
    os.makedirs(output_dir, exist_ok=True)
    with open(output_jsonl, 'w') as outf:
        for instance in instances:
            output_dict = instance.to_dict()
            outf.write(json.dumps(output_dict))
            outf.write("\n")
    print("Done")
