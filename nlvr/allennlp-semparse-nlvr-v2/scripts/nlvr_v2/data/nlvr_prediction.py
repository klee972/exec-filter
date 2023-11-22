from typing import List, Dict, Tuple, Union, Set
import json


class NlvrPredictionInstance:
    def __init__(self, prediction_dict: Dict):
        self.identifier: str = prediction_dict["identifier"]
        self.sentence: str = prediction_dict["sentence"]
        self.best_action_strings: List[str] = prediction_dict.get("best_action_strings", [])
        self.decoded_logical_forms: List[str] = prediction_dict.get("best_logical_forms", [])
        self.best_logical_form: str = (
            self.decoded_logical_forms[0] if len(self.decoded_logical_forms) > 0 else ""
        )
        self.label_strings: List[str] = prediction_dict.get("label_strings", [])
        # Could be denotations for the best logical-form OR for all programs in the beam
        self.denotations: Union[List[str], List[List[str]]] = prediction_dict.get("denotations", [])
        self.sequence_is_correct: List[bool] = prediction_dict.get("sequence_is_correct", [False])
        self.consistent: bool = prediction_dict.get("consistent", False)
        self.consistent_programs: List[Tuple[str, float]] = prediction_dict.get("consistent_programs", None)


def read_nlvr_predictions(predictions_jsonl: str) -> List[NlvrPredictionInstance]:
    instances = []
    with open(predictions_jsonl) as f:
        for line in f.readlines():
            prediction_dict: Dict = json.loads(line)
            instance = NlvrPredictionInstance(prediction_dict)
            instances.append(instance)
    return instances


class NlvrDatasetPredictions:
    def __init__(
        self, predictions_jsonl: str = None, instances: List[NlvrPredictionInstance] = None
    ):
        if instances:
            self.instances = instances
        else:
            self.instances = read_nlvr_predictions(predictions_jsonl)

        self.id2instance = {instance.identifier: instance for instance in self.instances}

        self.consistent_instance_ids: List[str] = [
            instance.identifier for instance in self.instances if instance.consistent
        ]

        self.num_instances = len(self.instances)
        self.num_consistent = len(self.consistent_instance_ids)
        self.avg_consistency = self.compute_avg_consistency()

    def compute_avg_consistency(self):
        avg_consistency = 100.0 * (float(self.num_consistent) / self.num_instances)
        return avg_consistency

    def compute_avg_consistency_for_subset(self, identifiers: Set[str]):
        identifiers = set(identifiers)
        subset = []
        for instance in self.instances:
            if instance.identifier in identifiers:
                subset.append(instance)

        num_consistent = len([
            instance.identifier for instance in subset if instance.consistent
        ])

        # print("Computing avg. consistency for subset; given: {} found: {}".format(
        #     len(identifiers), len(subset)
        # ))

        avg_consistency = 100.0 * (float(num_consistent) / len(subset))
        return avg_consistency
