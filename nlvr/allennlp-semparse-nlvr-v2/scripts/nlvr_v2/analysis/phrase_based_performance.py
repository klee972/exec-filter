#! /usr/bin/env python
from typing import List, Dict, Tuple
import argparse
import os
import sys
import json

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
    ),
)

from scripts.nlvr_v2.data.nlvr_prediction import NlvrDatasetPredictions, NlvrPredictionInstance
from scripts.nlvr_v2.paired_supervision.generate_paired_data import convert_abstract_phrase_to_grounded, PairedPhrase


def get_phrase_clusters(paired_phrases_json):
    print("Reading paired data ... ")
    # Each inner list is a set of equivalent abstract phrases; e.g. ["COLOR1 as the base", "the base is COLOR1"]
    with open(paired_phrases_json) as f:
        paired_phrases_list: List[List[str]] = json.load(f)

    paired_phrases: List[PairedPhrase] = []

    abstractcluster2abstractphrases = {}
    abstractcluster2groundedphrases = {}
    for cluster_id, equivalent_abstract_set in enumerate(paired_phrases_list):
        # One set of equivalent abstract phrases is converted to many equivalent sets after grounding
        equivalent_grounded_sets: List[List[str]] = convert_abstract_phrase_to_grounded(
            equivalent_abstract_set
        )

        abstractcluster2abstractphrases[cluster_id] = equivalent_abstract_set
        abstractcluster2groundedphrases[cluster_id] = [x for y in equivalent_grounded_sets for x in y]


    return abstractcluster2abstractphrases, abstractcluster2groundedphrases


def get_cluster2instance_mapping(abstractcluster2groundedphrases, instances: List[NlvrPredictionInstance]):
    abstractcluster2identifiers = {}
    for instance in instances:
        instance: NlvrPredictionInstance = instance
        sentence = instance.sentence
        identifer = instance.identifier

        for cluster_id, equivalent_phrases in abstractcluster2groundedphrases.items():
            if any(x in sentence for x in equivalent_phrases):
                if cluster_id not in abstractcluster2identifiers:
                    abstractcluster2identifiers[cluster_id] = set()
                abstractcluster2identifiers[cluster_id].add(identifer)
    return abstractcluster2identifiers



def compare_models(
    predictions_jsonl_1: str,
    predictions_jsonl_2: str,
    paired_phrases_json: str,
) -> None:
    """
    Reads predictions on an Nlvr dataset from two different models and compares them. The predictions are given in JsonL
    format.
    """
    model1: NlvrDatasetPredictions = NlvrDatasetPredictions(predictions_jsonl=predictions_jsonl_1)
    model2: NlvrDatasetPredictions = NlvrDatasetPredictions(predictions_jsonl=predictions_jsonl_2)

    abstractcluster2abstractphrases, abstractcluster2groundedphrases = get_phrase_clusters(paired_phrases_json)

    print("Model-1: {}".format(predictions_jsonl_1))
    print(
        "Num instances: {}   Consistent: {}   Avg. consistency: {}".format(
            model1.num_instances, model1.num_consistent, model1.avg_consistency
        )
    )
    print("Model-2: {}".format(predictions_jsonl_2))
    print(
        "Num instances: {}   Consistent: {}   Avg. consistency: {}".format(
            model2.num_instances, model2.num_consistent, model2.avg_consistency
        )
    )
    print()

    print("")
    # Instances for both model's should be same; so computing using just 1
    abstractcluster2identifiers = get_cluster2instance_mapping(abstractcluster2groundedphrases, model1.instances)

    for clusterid, abstract_phrases in abstractcluster2abstractphrases.items():
        if clusterid in abstractcluster2identifiers:
            print("Cluster ID: {}".format(clusterid))
            print(abstract_phrases)
            identifiers = abstractcluster2identifiers[clusterid]
            print("NumInstances: {}".format(len(identifiers)))
            consistency = model1.compute_avg_consistency_for_subset(identifiers)
            print("M1 Consistency: {}".format(consistency))
            consistency = model2.compute_avg_consistency_for_subset(identifiers)
            print("M2 Consistency: {}".format(consistency))
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paired_phrases_json", type=str, help="Json containing clusters of abstract paired phrases"
    )
    parser.add_argument(
        "predictions_jsonl_1", type=str, help="JsonL file containing predictions from Model"
    )
    parser.add_argument(
        "predictions_jsonl_2", type=str, help="JsonL file containing predictions from Model"
    )


    args = parser.parse_args()
    compare_models(
        args.predictions_jsonl_1,
        args.predictions_jsonl_2,
        args.paired_phrases_json,
    )
