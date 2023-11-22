#! /usr/bin/env python
import argparse
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
    ),
)

from scripts.nlvr_v2.data.nlvr_prediction import NlvrDatasetPredictions, NlvrPredictionInstance


def compare_models(
    predictions_jsonl_1: str,
    predictions_jsonl_2: str,
) -> None:
    """
    Reads predictions on an Nlvr dataset from two different models and compares them. The predictions are given in JsonL
    format.
    """
    model1: NlvrDatasetPredictions = NlvrDatasetPredictions(predictions_jsonl=predictions_jsonl_1)
    model2: NlvrDatasetPredictions = NlvrDatasetPredictions(predictions_jsonl=predictions_jsonl_2)

    print("Model-1: {}".format(predictions_jsonl_1))
    print(
        "Num instances: {}   Consistent: {}   Avg. consistency: {}".format(
            model1.num_instances, model1.num_consistent, model1.avg_consistency
        )
    )

    print("\nModel-2: {}".format(predictions_jsonl_2))
    print(
        "Num instances: {}   Consistent: {}   Avg. consistency: {}".format(
            model2.num_instances, model2.num_consistent, model2.avg_consistency
        )
    )
    print()

    correct_instance_ids_1 = set(model1.consistent_instance_ids)
    correct_instance_ids_2 = set(model2.consistent_instance_ids)
    common_correct = correct_instance_ids_1.intersection(correct_instance_ids_2)
    correct_1_not_2 = correct_instance_ids_1.difference(correct_instance_ids_2)
    correct_2_not_1 = correct_instance_ids_2.difference(correct_instance_ids_1)

    print(f"Common correct: {len(common_correct)}")
    print(f"correct_1_not_2: {len(correct_1_not_2)}")
    print(f"correct_2_not_1: {len(correct_2_not_1)}")

    print("Correct in Model-1 but not in Model-2")
    for idf in correct_1_not_2:
        instance1: NlvrPredictionInstance = model1.id2instance[idf]
        instance2: NlvrPredictionInstance = model2.id2instance[idf]
        print(instance1.sentence)
        print("M1: {}".format(instance1.best_logical_form))
        print("M2: {}".format(instance2.best_logical_form))
        print()

    print("\n\nCorrect in Model-2 but not in Model-1")
    for idf in correct_2_not_1:
        instance1: NlvrPredictionInstance = model1.id2instance[idf]
        instance2: NlvrPredictionInstance = model2.id2instance[idf]
        print(instance1.sentence)
        print("M1: {}".format(instance1.best_logical_form))
        print("M2: {}".format(instance2.best_logical_form))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "predictions_jsonl_1", type=str, help="JsonL file containing predictions from Model 1"
    )
    parser.add_argument(
        "predictions_jsonl_2", type=str, help="JsonL file containing predictions from Model 2"
    )

    args = parser.parse_args()
    compare_models(
        args.predictions_jsonl_1,
        args.predictions_jsonl_2,
    )
