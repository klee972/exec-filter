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
    predictions_jsonl: str,
) -> None:
    """
    Reads predictions on an Nlvr dataset from two different models and compares them. The predictions are given in JsonL
    format.
    """
    model: NlvrDatasetPredictions = NlvrDatasetPredictions(predictions_jsonl=predictions_jsonl)

    print("Model-1: {}".format(predictions_jsonl))
    print(
        "Num instances: {}   Consistent: {}   Avg. consistency: {}".format(
            model.num_instances, model.num_consistent, model.avg_consistency
        )
    )

    topk_consistent = 0
    for instance in model.instances:
        if instance.consistent_programs is not None and len(instance.consistent_programs) > 0:
            topk_consistent += 1
    topk_consistent = 100.0 * float(topk_consistent)/model.num_instances
    print(f"topk_consistent: {topk_consistent}")

    count = 0
    extra_consistent = 0
    for instance in model.instances:
        if not instance.consistent:
            count += 1
            print(f"{count}: {instance.sentence}")
            print(f"Prediction: {instance.best_logical_form}")
            if instance.consistent_programs:
                print("Consistent in beam:")
                programs = [x[0] for x in instance.consistent_programs]
                print("\n".join(programs))
                extra_consistent += 1
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "predictions_jsonl", type=str, help="JsonL file containing predictions from Model"
    )

    args = parser.parse_args()
    compare_models(
        args.predictions_jsonl,
    )
