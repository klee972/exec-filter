from typing import List
import os
import json
import subprocess
import argparse


def set_environ_var(variable, value):
    os.environ[variable] = str(value)


def get_metrics(metrics_json):
    try:
        with open(metrics_json, 'r') as f:
            metrics_dict = json.load(f)
        denotation_acc = metrics_dict["best_validation_denotation_acc"]
    except:
        print("Error reading: {}".format(metrics_json))
        denotation_acc = 0.0
    return denotation_acc


def print_iterative_training_metrics(checkpoint_dir,
                                     all_max_decoding_steps: List[int] = [10, 12, 14, 16, 18, 20]):
    """Iteratively train a parer by alternating between MML and ERM parsers.

    The directory structure inside checkpoint root is:
    SEED_S/
        MML/Iter${ITER}_MDS${MDS}/
        PairedERM/Iter${ITER}_MDS${MDS}/
        GenData/train_ERM_Iter${ITER}.json


    Parameters:
    -----------
    full_train_json, full_dev_json: Complete grouped NLVR train/dev data.

    """

    print("\nCheckpoint dir: {}".format(checkpoint_dir))

    mml_ckpt_dir = os.path.join(checkpoint_dir, "MML")
    erm_ckpt_dir = os.path.join(checkpoint_dir, "ERM")

    # Print metrics
    iteration = 0
    mds = all_max_decoding_steps[iteration]
    metrics_json = os.path.join(mml_ckpt_dir, "Iter{}_MDS{}".format(iteration, mds), "metrics.json")
    print(metrics_json)
    den = get_metrics(metrics_json)
    print("MML Iteration: {}  MDS: {}  Acc: {}".format(iteration, mds, den))
    for mds in all_max_decoding_steps[1:]:
        print()
        iteration += 1
        metrics_json = os.path.join(erm_ckpt_dir, "Iter{}_MDS{}".format(iteration, mds), "metrics.json")
        den = get_metrics(metrics_json)
        print("ERM Iteration: {}  MDS: {}  Acc: {}".format(iteration, mds, den))

        metrics_json = os.path.join(mml_ckpt_dir, "Iter{}_MDS{}".format(iteration, mds), "metrics.json")
        den = get_metrics(metrics_json)
        print("MML Iteration: {}  MDS: {}  Acc: {}".format(iteration, mds, den))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_dir", type=str, help="CKPT dir containing MML/ ERM/ dirs")

    args = parser.parse_args()
    print(args)
    print_iterative_training_metrics(checkpoint_dir=args.ckpt_dir)
