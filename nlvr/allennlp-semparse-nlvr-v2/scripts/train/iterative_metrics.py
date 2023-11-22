import os
import json
import subprocess
import argparse


def set_environ_var(variable, value):
    os.environ[variable] = str(value)
    # subprocess.call(["export", "{}={}".format(variable, value)])


def allennlp_train(serialization_dir, configfile):
    allennlp_command = "allennlp train --include-package allennlp_semparse -s {} {}".format(serialization_dir,
                                                                                            configfile)
    subprocess.run(allennlp_command.split(" "))


def get_metrics(metrics_json):
    try:
        with open(metrics_json, 'r') as f:
            metrics_dict = json.load(f)
        denotation_acc = metrics_dict["best_validation_denotation_accuracy"]
        consistency = metrics_dict["best_validation_consistency"]
    except:
        print("Error reading: {}".format(metrics_json))
        denotation_acc, consistency = 0.0, 0.0
    return denotation_acc, consistency


def print_iterative_training_metrics(checkpoint_dir, all_max_decoding_steps=[12, 14, 16, 18, 20, 22]):
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
    den, con = get_metrics(metrics_json)
    print("MML Iteration: {}  MDS: {}".format(iteration, mds))
    print("Denotation Acc: {} Consistency: {}".format(den, con))
    for mds in all_max_decoding_steps[1:]:
        iteration += 1
        metrics_json = os.path.join(erm_ckpt_dir, "Iter{}_MDS{}".format(iteration, mds), "metrics.json")
        print("\nERM Iteration: {}  MDS: {}".format(iteration, mds))
        den, con = get_metrics(metrics_json)
        print("Denotation Acc: {} Consistency: {}".format(den, con))

        metrics_json = os.path.join(mml_ckpt_dir, "Iter{}_MDS{}".format(iteration, mds), "metrics.json")
        print("MML Iteration: {}  MDS: {}".format(iteration, mds))
        den, con = get_metrics(metrics_json)
        print("Denotation Acc: {} Consistency: {}".format(den, con))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_dir", type=str, help="CKPT dir containing MML/ ERM/ dirs")

    set_environ_var("OMP_NUM_THREADS", 2)
    set_environ_var("OPENBLAS_NUM_THREADS", 2)
    set_environ_var("OPENMP_NUM_THREADS", 2)
    set_environ_var("MKL_NUM_THREADS", 2)

    args = parser.parse_args()
    print(args)
    print_iterative_training_metrics(checkpoint_dir=args.ckpt_dir)
