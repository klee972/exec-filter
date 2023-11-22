import os
import sys
import json
import numpy
import argparse

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
)


from scripts.train.ASD import perform_computations as compute_statsign


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


def get_metrics(metrics_json):
    try:
        with open(metrics_json, 'r') as f:
            metrics_dict = json.load(f)
        if "best_validation_denotation_accuracy" in metrics_dict:
            denotation_acc = metrics_dict["best_validation_denotation_accuracy"]
        elif "denotation_accuracy" in metrics_dict:
            denotation_acc = metrics_dict["denotation_accuracy"]
        else:
            raise NotImplementedError

        if "best_validation_consistency" in metrics_dict:
            consistency = metrics_dict["best_validation_consistency"]
        elif "consistency" in metrics_dict:
            consistency = metrics_dict["consistency"]
        else:
            raise NotImplementedError
    except:
        print("Error reading: {}".format(metrics_json))
        denotation_acc, consistency = 0.0, 0.0
    return denotation_acc, consistency


def get_consistency_and_denotationaccs(metric_json_files):
    denotations, consistencies = [], []
    for metrics_json in metric_json_files:
        if os.path.exists(metrics_json):
            den, con = get_metrics(metrics_json)
            denotations.append(den)
            consistencies.append(con)
        else:
            print("Error reading: {}".format(metrics_json))

    consistencies = [c * 100.0 for c in consistencies]
    denotations = [d * 100.0 for d in denotations]

    consistencies = round_all(consistencies, 3)
    denotations = round_all(denotations, 3)

    return consistencies, denotations


def get_average(accuracies):
    return numpy.average(accuracies)


def get_dev_and_test_metrics(checkpoint_dir):
    # This directory should contain multiple SEED_X directories, each contains MML & ERM directories
    # Each of those contains Iter1_MDS14/ Iter2_MDS16/ Iter3_MDS18/ Iter4_MDS20/ Iter5_MDS22/
    seed_dirs = os.listdir(checkpoint_dir)
    seed_dirs = sorted(seed_dirs)
    print(seed_dirs)

    traintime_metrics_files = []
    dev_metrics_files = []
    test_metrics_files = []

    for seed_dir in seed_dirs:
        serdir = os.path.join(checkpoint_dir, seed_dir, "ERM", "Iter{}_MDS{}".format(5, 22))
        metrics_json = os.path.join(serdir, "metrics.json")
        traintime_metrics_files.append(metrics_json)

        dev_metrics_json = os.path.join(serdir, "predictions", "dev-metrics.json")
        dev_metrics_files.append(dev_metrics_json)

        test_metrics_json = os.path.join(serdir, "predictions", "test-metrics.json")
        test_metrics_files.append(test_metrics_json)
    # print("\nTraining time best dev: ")
    # traintime_con, traintime_den = get_consistency_and_denotationaccs(traintime_metrics_files)
    # # print("Consistencies: {}".format(traintime_con))
    # print("Average consistency: {}".format(get_average(traintime_con)))
    dev_con, dev_den = get_consistency_and_denotationaccs(dev_metrics_files)
    test_con, test_den = get_consistency_and_denotationaccs(test_metrics_files)
    return dev_con, dev_den, test_con, test_den


def print_metrics(cons, denacc):
    print("Consistencies: {}".format(cons))
    print("Average consistency: {} \t std: {}".format(get_average(cons), numpy.std(cons)))
    print("Denotation Acc: {}".format(denacc))
    print("Average Deno Acc: {} \t std: {}".format(get_average(denacc), numpy.std(denacc)))


def print_iterative_training_metrics(checkpoint_dir, ckptdir_baseline, all_max_decoding_steps=[12, 14, 16, 18, 20, 22]):
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
    print("Reading dev and test metrics ... ")
    dev_cons, dev_devacc, test_cons, test_devacc = get_dev_and_test_metrics(checkpoint_dir)
    print("Dev metrics: ")
    print_metrics(dev_cons, dev_devacc)

    print("\nTest metrics: ")
    print_metrics(test_cons, test_devacc)

    if ckptdir_baseline is None:
        return

    print("\nCKPT Baseline: {}".format(ckptdir_baseline))
    print("Reading dev and test metrics ... ")
    base_dev_cons, base_dev_denacc, base_test_cons, base_test_devacc = get_dev_and_test_metrics(ckptdir_baseline)
    print("\nBaseline Dev metrics: ")
    print_metrics(base_dev_cons, base_dev_denacc)
    print("\nBaseline Test metrics: ")
    print_metrics(base_test_cons, base_test_devacc)

    print("\n\nStatistical significance tests ...")

    print("\nDev stat-sign test ...")
    print("Average model dev: {}".format(get_average(dev_cons), numpy.std(dev_cons)))
    print("Average baseline dev: {} \t std: {}".format(get_average(base_dev_cons), numpy.std(base_dev_cons)))
    compute_statsign(dev_cons, base_dev_cons, alpha=0.05)
    compute_statsign(dev_cons, base_dev_cons, alpha=0.01)

    print("\nTest stat-sign test ...")
    print("Average model test: {}".format(get_average(test_cons)))
    print("Average baseline test: {} \t std: {}".format(get_average(base_test_cons), numpy.std(base_test_cons)))
    compute_statsign(test_cons, base_test_cons, alpha=0.05)
    compute_statsign(test_cons, base_test_cons, alpha=0.01)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_dir", type=str, help="CKPT dir containing MML/ ERM/ dirs")
    parser.add_argument("--ckpt_base", type=str, help="CKPT dir containing MML/ ERM/ dirs")

    args = parser.parse_args()
    print(args)
    print_iterative_training_metrics(checkpoint_dir=args.ckpt_dir, ckptdir_baseline=args.ckpt_base)
