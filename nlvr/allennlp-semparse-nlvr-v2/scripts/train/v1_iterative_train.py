import os
import sys
import json
import tempfile
import subprocess
import argparse

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
)


# Set environment variables for threads before import torch/numpy
def set_environ_var(variable, value):
    os.environ[variable] = str(value)


set_environ_var("OMP_NUM_THREADS", 1)
set_environ_var("OPENBLAS_NUM_THREADS", 1)
set_environ_var("OPENMP_NUM_THREADS", 1)
set_environ_var("MKL_NUM_THREADS", 1)

from allennlp.commands.train import train_model_from_file
from allennlp_semparse.models import NlvrDirectSemanticParser, NlvrCoverageSemanticParser
from allennlp_semparse.dataset_readers import NlvrDatasetReader
from scripts.nlvr import generate_data_from_erm_model
from scripts.train.iterative_metrics import print_iterative_training_metrics


def query_yes_no(question, default="no"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def copy_file(input_file, output_file):
    with open(output_file, "w") as fw, open(input_file, "r") as fr:
        fw.writelines(l for l in fr)


def allennlp_train(serialization_dir, configfile):
    set_environ_var("OMP_NUM_THREADS", 2)
    set_environ_var("OPENBLAS_NUM_THREADS", 2)
    set_environ_var("OPENMP_NUM_THREADS", 2)
    set_environ_var("MKL_NUM_THREADS", 2)
    train_model_from_file(parameter_filename=configfile, serialization_dir=serialization_dir)


def get_metrics(metrics_json):
    with open(metrics_json, 'w') as f:
        metrics_dict = json.load(f)
    denotation_acc = metrics_dict["best_validation_denotation_accuracy"]
    consistency = metrics_dict["best_validation_consistency"]
    return denotation_acc, consistency


def train_iterative_parser(full_train_json, full_dev_json, train_searchcands_json, checkpoint_root, seed):
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

    MML_config = "training_config/nlvr_direct_parser.jsonnet"
    ERM_config = "training_config/nlvr_v1_coverage_parser.jsonnet"

    checkpoint_dir = os.path.join(checkpoint_root, "SEED_{}".format(seed))
    if os.path.exists(checkpoint_dir):
        print(f"CKPT Dir exists: {checkpoint_dir}")
        exit()

    print()
    print("MML Config: {}".format(MML_config))
    print("ERM Config: {}".format(ERM_config))
    print("ERM Training data: {}".format(full_train_json))
    print("Initial MML Training data: {}".format(train_searchcands_json))
    print("Checkpoint dir: {}".format(checkpoint_dir))

    print()
    mml_config_tmpfile = tempfile.mkstemp(suffix=".jsonnet")[1]
    copy_file(input_file=MML_config, output_file=mml_config_tmpfile)
    print("Copied MML config ({}) to: {}".format(MML_config, mml_config_tmpfile))
    MML_config = mml_config_tmpfile

    erm_config_tmpfile = tempfile.mkstemp(suffix=".jsonnet")[1]
    copy_file(input_file=ERM_config, output_file=erm_config_tmpfile)
    print("Copied ERM config ({}) to: {}".format(ERM_config, erm_config_tmpfile))
    ERM_config = erm_config_tmpfile
    tmp_dir = os.path.split(erm_config_tmpfile)[0]
    copy_file(input_file="training_config/utils.libsonnet", output_file=os.path.join(tmp_dir, "utils.libsonnet"))

    if not query_yes_no("\nStart training?"):
        print("Exiting ...\n")
        exit()

    mml_ckpt_dir = os.path.join(checkpoint_dir, "MML")
    erm_ckpt_dir = os.path.join(checkpoint_dir, "ERM")
    gendata_dir = os.path.join(checkpoint_dir, "GenData")

    os.makedirs(mml_ckpt_dir, exist_ok=True)
    os.makedirs(erm_ckpt_dir, exist_ok=True)
    os.makedirs(gendata_dir, exist_ok=True)

    set_environ_var("SEED", seed)   # Seed
    set_environ_var("CUDA", -1)     # Run on CPU
    set_environ_var("EPOCHS", 50)   # Total Epochs
    set_environ_var("DEV_DATA", full_dev_json)   # All models evaluated on full dev

    all_max_decoding_steps = [10, 12, 14, 16, 18, 20, 22]

    """ Iteration 0 - Train MML model on train_searchcands """
    iteration = 0
    configfile = MML_config
    # Train MML iteration 0 on exhaustive-search candidates
    trainpath = train_searchcands_json
    set_environ_var("TRAIN_DATA", trainpath)
    mds = all_max_decoding_steps[iteration]
    set_environ_var("MDS", mds)
    serialization_dir = os.path.join(mml_ckpt_dir, "Iter{}_MDS{}".format(iteration, mds))

    print("\nTraining MML parser, Iteration {} MDS: {}".format(iteration, mds))
    print("CKPT Dir: {}\n".format(serialization_dir))

    allennlp_train(serialization_dir, configfile)
    mml_model_targz = os.path.join(serialization_dir, "model.tar.gz")

    for mds in all_max_decoding_steps[1:]:
        iteration += 1
        mds = all_max_decoding_steps[iteration]

        """Run ERM training initialized from previous MML parser"""
        configfile = ERM_config
        # Train MML iteration 0 on exhaustive-search candidates
        trainpath = full_train_json
        set_environ_var("TRAIN_DATA", trainpath)
        set_environ_var("MDS", mds)
        # Initialize ERM model from previous MML model
        set_environ_var("MML_MODEL_TAR", mml_model_targz)
        serialization_dir = os.path.join(erm_ckpt_dir, "Iter{}_MDS{}".format(iteration, mds))
        print("\nTraining ERM parser, Iteration {} MDS: {}".format(iteration, mds))
        print("initializing from: {}".format(mml_model_targz))
        print("CKPT Dir: {}\n".format(serialization_dir))
        allennlp_train(serialization_dir, configfile)
        erm_model_targz = os.path.join(serialization_dir, "model.tar.gz")

        """Generate data from ERM model"""
        erm_datagen_train_json = os.path.join(gendata_dir, "train_Iter{}_MDS{}.json".format(iteration, mds))
        print("\nGenerating train program-candidates from ERM model: {}\n".format(erm_model_targz))
        # One of scripts/nlvr_v2/{generate_data_from_paired_model.py OR generate_data_from_coverage_model.py}
        python_data_generate_script = generate_data_from_erm_model.make_data
        python_data_generate_script(input_file=full_train_json, output_file=erm_datagen_train_json,
                                    archived_model_file=erm_model_targz, max_num_decoded_sequences=20)

        """Run MML parser using data generated in previous state"""
        configfile = MML_config
        # Train MML from erm-model's candidates
        trainpath = erm_datagen_train_json
        set_environ_var("TRAIN_DATA", trainpath)
        set_environ_var("MDS", mds)
        serialization_dir = os.path.join(mml_ckpt_dir, "Iter{}_MDS{}".format(iteration, mds))
        print("\nTraining MML parser, Iteration {} MDS: {}".format(iteration, mds))
        print("CKPT Dir: {}\n".format(serialization_dir))
        allennlp_train(serialization_dir, configfile)
        mml_model_targz = os.path.join(serialization_dir, "model.tar.gz")

    # Print metrics for all iterations
    print_iterative_training_metrics(checkpoint_dir=checkpoint_dir)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, help="NLVR data file",
                        default="./resources/data/nlvr/v1/train_grouped.json")
    parser.add_argument("--dev_json", type=str, help="Processed output",
                        default="./resources/data/nlvr/v1/dev_grouped.json")
    parser.add_argument("--train_search_json", type=str, help="Processed output",
                        default="./resources/data/nlvr/v1/train_ES.json")
    parser.add_argument("--ckpt_root", type=str, help="CKPT root")
    parser.add_argument("--seed", type=str, help="seed")

    args = parser.parse_args()
    print(args)
    train_iterative_parser(
        full_train_json=args.train_json,
        full_dev_json=args.dev_json,
        train_searchcands_json=args.train_search_json,
        checkpoint_root=args.ckpt_root,
        seed=args.seed)
