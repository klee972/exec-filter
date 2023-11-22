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
from allennlp_semparse.models import NlvrMMLSemanticParser, NlvrPairedSemanticParser, NlvrCoverageSemanticParser, \
    NlvrERMCoverageSemanticParser
from allennlp_semparse.dataset_readers import NlvrV2PairedDatasetReader, NlvrV2DatasetReader
from scripts.nlvr_v2 import generate_data_from_paired_model, generate_data_from_coverage_model, generate_data_from_worldcode_model
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
    set_environ_var("TMPDIR", "/net/nfs2.corp/allennlp/nitishg/tmp")
    train_model_from_file(parameter_filename=configfile, serialization_dir=serialization_dir)


def get_metrics(metrics_json):
    with open(metrics_json, 'w') as f:
        metrics_dict = json.load(f)
    denotation_acc = metrics_dict["best_validation_denotation_accuracy"]
    consistency = metrics_dict["best_validation_consistency"]
    return denotation_acc, consistency


def train_iterative_parser(full_train_json, full_dev_json, train_searchcands_json, erm_model,
                           cuda_device, checkpoint_root, seed):
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

    assert erm_model in ["paired", "coverage", "worldcode"], "ERM model not supported. Given: {}".format(erm_model)
    erm_config_map = {
        "paired": "training_config/nlvr_paired_parser.jsonnet",
        "coverage": "training_config/nlvr_coverage_parser.jsonnet",
        "worldcode": "training_config/nlvr_worldcode_parser.jsonnet",
    }
    erm_data_generate_pythonscript_map = {
        "paired": generate_data_from_paired_model.make_data,
        "coverage": generate_data_from_coverage_model.make_data,
        "worldcode": generate_data_from_worldcode_model.make_data
    }

    MML_config = "training_config/nlvr_mml_parser.jsonnet"
    ERM_config = erm_config_map[erm_model]

    checkpoint_dir = os.path.join(checkpoint_root, "SEED_{}".format(seed))
    if os.path.exists(checkpoint_dir):
        print(f"CKPT Dir exists: {checkpoint_dir}")
        exit()

    os.makedirs(checkpoint_dir, exist_ok=True)

    print()
    print("MML Config: {}".format(MML_config))
    print("ERM Config: {}".format(ERM_config))
    print("ERM Training data: {}".format(full_train_json))
    print("Initial MML Training data: {}".format(train_searchcands_json))
    print("Checkpoint dir: {}".format(checkpoint_dir))

    print()
    # mml_config_tmpfile = tempfile.mkstemp(suffix=".jsonnet")[1]
    mml_config_filename = os.path.split(MML_config)[1]
    new_mml_path = os.path.join(checkpoint_dir, mml_config_filename)
    # copy_file(input_file=MML_config, output_file=mml_config_tmpfile)
    copy_file(input_file=MML_config, output_file=new_mml_path)
    print("Copied MML config ({}) to: {}".format(MML_config, new_mml_path))
    # MML_config = mml_config_tmpfile
    MML_config = new_mml_path

    # erm_config_tmpfile = tempfile.mkstemp(suffix=".jsonnet")[1]
    erm_config_filename = os.path.split(ERM_config)[1]
    new_erm_path = os.path.join(checkpoint_dir, erm_config_filename)
    # copy_file(input_file=ERM_config, output_file=erm_config_tmpfile)
    copy_file(input_file=ERM_config, output_file=new_erm_path)
    # print("Copied ERM config ({}) to: {}".format(ERM_config, erm_config_tmpfile))
    print("Copied ERM config ({}) to: {}".format(ERM_config, new_erm_path))
    # ERM_config = erm_config_tmpfile
    ERM_config = new_erm_path

    # tmp_dir = os.path.split(erm_config_tmpfile)[0]
    # copy_file(input_file="training_config/utils.libsonnet", output_file=os.path.join(tmp_dir, "utils.libsonnet"))
    copy_file(input_file="training_config/utils.libsonnet", output_file=os.path.join(checkpoint_dir, "utils.libsonnet"))


    # if not query_yes_no("\nStart training?"):
    #     print("Exiting ...\n")
    #     exit()

    mml_ckpt_dir = os.path.join(checkpoint_dir, "MML")
    erm_ckpt_dir = os.path.join(checkpoint_dir, "ERM")
    gendata_dir = os.path.join(checkpoint_dir, "GenData")

    os.makedirs(mml_ckpt_dir, exist_ok=True)
    os.makedirs(erm_ckpt_dir, exist_ok=True)
    os.makedirs(gendata_dir, exist_ok=True)

    set_environ_var("SEED", seed)   # Seed
    set_environ_var("CUDA", cuda_device)     # Run on GPU
    set_environ_var("EPOCHS", 50)   # Total Epochs
    set_environ_var("DEV_DATA", full_dev_json)   # All models evaluated on full dev

    all_max_decoding_steps = [12, 14, 16, 18, 20, 22]

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

        selection_threshold = 0.8

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
        python_data_generate_script = erm_data_generate_pythonscript_map[erm_model]
        if erm_model == "worldcode":
            python_data_generate_script(input_file=full_train_json, output_file=erm_datagen_train_json,
                                        archived_model_file=erm_model_targz, max_num_decoded_sequences=20,
                                        cuda_device=cuda_device, prune_data=False,
                                        selection_threshold=selection_threshold
                                        )
        else:
            python_data_generate_script(input_file=full_train_json, output_file=erm_datagen_train_json,
                                        archived_model_file=erm_model_targz, max_num_decoded_sequences=20,
                                        cuda_device=cuda_device, prune_data=False
                                        )

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
    print_iterative_training_metrics(checkpoint_dir=checkpoint_dir, all_max_decoding_steps=all_max_decoding_steps)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, help="NLVR data file",
                        default="./resources/data/nlvr/processed/train_grouped.json")
    parser.add_argument("--dev_json", type=str, help="Processed output",
                        default="./resources/data/nlvr/processed/dev_grouped.json")
    parser.add_argument("--train_search_json", type=str, help="Processed output",
                        default="./resources/data/nlvr/processed/agenda_v6_ML11/train_grouped.json")
    parser.add_argument("--erm_model", type=str, default="paired",
                        help="Which model to use for ERM model (paired/coverage)")
    parser.add_argument("--cuda-device", dest="cuda_device", type=int, default=-1)
    parser.add_argument("--ckpt_root", type=str, help="CKPT root")
    parser.add_argument("--seed", type=str, help="seed")

    args = parser.parse_args()
    print(args)
    train_iterative_parser(
        full_train_json=args.train_json,
        full_dev_json=args.dev_json,
        train_searchcands_json=args.train_search_json,
        erm_model=args.erm_model,
        cuda_device=args.cuda_device,
        checkpoint_root=args.ckpt_root,
        seed=args.seed)
