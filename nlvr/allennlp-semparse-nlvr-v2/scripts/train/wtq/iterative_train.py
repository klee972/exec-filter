import os
import sys
import json
import tempfile
import subprocess
import argparse

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))))
)

# Set environment variables for threads before import torch/numpy
def set_environ_var(variable, value):
    os.environ[variable] = str(value)


set_environ_var("OMP_NUM_THREADS", 1)
set_environ_var("OPENBLAS_NUM_THREADS", 1)
set_environ_var("OPENMP_NUM_THREADS", 1)
set_environ_var("MKL_NUM_THREADS", 1)

from allennlp.commands.train import train_model_from_file
from allennlp_semparse.models import WikiTablesErmSemanticParser, WikiTablesMmlSemanticParser
from allennlp_semparse.dataset_readers import WikiTablesDatasetReader
import scripts.wikitables.generate_data_from_erm_model as generate_data_from_erm_model


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
    denotation_acc = metrics_dict["best_validation_denotation_acc"]
    return denotation_acc


def train_iterative_parser(train_examples_file, dev_examples_file, train_search_dir,
                           tables_dir, checkpoint_root, split_num: int, seed=None, erm_model=None):
    """Iteratively train a parer by alternating between MML and ERM parsers.

    The directory structure inside checkpoint root is:
      split_X/
        MML/Iter${ITER}_MDS${MDS}/
        PairedERM/Iter${ITER}_MDS${MDS}/
        GenData/Iter${ITER}/

    Parameters:
    -----------
    train_examples_file, dev_examples_file: random-split-X-{train, dev}.examples
    train_search_dir: Directory containing candidates from exhaustive search
    """

    # assert erm_model in ["paired", "coverage"], "ERM model not supported. Given: {}".format(erm_model)
    # erm_config_map = {
    #     "paired": "training_config/nlvr_paired_parser.jsonnet",
    #     "coverage": "training_config/nlvr_coverage_parser.jsonnet",
    # }
    # erm_config_map[erm_model]
    # erm_data_generate_pythonscript_map = {
    #     "paired": generate_data_from_paired_model.make_data,
    #     "coverage": generate_data_from_coverage_model.make_data
    # }

    MML_config = "training_config/wikitables_mml_parser.jsonnet"
    ERM_config = "training_config/wikitables_erm_parser.jsonnet"

    checkpoint_dir = os.path.join(checkpoint_root, "split_{}".format(split_num))

    # if os.path.exists(checkpoint_dir):
    #     print(f"CKPT Dir exists: {checkpoint_dir}")
    #     exit()

    os.makedirs(checkpoint_dir, exist_ok=True)

    print()
    print("MML Config: {}".format(MML_config))
    print("ERM Config: {}".format(ERM_config))
    print("Tables dir: {}".format(tables_dir))
    print("Dev data: {}".format(dev_examples_file))
    print("ERM Training data: {}".format(train_examples_file))
    print("Initial MML Training data: {}".format(train_search_dir))
    print("Checkpoint dir: {}".format(checkpoint_dir))

    print()
    mml_config_filename = os.path.split(MML_config)[1]
    new_mml_path = os.path.join(checkpoint_dir, mml_config_filename)
    copy_file(input_file=MML_config, output_file=new_mml_path)
    print("Copied MML config ({}) to: {}".format(MML_config, new_mml_path))
    MML_config = new_mml_path

    erm_config_filename = os.path.split(ERM_config)[1]
    new_erm_path = os.path.join(checkpoint_dir, erm_config_filename)
    copy_file(input_file=ERM_config, output_file=new_erm_path)
    print("Copied ERM config ({}) to: {}".format(ERM_config, new_erm_path))
    ERM_config = new_erm_path

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

    # set_environ_var("SEED", seed)   # Seed
    # set_environ_var("CUDA", -1)     # Run on CPU
    # set_environ_var("EPOCHS", 50)   # Total Epochs
    set_environ_var("TRAIN_DATA", train_examples_file)  # All models evaluated on full dev
    set_environ_var("DEV_DATA", dev_examples_file)   # All models evaluated on full dev

    all_max_decoding_steps = [10, 12, 14, 16, 18, 20]

    """ Iteration 0 - Train MML model on train_searchcands """
    iteration = 0
    configfile = MML_config
    # Train MML iteration 0 on exhaustive-search candidates
    trainpath = train_search_dir
    set_environ_var("TRAIN_DATA", train_examples_file)
    set_environ_var("TRAIN_SEARCH_DIR", train_search_dir)
    set_environ_var("TABLES_DIR", tables_dir)
    mds = all_max_decoding_steps[iteration]
    set_environ_var("MDS", mds)
    serialization_dir = os.path.join(mml_ckpt_dir, "Iter{}_MDS{}".format(iteration, mds))

    print("\nTraining MML parser, Iteration {} MDS: {}".format(iteration, mds))
    print("CKPT Dir: {}\n".format(serialization_dir))

    # allennlp_train(serialization_dir, configfile)
    mml_model_targz = os.path.join(serialization_dir, "model.tar.gz")

    for mds in all_max_decoding_steps[1:]:
        iteration += 1
        mds = all_max_decoding_steps[iteration]

        """Run ERM training initialized from previous MML parser"""
        configfile = ERM_config
        # Train MML iteration 0 on exhaustive-search candidates
        set_environ_var("TRAIN_DATA", train_examples_file)
        set_environ_var("TABLES_DIR", tables_dir)
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
        erm_datagen_train_dir = os.path.join(gendata_dir, "Iter{}_MDS{}".format(iteration, mds))
        print("\nGenerating train program-candidates from ERM model: {}\n".format(erm_model_targz))
        # One of scripts/nlvr_v2/{generate_data_from_paired_model.py OR generate_data_from_coverage_model.py}
        # python_data_generate_script = erm_data_generate_pythonscript_map[erm_model]
        generate_data_from_erm_model.make_data(input_examples_file=train_examples_file,
                                               tables_directory=tables_dir,
                                               archived_model_file=erm_model_targz,
                                               output_dir=erm_datagen_train_dir,
                                               num_logical_forms=20)

        """Run MML parser using data generated in previous state"""
        configfile = MML_config
        # Train MML from erm-model's candidates
        train_candidates_dir = erm_datagen_train_dir
        set_environ_var("TRAIN_SEARCH_DIR", train_candidates_dir)
        set_environ_var("TRAIN_DATA", train_examples_file)
        set_environ_var("TABLES_DIR", tables_dir)
        set_environ_var("MDS", mds)
        serialization_dir = os.path.join(mml_ckpt_dir, "Iter{}_MDS{}".format(iteration, mds))
        print("\nTraining MML parser, Iteration {} MDS: {}".format(iteration, mds))
        print("CKPT Dir: {}\n".format(serialization_dir))
        allennlp_train(serialization_dir, configfile)
        mml_model_targz = os.path.join(serialization_dir, "model.tar.gz")

    # Print metrics for all iterations
    # print_iterative_training_metrics(checkpoint_dir=checkpoint_dir, all_max_decoding_steps=all_max_decoding_steps)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_examples", type=str, help="WTQ train.examples",
                        default="./resources/data/WikiTableQuestions/data/random-split-1-train.examples")
    parser.add_argument("--dev_examples", type=str, help="WTQ dev.examples",
                        default="./resources/data/WikiTableQuestions/data/random-split-1-dev.examples")
    parser.add_argument("--train_search_dir", type=str, help="Train initial candidates dir",
                        default="./resources/data/wtq/search_non_conservative")
    parser.add_argument("--tables_dir", type=str, help="WikiTablesQuestions Tables dir",
                        default="./resources/data/WikiTableQuestions")
    # parser.add_argument("--erm_model", type=str, default="paired",
    #                     help="Which model to use for ERM model (paired/coverage)")
    # parser.add_argument("--seed", type=str, help="seed")
    parser.add_argument("--ckpt_root", type=str, help="CKPT root")
    parser.add_argument("--split_num", type=int, help="Split-num")


    args = parser.parse_args()
    print(args)
    train_iterative_parser(
        train_examples_file=args.train_examples,
        dev_examples_file=args.dev_examples,
        train_search_dir=args.train_search_dir,
        tables_dir=args.tables_dir,
        checkpoint_root=args.ckpt_root,
        split_num=args.split_num,
        # seed=args.seed,
        # erm_model=args.erm_model,
    )
