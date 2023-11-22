local utils = import 'utils.libsonnet';
local train_data = std.extVar("TRAIN_DATA");
local dev_data = std.extVar("DEV_DATA");
local cuda_device = utils.parse_number(std.extVar("CUDA"));
local maximum_decoding_steps = utils.parse_number(std.extVar("MDS"));
local epochs = utils.parse_number(std.extVar("EPOCHS"));

local mml_model = std.extVar("MML_MODEL_TAR");

{
  "dataset_reader": {
    "type": "nlvr_v2_paired",
    "lazy": false,
    "output_agendas": false,
    "mode": "train",
//    "max_instances": 500,
  },
  "validation_dataset_reader": {
    "type": "nlvr_v2_paired",
    "lazy": false,
    "output_agendas": false,
    "mode": "test",
//    "max_instances": 100
},

  "vocabulary": {
    "non_padded_namespaces": ["rule_labels", "denotations"]
  },
  "train_data_path": train_data,
  "validation_data_path": dev_data,
  "model": {
    "type": "nlvr_paired_parser",
    "sentence_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 50,
          "trainable": true
        }
      }
    },
    "action_embedding_dim": 50,
    "encoder": {
      "type": "lstm",
      "input_size": 50,
      "hidden_size": 30,
      "num_layers": 1,
      "bidirectional": true
    },

    "beam_size": 10,
    "max_decoding_steps": maximum_decoding_steps,
    "attention": {"type": "dot_product"},
    "dropout": 0.2,
    "initial_mml_model_file": mml_model,
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "sorting_keys": ["sentence"],
      "batch_size": 4
    }
  },
  "trainer": {
    "checkpointer": {"num_serialized_models_to_keep": 1},
    "num_epochs": epochs,
    "patience": 15,
    "cuda_device": cuda_device,
    "validation_metric": "+consistency",
    "optimizer": {
      "type": "adam",
      "lr": 0.0005
    }
  },

  "random_seed": utils.parse_number(std.extVar("SEED")),
  "numpy_seed": utils.parse_number(std.extVar("SEED")),
  "pytorch_seed": utils.parse_number(std.extVar("SEED"))
}
