local utils = import 'utils.libsonnet';
local cuda_device = utils.parse_number(std.extVar("CUDA"));
local epochs = utils.parse_number(std.extVar("EPOCHS"));

local train_data = std.extVar("TRAIN_DATA");
local dev_data = std.extVar("DEV_DATA");

local maximum_decoding_steps = utils.parse_number(std.extVar("MDS"));
local mml_model = std.extVar("MML_MODEL_TAR");

{
  "dataset_reader": {
    "type": "nlvr",
    "lazy": false,
    "output_agendas": true,
    "mode": "train",
//    "max_instances": 50
  },
  "validation_dataset_reader": {
    "type": "nlvr",
    "lazy": false,
    "output_agendas": true,
    "mode": "test",
//    "max_instances": 50
  },

  "vocabulary": {
    "non_padded_namespaces": ["rule_labels", "denotations"]
  },
  "train_data_path": train_data,
  "validation_data_path": dev_data,
  "model": {
    "type": "nlvr_coverage_parser",
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
    "normalize_beam_score_by_length": false,
    "max_decoding_steps": maximum_decoding_steps,
    "checklist_cost_weight": 0.4,
    "penalize_non_agenda_actions": false,
    "dropout": 0.2,
    "initial_mml_model_file": mml_model,
    "attention": {"type": "dot_product"},
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
    "patience": 10,
    "cuda_device": cuda_device,
    "validation_metric": "+consistency",
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    }
  },

  "random_seed": utils.parse_number(std.extVar("SEED")),
  "numpy_seed": utils.parse_number(std.extVar("SEED")),
  "pytorch_seed": utils.parse_number(std.extVar("SEED"))
}
