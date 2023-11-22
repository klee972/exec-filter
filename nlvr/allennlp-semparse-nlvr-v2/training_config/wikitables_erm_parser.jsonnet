// The Wikitables data is available at https://ppasupat.github.io/WikiTableQuestions/
local utils = import 'utils.libsonnet';

local train_data = std.extVar("TRAIN_DATA");
local dev_data = std.extVar("DEV_DATA");

local tables_dir = std.extVar("TABLES_DIR");

local mml_model = std.extVar("MML_MODEL_TAR");

local maximum_decoding_steps = utils.parse_number(std.extVar("MDS"));

{
  "dataset_reader": {
    "type": "wikitables",
    "lazy": false,
    "output_agendas": true,
    "tables_directory": tables_dir,
    "keep_if_no_logical_forms": true,
//    "max_instances": 200
  },
  "vocabulary": {
    "min_count": {"tokens": 3},
    "tokens_to_add": {"tokens": ["-1"]}
  },
  "train_data_path": train_data,
  "validation_data_path": dev_data,
  "model": {
    "type": "wikitables_erm_parser",
    "question_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 200,
          "trainable": true
        }
      }
    },
    "action_embedding_dim": 100,
    "encoder": {
      "type": "lstm",
      "input_size": 400,
      "hidden_size": 100,
      "bidirectional": true,
      "num_layers": 1
    },
    "entity_encoder": {
      "type": "boe",
      "embedding_dim": 200,
      "averaged": true
    },
    "checklist_cost_weight": 0.2,
    "max_decoding_steps": maximum_decoding_steps,
    "decoder_beam_size": 20,
    "decoder_num_finished_states": 100,
    "attention": {
      "type": "bilinear",
      "vector_dim": 200,
      "matrix_dim": 200
    },
    "dropout": 0.5,
    "mml_model_file": mml_model
  },

  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "sorting_keys": ["question"],
      "batch_size": 10
    }
  },

  "trainer": {
    "checkpointer": {"num_serialized_models_to_keep": 1},
    "num_epochs": 30,
    "patience": 5,
    "validation_metric": "+denotation_acc",
    "cuda_device": -1,
    "optimizer": {
      "type": "sgd",
      "lr": 0.01
    }
  },

  "random_seed": 4536,
  "numpy_seed": 9834,
  "pytorch_seed": 953
}
