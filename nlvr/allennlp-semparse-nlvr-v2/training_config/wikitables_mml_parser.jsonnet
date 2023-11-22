// The Wikitables data is available at https://ppasupat.github.io/WikiTableQuestions/
local utils = import 'utils.libsonnet';

local train_data = std.extVar("TRAIN_DATA");
local dev_data = std.extVar("DEV_DATA");

local tables_dir = std.extVar("TABLES_DIR");
local train_search_dir = std.extVar("TRAIN_SEARCH_DIR");

local maximum_decoding_steps = utils.parse_number(std.extVar("MDS"));


{
  "dataset_reader": {
    "type": "wikitables",
    "tables_directory": tables_dir,
    "offline_logical_forms_directory": train_search_dir,
    "max_offline_logical_forms": 60,
    "lazy": false,
//    "max_instances": 200
  },
  "validation_dataset_reader": {
    "type": "wikitables",
    "tables_directory": tables_dir,
    "keep_if_no_logical_forms": true,
    "lazy": false,
    "max_instances": 200
  },
  "vocabulary": {
    "min_count": {"tokens": 3},
    "tokens_to_add": {"tokens": ["-1"]}
  },
  "train_data_path": train_data,
  "validation_data_path": dev_data,
  "model": {
    "type": "wikitables_mml_parser",
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
    "decoder_beam_search": {
      "beam_size": 10
    },
    "max_decoding_steps": maximum_decoding_steps,
    "attention": {
      "type": "bilinear",
      "vector_dim": 200,
      "matrix_dim": 200
    },
    "dropout": 0.5
  },

  "data_loader": {
    "batch_sampler": {
      "type": "basic",
      "sampler": {"type": "random"},
      "batch_size": 1,
      "drop_last": false,
    },
  },

  "trainer": {
    "checkpointer": {"num_serialized_models_to_keep": 1},
    "num_epochs": 100,
    "patience": 10,
    "cuda_device": -1,
    "grad_norm": 5.0,
    "validation_metric": "+denotation_acc",
    "optimizer": {
      "type": "sgd",
      "lr": 0.1
    },
    "learning_rate_scheduler": {
      "type": "exponential",
      "gamma": 0.99
    }
  },

  "random_seed": 4536,
  "numpy_seed": 9834,
  "pytorch_seed": 953
}