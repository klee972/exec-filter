from pathlib import Path

class Config():
    def __init__(self):
        # data dir
        self.reader_pkl = "processed/wikitable_glove_42B_minfreq_3.pkl"

        # sketch file
        self.sketch_pkl = "processed/train.pkl" 
        self.sketch_test_pkl = "processed/test.pkl"
        self.sketch_action_file = "processed/sketch.actions"

        # config for programmer
        # self.token_embed_size = 300 # Input Fixed Embedding Size
        # self.var_token_size  = 256 # Input Linear Projection Size
        # self.pos_embed_size = 64 # POS Embedding Size
        # self.prod_embed_size = 436 # Production Rule Embedding Size
        # self.column_type_embed_size = 16
        # self.token_indicator_size = 16
        # self.column_indicator_size = 16
        # self.op_embed_size = 128 # Operator Embedding Size
        # self.token_rnn_size = 218 # Encoder Hidden Size ?
        # self.token_dropout = 0.45 # Encoder Dropout
        # self.prod_rnn_size = 218 # AP Encoder Hidden Size
        # self.prod_dropout = 0.25 # AP Encoder Dropout
        # self.slot_hidden_score_size = 436 # MLP Hidden Size
        # self.slot_dropout = 0.25 # MLP Dropout

        self.token_embed_size = 300
        self.var_token_size  = 256
        self.token_dropout = 0.5
        self.token_rnn_size = 256
        self.token_indicator_size = 16
        self.pos_embed_size = 64
        self.slot_dropout = 0.25
        self.prod_embed_size = 512
        self.prod_rnn_size = 512
        self.prod_dropout = 0.25
        self.op_embed_size = 128

        self.column_type_embed_size = 16
        self.column_indicator_size = 16
        self.slot_hidden_score_size = 512

        self.model_type = "struct"

        # config for training
        self.lr = 1e-5
        self.l2 = 1e-5
        self.clip_norm = 3

        self.gpu_id = 3
        self.seed = 1337
        # seeds: 3264 5 42 1337

        self.wc_start_epo = 1
        self.filter_only_once = True
        self.hard_vote = False
        self.examples_file = "./examples/rsmpl_ft1_0.txt"
        # self.examples_file = "./examples/test.txt"

        self.ckpt_path = "./checkpoints/demo.model"

    def __repr__(self):
        return str(vars(self))

config = Config()
