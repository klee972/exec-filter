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

'''
demo:                                       0.432356 | 0.444291

wc0: len(worldcode) < 5, threshold 0.5
wc1: len(worldcode) < 10, threshold 0.5
wc2: len(worldcode) < 5, threshold 0.4
wc3: len(worldcode) < 10, threshold 0.4

ww0: 6epo, len(worldcode) < 5, threshold 0.4      437155
ww1: 6epo, len(worldcode) < 10, threshold 0.4     450046  444982  434622  444982
ww2: 6epo, len(worldcode) < 5, threshold 0.3      443831
ww3: 6epo, len(worldcode) < 10, threshold 0.3     440147

ww4: 10epo, len(worldcode) < 10, threshold 0.4    439917
ww5: 6epo, len(worldcode) < 15, threshold 0.5     446823
ww6: 6epo, len(worldcode) < 15, threshold 0.4     439917
ww7: 6epo, len(worldcode) < 15, threshold 0.3     446133

ww1_v2: 6epo, len(worldcode) < 10, threshold 0.4, execution_result_v2   443140

rsmpl0: 6epo, len(worldcode) < 10, threshold 0.3             441298  433932  444521  444061 | 440953
rsmpl1: 6epo, len(worldcode) < 10, threshold 0.4             441759  438536  445902 | 442066

GIVE SCORE TO EXCUTION ERROR BELOW
rsmpl2: 6epo-onlyonce, len(worldcode) < 10, threshold 0.2    436004  432781  450506  441759 | 440262
rsmpl3: 6epo-onlyonce, len(worldcode) < 10, threshold 0.3            440608  439917

USING HYPERPARAMETERS IN THE PAPER
    rsmpl4: 8epo-onlyonce, len(worldcode) < 10, threshold 0.4  439687  436004  431400  
    rsmpl5: 8epo-onlyonce, len(worldcode) < 10, threshold 0.2

rsmpl_ft0: finetune after 15epo normal training, len(worldcode) < 10, threshold 0.2, slot filling prob
    Dev:  430237  435182  433063  430943 | 432356
    Test: 447974  452578  447053  445442 | 448262 *

rsmpl_ft_hv0: same as rsmpl_ft0 but using hard vote
    Dev:  427411  433416  425645  427058 | 428383
    Test: 445442  446363  440838  439917 | 443140

rsmpl_ft1: finetune after 15epo normal training, len(worldcode) < 10, threshold 0.2, model prob
    Dev:  
    Test: 

prob0: filter by prob, 6epo, threshold 0.4  
'''