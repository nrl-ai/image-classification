import os
import torch
from datetime import datetime
from easydict import EasyDict
from pathlib import Path

def make_if_not_exist(path):
    if os.path.exists(path) is not True:
        os.makedirs(path)


def get_default_config(args):
    conf = EasyDict()
    
    conf.lr = args.learning_rate
    conf.milestones = [7, 15, 21, 27, 35]
    conf.gamma = 0.1
    conf.epochs = args.epochs
    conf.batch_size = args.batch_size
    conf.nb_workers = 8
    conf.pretrain = True
    conf.net = args.network
    conf.input_size = args.input_size
    conf.nb_classes = args.num_classes
    conf.data_dir = args.dataset
    conf.board_load_every = {'train': 30, 'val': 15}
    conf.save_every = 50
    conf.path_pretrain = args.path_pretrain
    conf.job_name = args.job_name
    conf.model_path = os.path.join( "out_snapshot", conf.job_name)
    make_if_not_exist(conf.model_path)

    return conf

def get_export_config(args):
    conf = EasyDict()
    conf.net = args.network
    conf.nb_classes = args.num_classes
    conf.input_size = args.input_size
    conf.path_model = args.path_model
    return conf
