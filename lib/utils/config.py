import os
import os.path as osp
import numpy as np
import torch
from easydict import EasyDict as edict
import os
# import wandb
from datetime import datetime


def read_run_cfg(model, dataset):
    options = dict()
    options["model_name"] = model
    options["epoch_train"] = 80

    data_dir = './data'
    cache_dir = './cache'
    options['gpus'] = '1'
    if model == "UNet":
        options["batch_size_train"] = 4
        options["learning_rate"] = 2 * 1e-4
        options["batch_size_test"] = 1
    elif model == "SUNet":
        options["batch_size_train"] = 8
        options["learning_rate"] = 2 * 1e-4
        options["batch_size_test"] = 1
    elif model == "SDUNet":
        options["batch_size_train"] = 8
        options["learning_rate"] = 2 * 1e-4
        options["batch_size_test"] = 1
    else:
        options["batch_size_train"] = 16
        options["learning_rate"] = 4 * 1e-4
        options["batch_size_test"] = 1

    options["data_set"] = dataset
    # options["batch_size_train"] = 32

    options["lr_delay"] = 0.2
    options["num_epoch_lr_decay"] = 40
    options["momentum"] = 0.9
    options["mgpus"] = True

    options["start_epoch"] = 0

    # file dir cfg
    if dataset == 'mass':
        options["data_dir"] = os.path.join(data_dir, 'rs/mass/mass_road')
    elif dataset == 'deepglobe':
        options["data_dir"] = os.path.join(data_dir, 'rs/deepglobe')

    result_suff = '1029'

    options["model_cache"] = os.path.join(
        cache_dir, 'model_cache/rs_road_train_{}'.format(result_suff), model, dataset)
    if not os.path.exists(options["model_cache"]):
        os.makedirs(options["model_cache"])
    # for train
    options["result_dir"] = os.path.join(
        cache_dir, 'result/rs_road_train_{}'.format(result_suff), model, dataset)
    if not os.path.exists(options["result_dir"]):
        os.makedirs(options["result_dir"])

    # for test
    test_suff = '1029'
    options["result_dir_test"] = os.path.join(
        cache_dir, 'result/rs_road_test_{}'.format(test_suff), model, dataset)
    if not os.path.exists(options["result_dir_test"]):
        os.makedirs(options["result_dir_test"])

    options['test_model_state_dict'] = os.path.join(
        cache_dir, 'model_cache/{}'.format("cache_mass"), model, dataset)
    # if not os.path.exists(options["test_model_state_dict"]):
    #     os.makedirs(options["test_model_state_dict"])
    # options["demo_dir"] = os.path.join(root_, 'ymx/result/demo/rs_road', model)
    # options["check_point_path"] = os.path.join(root_, 'ymx/model_cache', 'rs_road', model, dataset,
    #                                            'check_point_180.pth.tar')

    # file dir cfg
    if dataset == 'mass':
        # shape (1500, 1500, 3)
        # (0,988/2,988)
        options["img_size"] = (1500, 1500, 3)
        options["split_point"] = [0, 494, 988]
    elif dataset == 'deepglobe':
        # shape (1300, 1300, 3)
        options["img_size"] = (1300, 1300, 3)
        options["split_point"] = [0, 394, 788]

    return options