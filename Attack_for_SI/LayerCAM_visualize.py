#!/usr/bin/env python
# encoding: utf-8

from argparse import ArgumentParser
from operator import mod
import torch, os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
# from trainer import model_evaluation
from speaker_trainer.module_visualization import Model_visualization
from set_threshold import Load_enroll_spk_embedding

from pytorch_lightning.callbacks import Callback
import sys
import yaml
from pytorch_lightning.utilities import argparse 
setattr(argparse, "_gpus_arg_default", lambda x: 0)

torch.multiprocessing.set_sharing_strategy('file_system')

def cli_main():
    # args
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = Model_visualization.add_model_specific_args(parser)
    args = parser.parse_args()
    if args.test_list_path.split('/')[0] != 'adver-audio':
        args.vis_result = 'Visualization_pdf/0211/salient_no_vad_csi/' + args.visualization_type + '/' + args.nnet_type + '/' + args.target_layer
    else:
        attack_type = args.test_list_path.split('/')[1].split('-')[0] + '_' + args.test_list_path.split('/')[-2]
        args.vis_result = 'Visualization_pdf/0211/' + attack_type + '/' + args.visualization_type + '/' + args.nnet_type + '/' + args.target_layer
    np.random.seed(1)
    args.is_mel = True
    if args.nnet_type == 'RawNet3':
        args.config = 'pre-trained-models/yaml/RawNet3.yaml'
    elif args.nnet_type == 'ECAPATDNN':
        args.config = 'pre-trained-models/yaml/ECAPATDNN.yaml'
    elif args.nnet_type == 'ResNetSE34L':
        args.config = 'pre-trained-models/yaml/ResNetSE34L.yaml'
    elif args.nnet_type == 'ResNetSE34V2':
        args.config = 'pre-trained-models/yaml/ResNetSE34V2.yaml'
    args.is_mel = True
    args.root = './data'
    ## Parse YAML
    def find_option_type(key, parser):
        for opt in parser._get_optional_actions():
            if ('--' + key) in opt.option_strings:
                return opt.type
        raise ValueError
    if args.config is not None:
        with open(args.config, "r") as f:
            yml_config = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in yml_config.items():
            if k in args.__dict__:
                typ = find_option_type(k, parser)
                args.__dict__[k] = typ(v)
            else:
                sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))

    args.enroll_embedding = Load_enroll_spk_embedding(args.root, args.nnet_type)
    model = Model_visualization(**vars(args))

    if args.checkpoint_path is not None:
        state_dict = torch.load(args.checkpoint_path)
        if len(state_dict.keys()) == 1 and "model" in state_dict:
            state_dict = state_dict["model"]
            newdict = {}
            delete_list = []
            for name, param in state_dict.items():
                new_name = '__S__.'+name
                newdict[new_name] = param
                delete_list.append(name)
            state_dict.update(newdict)
            for name in delete_list:
                del state_dict[name]
        # pop loss Function parameter
        loss_weights = []
        for key, value in state_dict.items():
            if "loss" in key:
                loss_weights.append(key)
        for item in loss_weights:
            state_dict.pop(item)
        self_state = model.state_dict()
        for name, param in state_dict.items():
            origname = name
            if name not in self_state:
                name = name.replace('speaker_encoder', "__S__")

                if name not in self_state:
                    print("{} is not in the model.".format(origname));
                    continue;

            if self_state[name].size() != state_dict[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(), state_dict[origname].size()));
                continue;

            self_state[name].copy_(param);
        # state_dict = collections.OrderedDict([(k.split('.', 1)[-1], v) for k, v in state_dict.items()])
        # model.load_state_dict(state_dict, strict=False)
        print("initial parameter from pretrain model {}".format(args.checkpoint_path))

    
    if args.visualization is True:
        # model.hparams.dev_list_path = args.dev_list_path
        model.cuda()
        print(model.__S__.layer4)
        speaker_classification_model = model.__S__.eval()
        model.evaluate_visualization(speaker_classification_model, args.enroll_embedding)

    if args.visualization_metrics is True:
        # model.hparams.dev_list_path = args.dev_list_path
        model.cuda()
        # print(model.speaker_encoder.layer4[2].conv2)
        speaker_classification_model = model.__S__.eval()
        model.visualization_metrics(speaker_classification_model, args.enroll_embedding)
if __name__ == '__main__':  # pragma: no cover
    cli_main()

