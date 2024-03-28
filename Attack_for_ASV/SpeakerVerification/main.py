#!/usr/bin/env python
# encoding: utf-8

from argparse import ArgumentParser
import torch,os
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import time, collections
import yaml

from speaker_trainer import Model, model_evaluation
from pytorch_lightning.callbacks import Callback
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.multiprocessing.set_sharing_strategy('file_system')

def cli_main():
    # args
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = Model.add_model_specific_args(parser)
    args = parser.parse_args()
    args.model_num = 1
    if args.nnet_type == 'RawNet3':
        args.config = 'speaker_verification/pretrained_model/RawNet3_AAM.yaml'
    elif args.nnet_type == 'ECAPATDNN':
        args.config = 'speaker_verification/pretrained_model/Adver_training_ECAPATDNN.yaml'
    elif args.nnet_type == 'ResNetSE34L':
        args.config = 'speaker_verification/pretrained_model/SpecAug_ResNetSE34L.yaml'
    elif args.nnet_type == 'ResNetSE34V2':
        args.config = 'speaker_verification/pretrained_model/ResNetSE34V2.yaml'
    elif args.nnet_type == 'AdverTraining_ECAPATDNN':
        args.config = 'SpeakerVerification/pretrained_model/Adver_training_ECAPATDNN.yaml'
    elif args.nnet_type == 'SpecAug_ResNetSE34L':
        args.config = 'SpeakerVerification/pretrained_model/SpecAug_ResNetSE34L.yaml'
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

    model = Model(**vars(args))
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
                    continue

            if self_state[name].size() != state_dict[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(), state_dict[origname].size()));
                continue

            self_state[name].copy_(param)
        # state_dict = collections.OrderedDict([(k.split('.', 1)[-1], v) for k, v in state_dict.items()])
        # model.load_state_dict(state_dict, strict=False)
        print("initial parameter from pretrain model {}".format(args.checkpoint_path))

    if args.evaluate is not True:
        args.default_root_dir = "exp/" + args.nnet_type + "_" + args.pooling_type + "_" + args.loss_type + "_" + time.strftime('%Y-%m-%d-%H-%M-%S')
        checkpoint_callback = ModelCheckpoint(monitor='loss', save_top_k=args.save_top_k,
                filename="{epoch}_{train_loss:.2f}", dirpath=args.default_root_dir)
        args.checkpoint_callback = checkpoint_callback
        lr_monitor = LearningRateMonitor(logging_interval='step')
        args.callbacks = [model_evaluation(), lr_monitor]
        trainer = Trainer.from_argparse_args(args)
        trainer.fit(model)
    else:
        model.hparams.train_list_path = args.train_list_path
        model.cuda()
        model.eval()
        with torch.no_grad():
            model.cosine_evaluate()

if __name__ == '__main__':  # pragma: no cover
    cli_main()

