
'''
Part of the code is drawn from https://github.com/FAKEBOB-adversarial-attack/FAKEBOB
Paper: Who is Real Bob? Adversarial Attacks on Speaker Recognition Systems (IEEE S&P 2021)
'''

import torch
from torch.utils.data import DataLoader
import numpy as np
from defense.defense import parser_defense
import os, sys
import yaml
from pytorch_lightning import LightningModule, Trainer
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)
parent_path = os.path.dirname(current_path)
if parent_path not in sys.path:
    sys.path.append(parent_path)
from model.defended_model import defended_model
from dataset.Spk10_test import Spk10_test
from dataset.Spk10_imposter import Spk10_imposter
from speaker_trainer.module import Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
test_loader(10*100)与enroll_data(10*10)是相同说话人.
score_target_sv 指的是enroll-test是相同说话人1:1的得分,来自于
    1) test_loader: 与enroll_data相同说话人的得分(即,最高的score)
score_untarget_sv 指的是enroll-test是不同说话人1:1的得分,来自于
    1) test_loader: 与enroll_data不同说话人的得分(即,去掉最高得分,与其他9个说话人的scores)
    2) imposter_loader: imposter_loader与注册说话人全不相同, 所以该项scores为与imposter_loader的全部得分(10个不同说话人-10个scores)
score_target_osi 指的是enroll-test n:1 下 相同的enroll-test得分(即最高的score),来自于
    1) test_loader: 与enroll_data相同的说话人的得分(即最高得分)
score_untarget_osi 指的是enroll-test n:1 下 不同的enroll-test得分中最高的得分,来自于
    1) imposter_loader: imposter_loader与注册说话人全不相同, 所以该项score为这些scores的最大值

IER: Identification Error
    指的是在OSI条件下的错误识别率,其中这个OSI指的是'enroll set-test set '说话人相同,判定条件:1)label识别正确 2)score大与阈值
CSI-E accuracy:
    指的是在闭集SI条件下,label识别正确的概率
"""
class Load_enroll_spk_embedding(object):
    def __init__(self, root, nnet_type):
        enroll_dir = os.path.join(root, 'embedding',nnet_type)
        self.enroll_spk_ids = os.listdir(enroll_dir)
        self.enroll_data = [None] * len(self.enroll_spk_ids)
        self.enroll_spk_ids.sort()
        for idx, spk_emb in enumerate(self.enroll_spk_ids):
            enroll_path = os.path.join(enroll_dir, spk_emb)
            spk_id = spk_emb.split('.')[0]
            self.enroll_data[idx] = {}
            self.enroll_data[idx]['spk_id'] = spk_id
            self.enroll_data[idx]['embedding'] = torch.load(enroll_path, map_location=device)


def set_threshold(score_target, score_untarget):

    if not isinstance(score_target, np.ndarray):
        score_target = np.array(score_target)
    if not isinstance(score_untarget, np.ndarray):
        score_untarget = np.array(score_untarget)

    n_target = score_target.size
    n_untarget = score_untarget.size

    final_threshold = 0.
    min_difference = np.infty
    final_far = 0.
    final_frr = 0.
    for candidate_threshold in score_target:

        frr = np.argwhere(score_target < candidate_threshold).flatten().size * 100 / n_target
        far = np.argwhere(score_untarget >= candidate_threshold).flatten().size * 100 / n_untarget
        difference = np.abs(frr - far)
        if difference < min_difference:
            final_threshold = candidate_threshold
            final_far = far
            final_frr = frr
            min_difference = difference

    return final_threshold, final_frr, final_far

def main(args):
     # Step 1: load speaker model
    if args.nnet_type == 'RawNet3':
        args.config = 'pre-trained-models/yaml/RawNet3.yaml'
    elif args.nnet_type == 'ECAPATDNN':
        args.config = 'pre-trained-models/yaml/ECAPATDNN.yaml'
    elif args.nnet_type == 'ResNetSE34L':
        args.config = 'pre-trained-models/yaml/ResNetSE34L.yaml'
    elif args.nnet_type == 'ResNetSE34V2':
        args.config = 'pre-trained-models/yaml/ResNetSE34V2.yaml'
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
                    continue;

            if self_state[name].size() != state_dict[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(), state_dict[origname].size()));
                continue;

            self_state[name].copy_(param);
        # state_dict = collections.OrderedDict([(k.split('.', 1)[-1], v) for k, v in state_dict.items()])
        # model.load_state_dict(state_dict, strict=False)
        print("initial parameter from pretrain model {}".format(args.checkpoint_path))
    
    model.eval().cuda()

    defense, defense_name = parser_defense(args.defense, args.defense_param, args.defense_flag, args.defense_order)
    # model = defended_model(base_model=model, defense=defense, order=args.defense_order)
    
    #Step2: load dataset
    enroll_dir = os.path.join(args.root, 'Spk10_enroll')
    enroll_spk_ids = os.listdir(enroll_dir)
    test_dataset = Spk10_test(enroll_spk_ids, args.root, return_file_name=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)
    imposter_dataset = Spk10_imposter(enroll_spk_ids, args.root, return_file_name=True)
    imposter_loader = DataLoader(imposter_dataset, batch_size=1, num_workers=0)

    # Step 3: load enroll speaker embedding
    enroll_embedding = Load_enroll_spk_embedding(args.root, args.nnet_type)

    # Step 4: scoring
    # score_target = []
    # score_untarget = []
    score_target_sv = []
    score_untarget_sv = []
    score_target_osi = []
    score_untarget_osi = []
    trues = [] # used to calculate IER for OSI
    max_scores = [] # used to calculate IER for OSI
    decisions = [] # used to calculate IER for OSI

    acc_cnt = 0
    with torch.no_grad():
        for index, (origin, true, file_name) in enumerate(test_loader):
            origin = origin.to(device)
            true = true.cpu().item()
            # print(origin.shape)
            decision, scores = model.make_decision(origin, enroll_embedding, args.threshold)
            decision = decision.cpu().item()
            scores = scores.cpu().numpy().flatten() # (n_spks,)
            print(index, file_name[0], scores, true, decision)
            score_target_sv.append(scores[true])
            score_untarget_sv += np.delete(scores, true).tolist()
            if decision == true:
                score_target_osi.append(scores[true])
            trues.append(true)
            max_scores.append(np.max(scores))
            decisions.append(decision)
            
            if decision == true:
                acc_cnt += 1

        for index, (origin, true, file_name) in enumerate(imposter_loader):
            origin = origin.to(device)
            true = true.cpu().item()
            decision, scores = model.make_decision(origin, enroll_embedding, args.threshold)
            decision = decision.cpu().item()
            scores = scores.cpu().numpy().flatten() # (n_spks,)
            print(index, file_name[0], scores, true, decision)
            score_untarget_sv += scores.tolist()
            score_untarget_osi.append(np.max(scores))

    task = 'OSI'
    threshold, frr, far = set_threshold(score_target_osi, score_untarget_osi)
    IER_cnt = np.intersect1d(np.argwhere(max_scores >= threshold).flatten(),
                np.argwhere(decisions != trues).flatten()).flatten().size
    # # IER: Identification Error, 
    # for detail, refer to 'Who is Real Bob? Adversarial Attacks on Speaker Recognition Systems'
    IER = IER_cnt * 100 / len(trues) 
    print("----- Test of {}-based {}, result ---> threshold: {:.2f}, EER: {:.2f}, IER: {:.2f} -----".format(
        args.nnet_type, task, threshold, max(frr, far), IER))

    # CSI-E accuracy
    print('CSI ACC:', acc_cnt * 100 / len(test_loader))


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', default='./data')
    parser.add_argument('--defense', nargs='+', default=None)
    parser.add_argument('--defense_param', nargs='+', default=None)
    parser.add_argument('--defense_flag', nargs='+', default=None, type=int)
    parser.add_argument('--defense_order', default='sequential', choices=['sequential', 'average'])
    parser.add_argument('--is_mel', action='store_true', default=False)

    parser = Trainer.add_argparse_args(parser)
    parser = Model.add_model_specific_args(parser)
    args = parser.parse_args()
    main(args)