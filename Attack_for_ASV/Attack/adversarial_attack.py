import os
import torch
import sys
from argparse import ArgumentParser
from scipy.io import wavfile
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
from tqdm import tqdm
import collections
from attack_algorithm import *
from numpy import linalg as LA
import yaml

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
if parent_path not in sys.path:
    sys.path.append(parent_path)

from SpeakerVerification.speaker_trainer import Model
from SpeakerVerification.speaker_trainer.backend import compute_eer,compute_mindcf
from SpeakerVerification.local.try_ import compute_scores
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.backends.cudnn.benchmark = True 
torch.multiprocessing.set_sharing_strategy('file_system')

class Adversarial_Attack_Helper(object):
    def __init__(self, model, alpha=3.0, restarts=1, num_iters=5, epsilon=15, adv_save_dir="data/ACG_adv_data", device="cuda", eta = 2.0, W_set = [0,5,10,15,20,25,50,90], rho = 1.0, 
                        s_rho=0.5, momentum=1, sigma=16, N=10, attack_type = '', initial_const=1e-3, binary_search_steps=5,
                        max_iter=1000, stop_early=True, stop_early_iter=1000, lr=1e-2, confidence=0, dist_loss='L2', spectrum_transform_type='DCT', 
                        attack_transform='I', uni_r=2.0, var_number=10, meta_learning=None):
        self.model = model
        self.alpha = alpha
        self.restarts = restarts
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.eta = eta
        self.W_set = W_set
        self.rho = rho
        self.s_rho = s_rho
        self.momentum = momentum
        self.sigma = sigma
        self.N = N
        self.attack_type = attack_type
        self.initial_const = initial_const
        self.binary_search_steps = binary_search_steps
        self.max_iter = max_iter
        self.stop_early = stop_early
        self.stop_early_iter = stop_early_iter
        self.lr = lr
        self.confidence = confidence
        self.dist_loss = dist_loss
        self.spectrum_transform_type = spectrum_transform_type
        self.attack_transform = attack_transform
        self.momentum = momentum
        if self.alpha == None:
            self.alpha = float(format(epsilon / num_iters, ".2f"))
        if attack_transform == 'MI' or attack_transform == 'NI':
            attack_transform_name = attack_transform + str(momentum)
        elif attack_transform == 'I':
            attack_transform_name = attack_transform
        elif attack_transform == 'PGD':
            attack_transform_name = attack_transform
        elif arrack_transform == 'VMI':
            attack_transform_name = attack_transform + 'uni_r' + str(uni_r) + '_' + 'var_number' + str(var_number)
        if attack_type == 'S2A':
            alpha = args.alpha = self.alpha
            if meta_learning == 'Hybrid':
                adv_save_dir = adv_save_dir + '_epsilon{}_it{}_alpha{}_srho{}_N{}_sigma{}_{}_{}_{}'.format(epsilon, num_iters, alpha, s_rho, N, sigma, spectrum_transform_type, attack_transform_name, meta_learning)
            else:
                adv_save_dir = adv_save_dir + '_epsilon{}_it{}_alpha{}_srho{}_N{}_sigma{}_{}_{}'.format(epsilon, num_iters, alpha, s_rho, N, sigma, spectrum_transform_type, attack_transform_name)
        elif attack_type == 'CW':  # unenable early stop 
            adv_save_dir = adv_save_dir + '_ini-c{}_step{}_iter{}_lr{}_conf{}_loss{}'.format(initial_const, binary_search_steps, max_iter, lr, confidence, dist_loss)
        elif attack_type == 'BIM':
            adv_save_dir = adv_save_dir + '_epsilon{}_it{}_alpha{}_{}'.format(epsilon, num_iters, alpha, attack_transform_name)
        elif attack_type == 'PGD': # 初始化是随机初始化，p范数，这里用2
            adv_save_dir = adv_save_dir + '_epsilon{}_it{}_alpha{}'.format(epsilon, num_iters, self.alpha)
        elif attack_type == "ACG":
            adv_save_dir = adv_save_dir + '_epsilon{}_it{}_alpha{}_rho{}'.format(epsilon, num_iters, alpha, rho)
        else:
            adv_save_dir = adv_save_dir + '_epsilon{}_it{}_alpha{}'.format(epsilon, num_iters, alpha)
        self.adv_save_dir = adv_save_dir
        args.adv_save_dir = adv_save_dir

        if not os.path.exists(os.path.join(adv_save_dir, "wav")):
            os.makedirs(os.path.join(adv_save_dir, "wav"))

        self.trials = self.model.trials
        self.adv_trials_path = os.path.join(adv_save_dir, "adv_trials.lst")
        self.device = device
        self.model.eval()
        if self.device == "cuda":
            self.model.cuda()

    def evaluate(self, trials=None):
        if trials is None:
            trials = self.trials
        with torch.no_grad():
            eer, th = self.model.cosine_evaluate(trials) 

    def attack(self):
        # adversarial attack example generation
        # if os.path.exists(self.adv_trials_path):
        #     os.remove(self.adv_trials_path)
        adv_trials_file = open(self.adv_trials_path, "a+")
        target_score = []
        nontarget_score = []
        for idx, item in enumerate(tqdm(self.trials)):
            des_path = args.adv_save_dir + '/wav/' + '%08d' % idx + '.wav'
            if os.path.exists(des_path):
                print('*' * 40, '%08d' % idx, 'Exists, Skip', '*' * 40)
                continue

            if args.attack_type == 'BIM':
                label, enroll_path, adv_test_path, score = bim_adversarial_attack_step(model, idx, item, **vars(args))
            if args.attack_type == 'PGD':
                label, enroll_path, adv_test_path, score = pgd_adversarial_attack_step(model, idx, item, **vars(args))
            if args.attack_type == 'ACG':
                label, enroll_path, adv_test_path, score = Auto_conjugate_gradient_attack(model, idx, item, **vars(args))
            if args.attack_type == 'S2A':
                label, enroll_path, adv_test_path, score = Spectrum_Simulation_Attack(model, idx, item, **vars(args))
            if args.attack_type == 'CW':
                label, enroll_path, adv_test_path, score = CW_Attack(model, idx, item, **vars(args))
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            with torch.no_grad():
                adv_trials_file.write("{} {} {}\n".format(label, enroll_path, adv_test_path))
                if label == 0:
                    nontarget_score.append(score)
                else:
                    target_score.append(score)
        eer, th = compute_eer(target_score, nontarget_score)
        print("EER: {:.3f} %".format(eer*100))
        print("Threshold: {:.3f} %".format(th))
        return self.adv_trials_path
        

    def load_wav(self, path):
        sample_rate, audio = wavfile.read(path)
        audio = torch.FloatTensor(audio)
        if self.device == "cuda":
            audio = audio.cuda()
        return sample_rate, audio


if __name__ == "__main__":
    # args
    parser = ArgumentParser()
    parser.add_argument('--alpha', help='', type=float, default=4)
    parser.add_argument('--restarts', help='', type=int, default=1)
    parser.add_argument('--num_iters', help='', type=int, default=10)
    parser.add_argument('--epsilon', help='', type=int, default=40)
    parser.add_argument('--adv_save_dir', help='', type=str, default="Adversarial_examples")
    parser.add_argument('--attack_type', help='', type=str, default='S2A', choices=['PGD','ACG','S2A','CW','BIM'])
    parser.add_argument('--device', help='', type=str, default="cuda")
    parser.add_argument('--evaluate_only', action='store_true', default=False)
    # ACG attack parameters
    parser.add_argument('--eta', help='', type=float, default=2.0)  # 之前的eta一直等于2
    parser.add_argument('--W_set', help='', type =list, default=[5, 10, 15, 20])   ## 查询间断点集合
    parser.add_argument('--rho', help='', type=float, default=0.75)
    # S2A attack parameters
    parser.add_argument('--s_rho', help='Tuning factor', type=float, default=0.5)
    parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")
    parser.add_argument("--N", type=int, default=20, help="The number of Spectrum Transformations")
    parser.add_argument("--sigma", type=float, default=44.0, help="Std of random noise")
    parser.add_argument('--spectrum_transform_type', type=str, default='MDCT', help='', choices=['DCT','MDCT', 'FFT','Time'])
    parser.add_argument('--meta_learning', type=str, default=None, choices=['Hybrid', None])
    parser.add_argument('--attack_transform', type=str, default='PGD', choices=['I', 'MI', 'NI', 'VMI', 'PGD'])
    # CW attack parameters
    parser.add_argument('--initial_const', help='', type=float, default=1e-6)
    parser.add_argument('--binary_search_steps', help='', type=int, default=6)
    parser.add_argument('--max_iter', help='', type=int, default=100)
    parser.add_argument('--stop_early', action='store_false', default=True)
    parser.add_argument('--stop_early_iter', help='', type=int, default=1000)
    parser.add_argument('--lr', help='', type=float, default=0.01)
    parser.add_argument('--confidence', help='', type=float, default=0.05)
    parser.add_argument('--dist_loss', help='', default='L2', choices=['L2', 'Linf', 'RMS', 'SNR'])
    # VMI attack parameters
    parser.add_argument('--uni_r', help='uniform distribution parameters, the bound for variance tuning', type=float, default=2.0)
    parser.add_argument('--var_number', help='the number of images for variance tuning', type=int, default=10)
    # Ensemble attack parameters
    parser.add_argument('--model_num', type=int, default=1)
    parser.add_argument('--weight', type=list, default=[1])
    parser = Trainer.add_argparse_args(parser)
    parser = Model.add_model_specific_args(parser)
    args = parser.parse_args()
    args.adv_save_dir = args.adv_save_dir + '/' + args.nnet_type + '/' + args.attack_type + '_data'
    
    if args.nnet_type == 'RawNet3':
        args.config = parent_path + '/SpeakerVerification/pretrained_model/RawNet3.yaml'
    elif args.nnet_type == 'ECAPATDNN':
        args.config = parent_path + '/SpeakerVerification/pretrained_model/ECAPATDNN.yaml'
    elif args.nnet_type == 'ResNetSE34L':
        args.config = parent_path + '/SpeakerVerification/pretrained_model/ResNetSE34L.yaml'
    elif args.nnet_type == 'ResNetSE34V2':
        args.config = parent_path + '/SpeakerVerification/pretrained_model/ResNetSE34V2.yaml'
    elif args.nnet_type == 'AdverTraining_ECAPATDNN':
        args.config = parent_path + '/SpeakerVerification/pretrained_model/Adver_training_ECAPATDNN.yaml'
    elif args.nnet_type == 'SpecAug_ResNetSE34L':
        args.config = parent_path + '/SpeakerVerification/pretrained_model/SpecAug_ResNetSE34L.yaml'
        
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
    args.checkpoint_path = parent_path + args.checkpoint_path
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
        print("initial parameter from pretrain model {}".format(args.checkpoint_path))
    helper = Adversarial_Attack_Helper(model, args.alpha, args.restarts, args.num_iters, args.epsilon, args.adv_save_dir, args.device, args.eta, args.W_set, args.rho, args.s_rho, args.momentum, args.sigma, args.N, args.attack_type, args.initial_const, args.binary_search_steps,
                        args.max_iter, args.stop_early, args.stop_early_iter, args.lr, args.confidence, args.dist_loss, args.spectrum_transform_type, args.attack_transform, args.uni_r, args.var_number,args.meta_learning)

    # no_frame_scores = compute_scores(model)
    if (args.evaluate_only):
        print("evaluate in trials {}".format(args.trials_path))
        print("victim model:{}".format(args.nnet_type))
        print("file:{}".format(args.trials_path))
        helper.evaluate()
    else:
        # print("evaluate in raw data")
        # helper.evaluate()
        print("attacking ")
        print('\n')
        print("Algorithm {} is used to attack Model {}".format(args.attack_type, args.nnet_type))
        helper.attack()