import os
import argparse
import numpy as np
import os
import numpy as np
from numpy import linalg as LA
from pesq import pesq
from pystoi import stoi
import soundfile as sf
import tqdm
import pandas as pd
import soundfile
import threading
from scipy import signal
from scipy.io import wavfile
import torch
from random import sample
def get_option():
    parser = argparse.ArgumentParser()
    # In order to generate wave list file (.lst)
    parser.add_argument('--wave_path', help='',type=str, default='Adversarial_examples_ensemble/ECAPATDNN_ResNetSE34V2/S2A/epsilon30_it10_alpha4_srho0.5_N20_sigma8.0_DCT_I/wav')
    parser.add_argument('--wave_list', default='Adversarial_examples_ensemble/ECAPATDNN_ResNetSE34V2/S2A/epsilon30_it10_alpha4_srho0.5_N20_sigma8.0_DCT_I/wavlist.lst')
    parser.add_argument('--clean_trials', default='speaker_verification/data/trials_1000.lst')
    parser.add_argument('--tg_trials', default='Adversarial_examples_ensemble/ECAPATDNN_ResNetSE34V2/S2A/epsilon30_it10_alpha4_srho0.5_N20_sigma8.0_DCT_I/adv_trials.lst')
    args = parser.parse_args()
    return args
def get_list(wave_path, wave_list):
    f = open(wave_list,"w")
    for root, dirs, files in os.walk(wave_path):
        files.sort()
        for file in files:
            f.write("{}\n".format(root+'/'+file))
    return f
    # compute scores

def load_wav(path):
    sample_rate, audio = wavfile.read(path)
    audio = torch.FloatTensor(audio)
    audio = audio.cuda()
    return sample_rate, audio

def compute_scores(model):
    trials = np.loadtxt('Adversarial_examples/ECAPATDNN/CW_data_ini-c1e-12_step6_iter100_lr0.01_conf0.01_lossL2/adv_trials.lst', dtype=str)
    scores = {}
    for idx, items in enumerate(trials):
        enroll_path = items[1]
        test_path = items[2]
        sample_rate, enroll_wav = load_wav(enroll_path)
        _, test_wav = load_wav(test_path)
        enroll_embed = model.extract_speaker_embedding(enroll_wav).squeeze(0)
        test_embed = model.extract_speaker_embedding(test_wav).squeeze(0)
        score = enroll_embed.dot(test_embed.T)
        denom = torch.norm(enroll_embed) * torch.norm(test_embed)
        score = score/denom
        scores[idx] = score.tolist()
    return scores

def extract_part_clean_trials(path, num):
    tgt_path = path.split('.')[0] + '_random_' + str(num) + '.lst'
    raw_trials = np.loadtxt(path, dtype=str)
    target_trials_list = ([line for line in raw_trials if line[0] == '1'])
    non_target_trials_list = ([line for line in raw_trials if line[0] == '0'])
    target_num_trials = sample(target_trials_list, int(num/2))
    nontarget_num_trials = sample(non_target_trials_list, int(num/2))
    new_trials = []
    for i in range(int(num/2)):
        new_trials.append(target_num_trials[i])
        new_trials.append(nontarget_num_trials[i])
    f = open(tgt_path, 'w')
    for idx, item in enumerate(new_trials):
        f.write('{} {} {}\n'.format(item[0], item[1], item[2]))
    return

def main(args):
    get_list(args.wave_path, args.wave_list)
    f = open(args.tg_trials, 'w')
    line1 = np.loadtxt(args.clean_trials, dtype=str)
    line2 = np.loadtxt(args.wave_list, dtype=str)
    line1 = line1[0: len(line2)]
    for idx, item in enumerate(line1):
        f.write('{} {} {}\n'.format(item[0], item[1], line2[idx]))
    
    # audio_path = 'audio/00009.wav'
    # sample_rate, audio  = wavfile.read(audio_path)
    # i = 1
    # for i in range(6):
    #     s_audio = audio / 2**i
    #     newpath = audio_path.split('.')[0] + '_gen_' + str(i) + '.wav'
    #     wavfile.write(newpath, sample_rate, s_audio.astype(np.int16))



if __name__ == "__main__":
    args = get_option()
    # extract_part_clean_trials(args.clean_trials, 1000)
    main(args)
