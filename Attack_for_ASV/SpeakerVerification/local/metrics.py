#!/usr/bin/env python
# encoding: utf-8

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
from speaker_verification.speaker_trainer.backend import compute_mindcf
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
def get_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default = 0.051)
    parser.add_argument('--victim_model', help='', type=str, default="ECAPATDNNadt", choices=['ECAPATDNN', 'ResNetSE34L', 'RawNet3', 'ResNetSE34V2', 'ResNetSE34Laug', 'ECAPATDNNadt'])

    # In order to generate wave list file (.lst)
    parser.add_argument('--wave_path', help='',type=str, default='Adversarial_examples_1000/ResNetSE34L/S2A_data_epsilon40_it10_alpha4_srho0.75_N20_sigma44.0_MDCT_I/wav')
    parser.add_argument('--wave_list', default='Adversarial_examples_1000/ResNetSE34L/S2A_data_epsilon40_it10_alpha4_srho0.75_N20_sigma44.0_MDCT_I/adv_trials.lst')

    # In order to generate 'label-en-test-adv-score.lst'
    parser.add_argument('--trials_score_file', help='trials with score', type=str, default='Adversarial_examples_1000/ResNetSE34L/S2A_data_epsilon40_it10_alpha4_srho0.75_N20_sigma44.0_MDCT_I/adv_trials_victim_ECAPATDNNadt_score.lst')
    parser.add_argument('--source_trials_file', help='source trials file', type=str, default='speaker_verification/data/trials_1000.lst')

    args = parser.parse_args()
    return args

def get_list(wave_path, wave_list):
    f = open(wave_list,"w")
    for root, dirs, files in os.walk(wave_path):
        files.sort()
        for file in files:
            f.write("{}\n".format(root+'/'+file))
    return f

def get_label_en_test_adv_score(trials_score_file, source_trials_file, dst_file):
    lines1 = np.loadtxt(trials_score_file, dtype=str)
    len_test = len(lines1)
    lines2 = np.loadtxt(source_trials_file, dtype=str)
    lines2 = lines2[0:len_test]
    f = open(dst_file, 'w')
    for idx, item in enumerate(lines2):
        f.write("{} {} {} {} {}\n".format(item[0], item[1], item[2], lines1[idx][2], lines1[idx][3]))
    return

def preprocess(data_root):
    data, fs = sf.read(data_root)
    norm_factor = np.abs(data).max()
    wav = data / norm_factor
    return wav

def Lp(benign_xx, adver_xx, p):
    return LA.norm(adver_xx-benign_xx, p)

def L2(benign_xx, adver_xx):
    return Lp(benign_xx, adver_xx, 2)

def L0(benign_xx, adver_xx):
    return Lp(benign_xx, adver_xx, 0)

def L1(benign_xx, adver_xx):
    return Lp(benign_xx, adver_xx, 1)

def Linf(benign_xx, adver_xx):
    return Lp(benign_xx, adver_xx, np.infty)

def SNR(benign_xx, adver_xx):
    noise = adver_xx - benign_xx
    power_noise = np.sum(noise ** 2)
    power_benign = np.sum(benign_xx ** 2)
    snr = 10 * np.log10(power_benign / max(power_noise, 1e-7))
    return snr 

def PESQ(benign_xx, adver_xx):
    # pesq_value = pesq(16_000, benign_xx, adver_xx, 'wb' if bits == 16 else 'nb')
    pesq_value = pesq(16_000, benign_xx, adver_xx, 'wb')
    return pesq_value

def STOI(benign_xx, adver_xx, fs=16_000):
    d = stoi(benign_xx, adver_xx, fs, extended=False)
    return d

def main(args):
    # get_list(args.wave_path, args.wave_list)
    assert args.victim_model == os.path.split(args.trials_score_file)[1].split('_')[-2]
    if args.victim_model == 'ECAPATDNN' or args.victim_model == 'RawNet3' or args.victim_model == 'ECAPATDNNadt':
        args.threshold = 0.03
    if args.victim_model == 'ResNetSE34L':
        args.threshold = 0.051
    if args.victim_model == 'ResNetSE34Laug':
        args.threshold = 0.049
    if args.victim_model == 'ResNetSE34V2':
        args.threshold = 0.034
    dst_file = os.path.split(args.trials_score_file)[0] + '/' + '_'.join(os.path.split(args.trials_score_file)[1].split('_')[:-1]) + '_label_en-test-adv-score.lst'
    if not os.path.exists(dst_file):
        get_label_en_test_adv_score(args.trials_score_file, args.source_trials_file, dst_file)
    
    metrics_path = os.path.split(args.trials_score_file)[0] + '/' + '_'.join(os.path.split(args.trials_score_file)[1].split('_')[2:4]) + '_metrics_results.csv'
    FAR = 0
    if not os.path.exists(metrics_path):
        trials = np.loadtxt(dst_file, dtype=str)
        bar = tqdm.tqdm(trials)
        asr =  0
        snr_all = 0
        pesq_all = 0
        stoi_all = 0
        l0_all = 0
        l1_all = 0
        l2_all = 0
        linf_all = 0
        result = []
        FAR = 0
        num = len(trials)
        for idx, line in enumerate(bar):
            label = line[0]
            x1 = preprocess(line[2])
            x2 = preprocess(line[3])
            shape1 = x1.shape[-1]
            shape2 = x2.shape[-1]
            if shape1 < shape2:
                x2 = x2.tolist()
                del x2[shape1: shape2]
                x2 = np.array(x2)
            else:
                x1 = x1.tolist()
                del x1[shape2: shape1]
                x1 = np.array(x1)
            snr = SNR(x1, x2)
            pesq = PESQ(x1, x2)
            stoi = STOI(x1, x2)
            l0 = L0(x1, x2)
            l1 = L1(x1, x2)
            l2 = L2(x1, x2)
            linf = Linf(x1, x2)
            
            snr_all += snr
            pesq_all += pesq
            stoi_all += stoi
            l0_all += l0
            l1_all += l1
            l2_all += l2
            linf_all += linf
            threshold = args.threshold
            if int(line[0]) == 1:
                if float(line[4]) < threshold: 
                    asr += 1
            if int(line[0]) == 0:
                if float(line[4]) > threshold:
                    asr += 1
                    FAR += 1
            result.append({'index':idx,  'label':label,  "ASR": (asr / (idx+1)), "SNR":snr, "PESQ": pesq, "STOI": stoi, "L0":l0, "L1":l1, "L2":l2, "Linf": linf, "FAR": None})
            output_str = "label:{}, ASR:{:5.2f}, SNR:{:5.2f}, PESQ: {:5.2f}, STOI: {:5.2f}, L0:{:5.2f}, L1:{:5.2f}, L2,{:5.2f}, Linf: {:5.2f}".format(line[0],asr/(idx+1), snr,pesq, stoi,l0,l1,l2, linf)
            bar.set_description(output_str)
        FAR = FAR / (num / 2)
        result.append({'index': None, 'label':None, "ASR": asr / (idx+1), "SNR":snr_all / num, "PESQ": pesq_all / num, "STOI": stoi_all / num, "L0":l0_all/num, "L1":l1_all / num, "L2":l2_all / num, "Linf": linf_all / num, "FAR": FAR})
        if len(result) > 0:
            pd.DataFrame(result).to_csv(metrics_path)
        bar.close()
    print('test_file:{}'.format(args.trials_score_file))
    all_labels = []
    all_scores = []
    score_trials_file = np.loadtxt(args.trials_score_file, dtype=str)
    for idx, items in enumerate(score_trials_file):
        all_labels.append(int(items[0]))
        all_scores.append(items[-1].tolist())
    mindcf = compute_mindcf(all_scores, all_labels)
    print('mindcf:{}'.format(mindcf))
    if FAR == 0:
        trials = np.loadtxt(dst_file, dtype=str)
        num_ = len(trials)
        for idx, line in enumerate(trials):
            if int(line[0]) == 0:
                if float(line[4]) > args.threshold:
                    FAR += 1
        FAR = FAR / (num_ / 2)
    print('FAR:{:.3f}'.format(FAR))
    
    return

if __name__ == "__main__":
    args = get_option()
    main(args)