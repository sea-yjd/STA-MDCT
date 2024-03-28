#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
import tqdm
import pandas as pd
import numpy as np
import pickle

def findAllSeqs(data_path):
    path_list = []
    spks = os.listdir(data_path)
    spks.sort()
    for idx, spk in enumerate(spks):
        paths = os.listdir(data_path + '/' + spk)
        for path in paths:
            path_list.append((spk, data_path + '/' + spk + '/' + path, idx))
    return path_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--extension', help='file extension name', type=str, default="wav")
    parser.add_argument('--data_dir', help='dataset dir', type=str, default="adver-audio/ResNetSE34L-CSI-Spk10_test/None/S2A/S2A-[10, 60, 20, 6, 44.0, 0.5, 2.0, MDCT]")    # TODO
    parser.add_argument('--target_label_path', default='Target_label/ECAPATDNN-CSI-None-Spk10_test-False.target_label')
    parser.add_argument('--type', default='enroll')
    # parser.add_argument('--data_list_path', help='list save path', type=str, default="data_list")
    parser.add_argument('--speaker_level', help='list save path', type=int, default=1)
    args = parser.parse_args()

    if args.target_label_path is not None:
        with open(args.target_label_path, 'rb') as reader:
            name2target = pickle.load(reader)

    speaker_name, utt_paths, utt_spk_int_labels, target = [], [], [], []
    path_list = findAllSeqs(args.data_dir)
    # path_list = pd.read_csv('data/test_copy.csv', index_col=False)
    # path_list_ = path_list['utt_paths']
    # for idx, line in enumerate(path_list_):
    #     speaker_name.append(line[0])
    #     utt_paths.append(line[1])
    #     utt_spk_int_labels.append(line[2])
    #     name = line.split('/')[-1].split('.')[0]
    #     target.append(name2target[name])

    for idx, line in enumerate(path_list):
        speaker_name.append(line[0])
        utt_paths.append(line[1])
        utt_spk_int_labels.append(line[2])
        name = line[1].split('/')[-1].split('.')[0]
        target.append(name2target[name])
    # speaker_name = path_list['speaker_name']
    # utt_paths = path_list['utt_paths']
    # utt_spk_int_labels = path_list['utt_spk_int_labels']

    csv_dict = {"speaker_name": speaker_name, 
            "utt_paths": utt_paths,
            "utt_spk_int_labels": utt_spk_int_labels,
            "target_label": target
            }
    df = pd.DataFrame(data=csv_dict)
    output_path = args.data_dir + '/' + 'data.csv'
    try:
        df.to_csv(output_path)
        print(f'Saved data list file at {output_path}')
    except OSError as err:
        print(f'Ran in an error while saving {output_path}: {err}')


