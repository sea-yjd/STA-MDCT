#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
import numpy as np
import os
# 并去除相同的wav
def findAllSeqs(dirName,
                extension='.wav',
                speaker_level=1):
    r"""
    Lists all the sequences with the given extension in the dirName directory.
    Output:
        outSequences, speakers
        outSequence
        A list of tuples seq_path, speaker where:
            - seq_path is the relative path of each sequence relative to the
            parent directory
            - speaker is the corresponding speaker index
        outSpeakers
        The speaker labels (in order)
    The speaker labels are organized the following way
    \dirName
        \speaker_label
            \..
                ...
                seqName.extension
    Adjust the value of speaker_level if you want to choose which level of
    directory defines the speaker label. Ex if speaker_level == 2 then the
    dataset should be organized in the following fashion
    \dirName
        \crappy_label
            \speaker_label
                \..
                    ...
                    seqName.extension
    Set speaker_label == 0 if no speaker label will be retrieved no matter the
    organization of the dataset.
    """
    if dirName[-1] != os.sep:
        dirName += os.sep
    prefixSize = len(dirName)
    speakersTarget = {}
    outSequences = []
    print("finding {}, Waiting...".format(extension))
    for root, dirs, filenames in tqdm.tqdm(os.walk(dirName, followlinks=True)):
        filtered_files = [f for f in filenames if f.endswith(extension)]
        if len(filtered_files) > 0:
            speakerStr = (os.sep).join(
                root[prefixSize:].split(os.sep)[:speaker_level])
            if speakerStr not in speakersTarget:
                speakersTarget[speakerStr] = len(speakersTarget)
            speaker = speakersTarget[speakerStr]
            for filename in filtered_files:
                full_path = os.path.join(root, filename)
                outSequences.append((speaker, full_path))
    outSpeakers = [None for x in speakersTarget]

    for key, index in speakersTarget.items():
        outSpeakers[index] = key

    print("find {} speakers".format(len(outSpeakers)))
    print("find {} utterance".format(len(outSequences)))

    return outSequences, outSpeakers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--voxceleb1_root', help='voxceleb1_root', type=str, default="/home/jdyao/data_yjd/voxceleb1/vox1_test_wav")
    parser.add_argument('--src_trials_path', help='src_trials_path', type=str, default="/home/jdyao/PycharmProjects/spot-adv-by-vocoder-main/voxceleb1_test_v2.txt")
    parser.add_argument('--dst_trials_path', help='dst_trials_path', type=str, default="/home/jdyao/PycharmProjects/spot-adv-by-vocoder-main/speaker_verification/data/trials.lst")
    parser.add_argument('--apply_vad', action='store_true', default=False)
    
    parser.add_argument('--path', default='')
    args = parser.parse_args()

    trials = np.loadtxt(args.src_trials_path, dtype=str)

    f = open(args.dst_trials_path, "a+")
    for item in trials:
        enroll_path = os.path.join(args.voxceleb1_root,  item[1])
        test_path = os.path.join(args.voxceleb1_root,  item[2])
        if args.apply_vad:
            enroll_path = enroll_path.strip("*.wav") + "*.vad"
            test_path = test_path.strip("*.wav") + "*.vad"
        f.write("{} {} {}\n".format(item[0], enroll_path, test_path))

