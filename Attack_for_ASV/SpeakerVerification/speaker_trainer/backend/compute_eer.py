import os
import glob
import sys
import time
from sklearn import metrics
import numpy
import pdb
from operator import itemgetter
# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):
    # Sort the scores from smallest to largest, and also get the corresponding
    # indexes of the sorted scores.  We will treat the sorted scores as the
    # thresholds at which the the error-rates are evaluated.
    sorted_indexes, thresholds = zip(*sorted(
        [(index, threshold) for index, threshold in enumerate(scores)],
        key=itemgetter(1)))
    sorted_labels = []
    labels = [labels[i] for i in sorted_indexes]
    fnrs = []
    fprs = []

    # At the end of this loop, fnrs[i] is the number of errors made by
    # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
    # is the total number of times that we have correctly accepted scores
    # greater than thresholds[i].
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i-1] + labels[i])
            fprs.append(fprs[i-1] + 1 - labels[i])
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm

    # Now divide by the total number of false negative errors to
    # obtain the false positive rates across all thresholds
    fnrs = [x / float(fnrs_norm) for x in fnrs]

    # Divide by the total number of corret positives to get the
    # true positive rate.  Subtract these quantities from 1 to
    # get the false positive rates.
    fprs = [1 - x / float(fprs_norm) for x in fprs]
    return fnrs, fprs, thresholds

# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target=0.05, c_miss=1, c_fa=1):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold

def compute_eer(target_scores, nontarget_scores):
    if isinstance(target_scores , list) is False:
        target_scores = list(target_scores)
    if isinstance(nontarget_scores , list) is False:
        nontarget_scores = list(nontarget_scores)

    target_scores = sorted(target_scores)
    nontarget_scores = sorted(nontarget_scores)
    target_size = len(target_scores)
    nontarget_size = len(nontarget_scores)

    target_position = 0
    for i in range(target_size-1):
        target_position = i
        nontarget_n = nontarget_size * float(target_position) / target_size
        nontarget_position = int(nontarget_size - 1 - nontarget_n)
        if nontarget_position < 0:
            nontarget_position = 0
        if nontarget_scores[nontarget_position] < target_scores[target_position]:
            break
    th = target_scores[target_position]
    eer = target_position * 1.0 / target_size

    return eer, th


def compute_eer_lst(lst_file):
    target_scores = []
    nontarget_scores = []
    with open(lst_file, 'r') as file:
        for line in file:
            label, _, _, score = line.strip().split(' ')
            if label == '1':
                target_scores.append(float(score))
            elif label == '0':
                nontarget_scores.append(float(score))
                
    if isinstance(target_scores , list) is False:
        target_scores = list(target_scores)
    if isinstance(nontarget_scores , list) is False:
        nontarget_scores = list(nontarget_scores)

    target_scores = sorted(target_scores)
    nontarget_scores = sorted(nontarget_scores)
    target_size = len(target_scores)
    nontarget_size = len(nontarget_scores)

    target_position = 0
    for i in range(target_size-1):
        target_position = i
        nontarget_n = nontarget_size * float(target_position) / target_size
        nontarget_position = int(nontarget_size - 1 - nontarget_n)
        if nontarget_position < 0:
            nontarget_position = 0
        if nontarget_scores[nontarget_position] < target_scores[target_position]:
            break
    th = target_scores[target_position]
    eer = target_position * 1.0 / target_size
    
    
    return eer, th

def calculate_fpr_at_frr(lst_file):
    target_scores = []
    nontarget_scores = []
    with open(lst_file, 'r') as file:
        for line in file:
            label, _, _, score = line.strip().split(' ')
            if label == '1':
                target_scores.append(float(score))
            elif label == '0':
                nontarget_scores.append(float(score))

    if not isinstance(target_scores, list):
        target_scores = list(target_scores)
    if not isinstance(nontarget_scores, list):
        nontarget_scores = list(nontarget_scores)
    target_scores = sorted(target_scores)
    nontarget_scores = sorted(nontarget_scores)

    P = sum(1 for sample in target_scores)  # 正样本数量
    N = sum(1 for sample in nontarget_scores)  # 负样本数量
    FN = int(0.05 * P)  # FRR为5%时对应的False Negative数量
    fpr_at_frr = sum(1 for sample in nontarget_scores if sample >= target_scores[FN]) / N

    return fpr_at_frr

    # P = len(target_scores)  # 正样本数量
    # N = len(nontarget_scores)  # 负样本数量
    # FN = 0  # False Negative计数
    # TP = 0  # True Positive计数
    # fpr_at_frr = None  # FRR为5%时对应的FPR

    # for score in target_scores:
    #     TP += 1
    #     FN = sum(1 for s in target_scores if s < score)
    #     frr = FN / P
    #     fpr = sum(1 for s in nontarget_scores if s < score) / N

    #     if frr <= 0.05:  # 当前FRR小于等于5%
    #         fpr_at_frr = fpr
    #         break

    # return fpr_at_frr

    

def compute_mindcf(sc, lab):
    fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
    mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds)
    return mindcf

if __name__ == '__main__':
    # trials_path = 'speaker_verification/data/trials_1000_victim_RawNet3_score.lst'
    # fpr_at_frr = calculate_fpr_at_frr(trials_path)
    # print(fpr_at_frr)
    trials_path = 'speaker_verification/data/trialsResNetSE34L_score.lst'
    eer, threshold = compute_eer_lst(trials_path)
    print('eer:', eer)
    print('threshold:', threshold)


