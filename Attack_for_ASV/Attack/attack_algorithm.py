from argparse import ArgumentParser
import os
import sys
import argparse
import torch
from scipy.io import wavfile
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torchaudio
from pytorch_lightning.callbacks import Callback
from numpy import linalg as LA
from tqdm import tqdm
import collections
from spectrum_transform import Spectrum_Trans_

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
if parent_path not in sys.path:
    sys.path.append(parent_path)

from SpeakerVerification.speaker_trainer import Model
from SpeakerVerification.speaker_trainer.backend import compute_eer
torch.backends.cudnn.benchmark = True 
torch.multiprocessing.set_sharing_strategy('file_system')
bits = 16

def load_wav(path):
    sample_rate, audio = wavfile.read(path)
    audio = torch.FloatTensor(audio)
    audio = audio.cuda()
    return sample_rate, audio

def cw_load_wav(path):
    audio, sample_rate = torchaudio.load(path)
    audio = torch.FloatTensor(audio)
    audio = audio.cuda()
    return sample_rate, audio

def save_audio(wav_path, samplerate, wav):
    wav = wav.squeeze()
    if 0.9 * wav.max() <= 1 and 0.9 * wav.min() >= -1:
        wav = wav * (2 ** (bits -1))
    if type(wav) == torch.Tensor:
        wav = wav.detach().cpu().numpy()
    wav = wav.astype(np.int16)
    wavfile.write(wav_path, samplerate, wav)

def loadWAV(filename, max_frames=400, evalmode=True, num_eval=10):
    '''
    Remark! we will set max_frames=0 for evaluation.
    If max_frames=0, then the returned feat is a whole utterance.
    '''
    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    audio, sample_rate = torchaudio.load(filename)
    audio = audio.squeeze()
    audiosize = audio.shape[-1]

    # padding
    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        audio = np.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]

    if evalmode:
        startframe = np.linspace(0, audiosize-max_audio, num=num_eval)
    else:
        startframe = np.array([np.int64(random.random()*(audiosize-max_audio))])

    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])
    feat = np.stack(feats, axis=0).astype(float)
    return torch.FloatTensor(audio), torch.FloatTensor(feat)

def div_frame(audio):
    max_frames=400
    num_eval=10   
    audio = audio.squeeze()
    audiosize = audio.shape[-1]
    max_audio = max_frames * 160 + 240
    # padding
    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        audio = np.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]
    startframe = np.linspace(0, audiosize-max_audio, num=num_eval)
    feats = []
    for asf in startframe:
        feats.append(audio[int(asf):int(asf)+max_audio])
    feat = torch.stack(feats, axis=0)
    return feat

def random_init_pertutbation(ori_audio):
    # p = 2
    b, n = ori_audio.shape
    eta = torch.randn(size=(b, n+2), device=ori_audio.device)
    eta = eta / eta.norm(p=2, dim=-1, keepdim=True)
    return eta[:, :n]
    

def Auto_conjugate_gradient_attack(models, idx, item, **kwargs):
    args = argparse.Namespace(**kwargs)
    if args.model_num == 1:
        model_ = []
        model_.append(models)
        models = model_
    eta_list = []
    loss_list = []
    nabla_list = []
    s_list = []
    x_list = []
    beta_list = []
    
    label = item[0]
    enroll_path = item[1]
    test_path = item[2]
    # init best_score and alpha
    label = int(label)
    if label == 1:
        best_score = torch.tensor(float('inf')).cuda()
        alpha = args.alpha*(-1.0)
    else:
        best_score = torch.tensor(float('-inf')).cuda()
        alpha = args.alpha*(1.0)

    # load data
    samplerate, enroll_wav = load_wav(enroll_path)
    samplerate, test_wav = load_wav(test_path)

    upper = test_wav + args.epsilon
    lower = test_wav - args.epsilon

    x_pre = x_adv = test_wav
    x_list.append(test_wav)
    test_wav.requires_grad = True
    # x_pre.requires_grad = True
    # x_adv.requires_grad = True

    # compute gradient
    beta_list.append(0)
    eta_list.append(args.eta)
    s_pre = 0
    score = 0
    for i, model in enumerate(models):
        with torch.no_grad():
            enroll_embedding = model.extract_speaker_embedding(enroll_wav).detach().squeeze(0)      # [256]
        test_embedding = model.extract_speaker_embedding(x_pre).squeeze(0)   # [256]
        sco= enroll_embedding.dot(test_embedding.T)
        denom = torch.norm(enroll_embedding.detach()) * torch.norm(test_embedding.detach())
        sco = sco/denom
        sco.backward()
        score += args.weight[i] * sco
        s_pre += args.weight[i] * test_wav.grad.detach()
        test_wav.grad.zero_()
    # auto conjugate gradient method
    loss_list.append(score)
    nabla_list.append(s_pre)
    s_list.append(s_pre)

    for k in range(args.num_iters):
        x_new = (x_list[k] + alpha * eta_list[k] * torch.sign(s_list[k])).detach()
        x_new = torch.min(torch.max(x_new, lower), upper)
        x_list.append(x_new)
        x_new.requires_grad = True
        score_new = 0
        score_adv = 0
        grad_x_new = torch.zeros(test_wav.shape).cuda()
        for i, model in enumerate(models):
            with torch.no_grad():
                enroll_embedding = model.extract_speaker_embedding(enroll_wav).detach().squeeze(0)
                test_embedding_adv = model.extract_speaker_embedding(x_adv).detach().squeeze(0)
            test_embedding_new = model.extract_speaker_embedding(x_new).squeeze(0)
            score_adv_l = enroll_embedding.dot(test_embedding_adv.T)
            denom_adv = torch.norm(enroll_embedding.detach()) * torch.norm(test_embedding_adv.detach())
            score_adv_l = score_adv_l / denom_adv
            score_adv += args.weight[i] * score_adv_l
            score_new_l = enroll_embedding.dot(test_embedding_new.T)
            denom_new = torch.norm(enroll_embedding.detach()) * torch.norm(test_embedding_new.detach())
            score_new_l = score_new_l / denom_new
            score_new += args.weight[i] * score_new_l
            score_new_l.backward()
            grad_x_new += args.weight[i] * (x_new.grad.detach())
            x_new.grad.zero_()
        loss_list.append(score_new)
        y_list = []
        if alpha * score_adv <= alpha * score_new:
            x_adv = torch.min(torch.max(x_list[k+1], lower), upper)
            x_pre = x_list[k]
            s_pre = s_list[k]
        eta_list.append([])
        eta_list[k+1] = eta_list[k]
        nabla_list.append(grad_x_new)

        with torch.no_grad():
            if k in args.W_set[1:]:
                if judgement(k, x_list, enroll_embedding, eta_list, loss_list, **vars(args)):
                    eta_list[k+1] = (eta_list[k] / 2)
                    x_list[k+1] = x_adv
                    x_list[k] = x_pre
                    s_list[k] = s_pre
            y_list.append(nabla_list[-1] - nabla_list[-2])
            beta_list.append(torch.dot(-nabla_list[-1], y_list[-1]) / (torch.dot(s_list[-1], y_list[-1])))
            s_list.append(nabla_list[-1] + beta_list[-1] * s_list[-1])
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    final_score_all = 0
    for i, model in enumerate(models):
        with torch.no_grad():
            enroll_embedding = model.extract_speaker_embedding(enroll_wav).squeeze(0)
            test_embedding = model.extract_speaker_embedding(x_adv).squeeze(0)
            final_score = enroll_embedding.dot(test_embedding.T)
            denom = torch.norm(enroll_embedding) * torch.norm(test_embedding)
            final_score = final_score/denom
            final_score_all += args.weight[i] * final_score

    if label == 1 and best_score >= final_score_all:
        best_score = torch.min(best_score, final_score_all)
    elif label == 0 and best_score <= final_score_all:
        best_score = torch.max(best_score, final_score_all)
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    # Get Adversarial Attack wav
    adv_wav = x_adv
    adv_wav = adv_wav.cpu().detach().numpy()
    final_score = best_score.cpu().detach().numpy()

    # save attack test wav
    idx = '%08d' % idx
    adv_test_path = os.path.join(args.adv_save_dir, "wav", idx+".wav")
    wavfile.write(adv_test_path, samplerate, adv_wav.astype(np.int16))
    return label, enroll_path, adv_test_path, final_score


def judgement(k, x, enroll_embedding, eta_list, loss_list, **kwargs):
    args = argparse.Namespace(**kwargs)
    flag = False
    count = 0
    j = args.W_set.index(k)
    for i in range(args.W_set[j-1], k-1):
        # test_embedding_new = self.model.extract_speaker_embedding(x[i+1]).squeeze(0)
        # score_new = enroll_embedding.dot(test_embedding_new.T)
        # denom_new = torch.norm(enroll_embedding) * torch.norm(test_embedding_new)
        # score_new = score_new / denom_new

        # test_embedding_old = self.model.extract_speaker_embedding(x[i]).squeeze(0)
        # score_old = enroll_embedding.dot(test_embedding_old.T)
        # denom_old = torch.norm(enroll_embedding) * torch.norm(test_embedding_old)
        # score_old = score_old / denom_old

        if args.alpha * loss_list[i+1] > args.alpha * loss_list[i]:
            count += 1
    if count < args.rho * (k - args.W_set[j-1]):
        flag = True
    if eta_list[k] == eta_list[args.W_set[j-1]] and loss_list[k] == loss_list[args.W_set[j-1]]:
        flag = True
    return flag


def Spectrum_Simulation_Attack(models, idx, item, **kwargs):
    args = argparse.Namespace(**kwargs)
    if args.model_num == 1:
        model_ = []
        model_.append(models)
        models = model_
    Spectrum_Trans = Spectrum_Trans_(args.spectrum_transform_type)
    label = item[0]
    enroll_path = item[1]
    test_path = item[2]
    grad = 0
    # load data
    samplerate, enroll_wav = load_wav(enroll_path)
    samplerate, test_wav = load_wav(test_path)
    adv_wav = test_wav
    best_adv_wav = test_wav
    # init best_score and alpha
    label = int(label)
    if label == 1:
        best_score = torch.tensor(float('inf')).cuda()
        alpha = args.alpha*(-1.0)
    else:
        best_score = torch.tensor(float('-inf')).cuda()
        alpha = args.alpha*(1.0)
    
    for t in range(args.num_iters):
        noise_freq = 0
        noise = 0
        for n in range(args.N):
            if args.spectrum_transform_type == 'Time':
                x_time = adv_wav
                x_time.requires_grad = True
            else:
                gauss = (torch.randn(adv_wav.shape) * args.sigma).cuda()
                x_spectrum = Spectrum_Trans.spectrum_T(adv_wav)
                x_gauss_spectrum = x_spectrum.cuda() + Spectrum_Trans.spectrum_T(gauss).cuda()
                mask = (torch.rand_like(x_gauss_spectrum) * 2 * args.s_rho + 1 - args.s_rho).cuda()
                x_i_spectrum = Spectrum_Trans.i_spectrum_T(x_gauss_spectrum * mask).cuda()
                if x_i_spectrum.shape < adv_wav.shape:
                    x_i_spectrum = torch.nn.functional.pad(x_i_spectrum, (0, len(adv_wav)-len(x_i_spectrum)))
                elif x_i_spectrum.shape > adv_wav.shape:
                    x_i_spectrum = x_i_spectrum[:len(adv_wav)]
                x_i_spectrum = x_i_spectrum.detach()
                x_i_spectrum.requires_grad = True
            noise_model = 0
            score_model = 0
            for i, model in enumerate(models):
                with torch.no_grad():
                    enroll_embedding = model.extract_speaker_embedding(enroll_wav).detach().squeeze(0)
                test_embedding = model.extract_speaker_embedding(x_i_spectrum).squeeze(0)
                score = enroll_embedding.dot(test_embedding.T)
                denom = torch.norm(enroll_embedding) * torch.norm(test_embedding)
                score = score/denom
                score_model += args.weight[i] * score
                score.backward()
                noise_model += args.weight[i] * x_i_spectrum.grad.detach()
                x_i_spectrum.grad.zero_()
            noise_freq += noise_model
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        noise_freq = noise_freq / args.N

        # Freq  --> Time
        noise_time = 0
        if args.meta_learning == 'Hybrid':
            noise_freq.requires_grad = True
            for cid, model in enumerate(models):
                test_embedding = model.extract_speaker_embedding(adv_wav + noise_freq).squeeze(0)
                # cosine score
                score = enroll_embedding.dot(test_embedding.T)
                denom = torch.norm(enroll_embedding) * torch.norm(test_embedding)
                score = score/denom
                # compute grad and update delta
                score.backward(retain_graph=True)  ## retain_graph=True
                k = noise_freq.grad.detach()
                noise_time += args.weight[cid] * k
                noise_freq.grad.zero_()

        noise_freq_time = noise_freq + noise_time
        noise = noise_freq_time

        if args.attack_transform == 'MI':
            noise = noise / torch.abs(noise)
            noise = args.momentum * grad + noise
            grad = noise
        elif args.attack_transform == 'PGD':
            all_k_norm = 1. / torch.linalg.norm(noise, keepdim=True, ord=2) * noise
            adv_wav = adv_wav + alpha*all_k_norm
        else:
            adv_wav = (adv_wav + alpha*noise.sign()).clamp(-1*args.epsilon + test_wav, args.epsilon + test_wav)
    final_score_all = 0
    for i, model in enumerate(models):
        with torch.no_grad():
            enroll_embedding = model.extract_speaker_embedding(enroll_wav).squeeze(0)
            test_embedding = model.extract_speaker_embedding(adv_wav).squeeze(0)
            final_score = enroll_embedding.dot(test_embedding.T)
            denom = torch.norm(enroll_embedding) * torch.norm(test_embedding)
            final_score = final_score/denom
            final_score_all += args.weight[i] * final_score

    if label == 1 and best_score >= final_score_all:
        best_score = torch.min(best_score, final_score_all)
        adv_wav_best = adv_wav
    elif label == 0 and best_score <= final_score_all:
        best_score = torch.max(best_score, final_score_all)
        adv_wav_best = adv_wav

    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    # Get Adversarial Attack wav
    adv_wav = adv_wav_best.cpu().detach().numpy()
    final_score = best_score.cpu().detach().numpy()

    # save attack test wav
    idx = '%08d' % idx
    adv_test_path = os.path.join(args.adv_save_dir, "wav", idx+".wav")
    wavfile.write(adv_test_path, samplerate, adv_wav.astype(np.int16))
    return label, enroll_path, adv_test_path, final_score


def CW_Attack(models, idx, item, **kwargs):
    args = argparse.Namespace(**kwargs)
    if args.model_num == 1:
        model_ = []
        model_.append(models)
        models = model_
    label = item[0]
    enroll_path = item[1]
    test_path = item[2]
    n_audios = len(label)
    # load data
    enroll_wav, enroll_stack_wav = loadWAV(str(enroll_path))
    test_wav, test_stack_wav = loadWAV(str(test_path))
    enroll_wav, enroll_stack_wav, test_wav, test_stack_wav = enroll_wav.cuda(), enroll_stack_wav.cuda(), test_wav.cuda(), test_stack_wav.cuda()
    # const c boundary
    const = torch.tensor([args.initial_const] * n_audios, dtype=torch.float, device=enroll_wav.device)
    lower_bound = torch.tensor([0] * n_audios, dtype=torch.float, device=enroll_wav.device)
    upper_bound = torch.tensor([1e10] * n_audios, dtype=torch.float, device=enroll_wav.device)

    global_best_l2 = [np.infty] * n_audios  # 越小越好
    global_best_adver_x = test_wav.clone()
    global_best_score = [-2] * n_audios
    cosine_score_best = None
    # cosim_score = # label=1,越小越好;label=0,越大越好
    grad = [0 for _ in range(len(kwargs['weight']))]
    for i in range(args.binary_search_steps):
        args.modifier = torch.zeros_like(test_wav, dtype=torch.float, requires_grad=True, device=test_wav.device)
        best_l2 = [np.infty] * n_audios
        best_score = [-2] * n_audios
        for n_iter in range(args.max_iter + 1):
            input_x = args.modifier + test_wav 
            input_x_stack = div_frame(input_x)
            grad_final = 0
            for cid, model in enumerate(models):
                with torch.no_grad():
                    enroll_embedding = model.extract_speaker_embedding(enroll_stack_wav)
                input_x_embedding = model.extract_speaker_embedding(input_x_stack)
                scores = torch.matmul(enroll_embedding, input_x_embedding.T)
                scores = torch.mean(scores)
                denom = torch.linalg.norm(enroll_embedding) * torch.linalg.norm(input_x_embedding)
                scores = scores/denom
                label = int(label)
                if label == 1:
                    loss1 = scores + args.confidence - args.threshold
                else:
                    loss1 = -scores + args.threshold + args.confidence
                # loss2 = torch.sum(torch.square(input_x - test_wav), dim=(1,2))
                loss2 = LA.norm(input_x.cpu().detach().numpy() - test_wav.cpu().detach().numpy(), 2)
                loss = const * loss1 + loss2

                if n_iter < args.max_iter:
                    loss.backward(retain_graph=True)
                    grad[cid] = args.modifier.grad + 0
                    args.modifier.grad.zero_()
            for id_, s in enumerate(grad):
                grad_final += args.weight[id_] * s
            args.modifier.grad = grad_final
            args.optimizer = torch.optim.Adam([args.modifier], lr=args.lr)
            args.optimizer.step()
            args.modifier.grad.zero_()
            
            scores = [scores.detach().cpu().numpy().tolist()]
            loss = loss.detach().cpu().numpy().tolist()
            loss1 = [loss1.detach().cpu().numpy().tolist()]
            loss2 = [loss2.tolist()]
            print("step: {}, c: {}, iter: {}, loss: {}, loss1: {}, loss2: {}, cosine_score: {}".format(
                        i, const.detach().cpu().numpy(), n_iter, 
                        loss, loss1, loss2, scores))
            for ii,(l2, l1, score) in enumerate(zip(loss2, loss1, scores)):
                if l1 <= 0 and l2 < best_l2[ii]:
                    best_l2[ii] = l2
                    best_score[ii] = 1
                if l1 <= 0 and l2 < global_best_l2[ii]:
                    global_best_l2[ii] = l2
                    global_best_adver_x = input_x
                    cosine_score_best = score
                    global_best_score[ii] = 1
        for jj, k in enumerate(best_score):
            if k != -2:
                upper_bound[jj] = min(upper_bound[jj], const[jj])
                if upper_bound[jj] < 1e9:
                    const[jj] = (lower_bound[jj] + upper_bound[jj]) / 2
            else:
                lower_bound[jj] = max(lower_bound[jj], const[jj])
                if upper_bound[jj] < 1e9:
                    const[jj] = (lower_bound[jj] + upper_bound[jj]) / 2
                else:
                    const[jj] *= 10
        print(const.detach().cpu().numpy(), best_l2, global_best_l2)

    # Get Adversarial Attack wav
    if cosine_score_best is not None:
        adv_wav = global_best_adver_x.cpu().detach().numpy()
        final_score = cosine_score_best
    else:
        adv_wav = input_x.cpu().detach().numpy()
        final_score = scores[0]

    # save attack test wav
    idx = '%08d' % idx
    adv_test_path = os.path.join(args.adv_save_dir, "wav", idx+".wav")
    save_audio(adv_test_path, 16000, adv_wav)
    # wavfile.write(adv_test_path, samplerate, adv_wav.astype(np.int16))
    return label, enroll_path, adv_test_path, final_score


def bim_adversarial_attack_step(models, idx, item, **kwargs):
    args = argparse.Namespace(**kwargs)
    if args.model_num == 1:
        model_ = []
        model_.append(models)
        models = model_
    if args.attack_transform == 'MI' or args.attack_transform == 'I' or args.attack_transform == 'VMI':
        ## MI/I
        label = item[0]
        enroll_path = item[1]
        test_path = item[2]
        grad=[0 for _ in range(len(kwargs['weight']))]
        # load data
        samplerate, enroll_wav = load_wav(enroll_path)
        samplerate, test_wav = load_wav(test_path)
        max_delta = torch.zeros_like(test_wav).cuda()

        # init best_score and alpha
        label = int(label)
        if label == 1:
            best_score = torch.tensor(float('inf')).cuda()
            alpha = args.alpha*(-1.0)
        else:
            best_score = torch.tensor(float('-inf')).cuda()
            alpha = args.alpha*(1.0)

        for i in range(args.restarts):
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            delta = torch.zeros_like(test_wav, requires_grad=True).cuda()
            for t in range(args.num_iters):
                # extract test speaker embedding
                all_k = 0
                for cid, model in enumerate(models):
                    with torch.no_grad():
                        enroll_embedding = model.extract_speaker_embedding(enroll_wav).squeeze(0)
                    test_embedding = model.extract_speaker_embedding(test_wav + delta).squeeze(0)
                    # cosine score
                    score = enroll_embedding.dot(test_embedding.T)
                    denom = torch.norm(enroll_embedding) * torch.norm(test_embedding)
                    score = score/denom

                    # compute grad and update delta
                    score.backward()  ## retain_graph=True
                    k = delta.grad.detach()
                    if args.attack_transform == 'MI':
                        k = k / torch.abs(k)
                        k = args.momentum * grad[cid] + k
                        grad[cid] = k
                    all_k += args.weight[cid] * k
                    delta.grad.zero_()
                delta.data = (delta + alpha * all_k.sign()).clamp(-1*args.epsilon, args.epsilon)
                delta.grad.zero_()
            final_score = torch.zeros(len(args.weight))
            final_score_ = 0
            for id_, model in enumerate(models):
                with torch.no_grad():
                    enroll_embedding = model.extract_speaker_embedding(enroll_wav).squeeze(0)
                    test_embedding = model.extract_speaker_embedding(test_wav+delta).squeeze(0)
                    final_score[id_] = enroll_embedding.dot(test_embedding.T)
                    denom = torch.norm(enroll_embedding) * torch.norm(test_embedding)
                    final_score[id_] = final_score[id_]/denom
            for id_, s in enumerate(final_score):
                final_score_ += args.weight[id_] * s
            if label == 1 and best_score >= final_score_:
                max_delta = delta.data
                best_score = torch.min(best_score, final_score_)
            elif label == 0 and best_score <= final_score_:
                max_delta = delta.data
                best_score = torch.max(best_score, final_score_)

        # Get Adversarial Attack wav
        adv_wav = test_wav + max_delta
        adv_wav = adv_wav.cpu().detach().numpy()
        final_score = best_score.cpu().detach().numpy()

        # save attack test wav
        idx = '%08d' % idx
        adv_test_path = os.path.join(args.adv_save_dir, "wav", idx+".wav")
        wavfile.write(adv_test_path, samplerate, adv_wav.astype(np.int16))

    if args.attack_transform == 'NI':
        ## ==============================================
        ## NI 
        label = item[0]
        enroll_path = item[1]
        test_path = item[2]
        grad = 0
        # load data
        samplerate, enroll_wav = load_wav(enroll_path)
        samplerate, test_wav = load_wav(test_path)
        # init best_score and alpha
        label = int(label)
        grad=[0 for _ in range(len(kwargs['weight']))]
        if label == 1:
            best_score = torch.tensor(float('inf')).cuda()
            alpha = args.alpha*(-1.0)
        else:
            best_score = torch.tensor(float('-inf')).cuda()
            alpha = args.alpha*(1.0)

        x_adv = test_wav
        x_adv.requires_grad = True
        for i in range(args.restarts):
            for t in range(args.num_iters):
                all_k = 0
                for cid, model in enumerate(models):
                    x_adv.requires_grad = True
                    with torch.no_grad():
                        enroll_embedding = model.extract_speaker_embedding(enroll_wav).squeeze(0)
                    x_nes = x_adv + alpha * args.momentum * grad[cid]
                    embedding_nes = model.extract_speaker_embedding(x_nes).squeeze(0)
                # cosine score
                    score = enroll_embedding.dot(embedding_nes.T)
                    denom = torch.norm(enroll_embedding) * torch.norm(embedding_nes)
                    score = score/denom
                # compute grad and update delta
                    score.backward(retain_graph=True)
                    k = x_adv.grad.detach()
                    k = k / torch.abs(k)
                    k = args.momentum * grad[cid] + k
                    grad[cid] = k
                all_k += args.weight[cid] * k
                x_adv = ((x_adv + alpha*k.sign()).clamp(-1*args.epsilon + test_wav, args.epsilon + test_wav)).detach()

            final_score = torch.zeros(len(args.weight))
            final_score_ = 0
            for id_, model in enumerate(models):
                with torch.no_grad():
                    enroll_embedding = model.extract_speaker_embedding(enroll_wav).squeeze(0)
                    test_embedding = model.extract_speaker_embedding(x_adv).squeeze(0)
                    final_score[id_] = enroll_embedding.dot(test_embedding.T)
                    denom = torch.norm(enroll_embedding) * torch.norm(test_embedding)
                    final_score[id_] = final_score[id_]/denom
            for id_, s in enumerate(final_score):
                final_score_ += args.weight[id_] * s

            if label == 1 and best_score >= final_score_:
                adv_wav = x_adv
                best_score = torch.min(best_score, final_score_)
            elif label == 0 and best_score <= final_score_:
                adv_wav = x_adv
                best_score = torch.max(best_score, final_score_)

        # Get Adversarial Attack wav
        adv_wav = adv_wav.cpu().detach().numpy()
        final_score = best_score.cpu().detach().numpy()

        # save attack test wav
        idx = '%08d' % idx
        adv_test_path = os.path.join(args.adv_save_dir, "wav", idx+".wav")
        wavfile.write(adv_test_path, samplerate, adv_wav.astype(np.int16))

    """
    if args.attack_transform == 'NRI':
        ## ==============================================
        ## NRI 
        beta2 = 0.9
        mu = 1
        grad = 0
        gama = 0
        eps = 1e-8
        label = item[0]
        enroll_path = item[1]
        test_path = item[2]
        # load data
        samplerate, enroll_wav = load_wav(enroll_path)
        samplerate, test_wav = load_wav(test_path)
        # init best_score and alpha
        label = int(label)
        if label == 1:
            best_score = torch.tensor(float('inf')).cuda()
            alpha = args.alpha*(-1.0)
        else:
            best_score = torch.tensor(float('-inf')).cuda()
            alpha = args.alpha*(1.0)
        enroll_embedding = model.extract_speaker_embedding(enroll_wav).squeeze(0)
        x_adv = test_wav

        for i in range(args.restarts):
            for t in range(args.num_iters):
                x_adv.requires_grad = True
                x_nes = x_adv + alpha * args.momentum * grad
                embedding_nes = model.extract_speaker_embedding(x_nes).squeeze(0)
                # cosine score
                score = enroll_embedding.dot(embedding_nes.T)
                denom = torch.norm(enroll_embedding) * torch.norm(embedding_nes)
                score = score/denom
                
                final_score = score
                if label == 1 and best_score > final_score:
                    adv_wav = x_adv
                    best_score = torch.min(best_score, final_score)
                elif label == 0 and best_score < final_score:
                    adv_wav = x_adv
                    best_score = torch.max(best_score, final_score)

                # compute grad and update delta
                score.backward(retain_graph=True)
                k = x_adv.grad.detach()
                k = k / torch.abs(k)
                k = args.momentum * grad + k
                grad = k
                gama = beta2 * gama + (1 - beta2) * (k**2)
                alpha = mu * alpha * k / torch.sqrt(gama + eps)
                x_adv = ((x_adv + alpha*k.sign()).clamp(-1*args.epsilon + test_wav, args.epsilon + test_wav)).detach()
            with torch.no_grad():
                test_embedding = model.extract_speaker_embedding(x_adv).squeeze(0)
                final_score = enroll_embedding.dot(test_embedding.T)
                denom = torch.norm(enroll_embedding) * torch.norm(test_embedding)
                final_score = final_score/denom
            if label == 1 and best_score >= final_score:
                adv_wav = x_adv
                best_score = torch.min(best_score, final_score)
            elif label == 0 and best_score <= final_score:
                adv_wav = x_adv
                best_score = torch.max(best_score, final_score)

        # Get Adversarial Attack wav
        adv_wav = adv_wav.cpu().detach().numpy()
        final_score = best_score.cpu().detach().numpy()

        # save attack test wav
        idx = '%08d' % idx
        adv_test_path = os.path.join(args.adv_save_dir, "wav", idx+".wav")
        wavfile.write(adv_test_path, samplerate, adv_wav.astype(np.int16))
    """
    return label, enroll_path, adv_test_path, final_score


def pgd_adversarial_attack_step(models, idx, item, **kwargs):
    args = argparse.Namespace(**kwargs)
    if args.model_num == 1:
        model_ = []
        model_.append(models)
        models = model_
    
    label = item[0]
    enroll_path = item[1]
    test_path = item[2]
    grad=[0 for _ in range(len(kwargs['weight']))]
    # load data
    samplerate, enroll_wav = load_wav(enroll_path)
    samplerate, test_wav = load_wav(test_path)
        
    # init best_score and alpha
    label = int(label)
    if label == 1:
        best_score = torch.tensor(float('inf')).cuda()
        alpha = args.alpha*(-1.0)
    else:
        best_score = torch.tensor(float('-inf')).cuda()
        alpha = args.alpha*(1.0)

    for i in range(args.restarts):
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        # PGD与BIM初始化方式不同 
        delta = torch.tensor(np.random.uniform(-40, 40, test_wav.shape), device=test_wav.device, dtype=test_wav.dtype, requires_grad=True) # 初始化方式与BIM不同
        # delta = (adv - test_wav).requires_grad_(True)
        
        for t in range(args.num_iters):
            # extract test speaker embedding
            all_k = 0
            for cid, model in enumerate(models):
                with torch.no_grad():
                    enroll_embedding = model.extract_speaker_embedding(enroll_wav).squeeze(0)
                test_embedding = model.extract_speaker_embedding(test_wav + delta).squeeze(0)
                # cosine score
                score = enroll_embedding.dot(test_embedding.T)
                denom = torch.norm(enroll_embedding) * torch.norm(test_embedding)
                score = score/denom

                # compute grad and update delta
                score.backward()  ## retain_graph=True
                k = delta.grad.detach()
                all_k += args.weight[cid] * k
                delta.grad.zero_()
                
            # apply norm bound
            # all_k is the grad
            all_k_norm = 1. / torch.linalg.norm(all_k, keepdim=True, ord=2) * all_k
            per = delta + alpha * all_k_norm
            delta.data = per
            # delta.data = per.clamp(-1*args.epsilon, args.epsilon)
            
            delta.grad.zero_()
            
            
        final_score = torch.zeros(len(args.weight))
        final_score_ = 0
        for id_, model in enumerate(models):
            with torch.no_grad():
                enroll_embedding = model.extract_speaker_embedding(enroll_wav).squeeze(0)
                test_embedding = model.extract_speaker_embedding(test_wav+delta).squeeze(0)
                final_score[id_] = enroll_embedding.dot(test_embedding.T)
                denom = torch.norm(enroll_embedding) * torch.norm(test_embedding)
                final_score[id_] = final_score[id_]/denom
        for id_, s in enumerate(final_score):
            final_score_ += args.weight[id_] * s
        if label == 1 and best_score >= final_score_:
            max_delta = delta.data
            best_score = torch.min(best_score, final_score_)
        elif label == 0 and best_score <= final_score_:
            max_delta = delta.data
            best_score = torch.max(best_score, final_score_)

        # Get Adversarial Attack wav
        adv_wav = test_wav + max_delta
        adv_wav = adv_wav.cpu().detach().numpy()
        final_score = best_score.cpu().detach().numpy()

        # save attack test wav
        idx = '%08d' % idx
        adv_test_path = os.path.join(args.adv_save_dir, "wav", idx+".wav")
        wavfile.write(adv_test_path, samplerate, adv_wav.astype(np.int16))
    return label, enroll_path, adv_test_path, final_score
