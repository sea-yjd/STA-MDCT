#!/usr/bin/env python
# encoding: utf-8

import os
from argparse import ArgumentParser
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks import ModelCheckpoint
import torchaudio
from tqdm import tqdm

import importlib
from collections import OrderedDict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from .backend.plda import PldaAnalyzer

from .utils import PreEmphasis
from .dataset_loader import Train_Dataset, Train_Sampler, Test_Dataset, Dev_Dataset, Test_classification_Dataset
from .backend import compute_eer, cosine_score, PLDA_score, save_cosine_score
from .cam.cam_give_gt import CAM, GradCAM, GradCAMpp, ScoreCAM, LayerCAM
from .cam import *
import matplotlib.pyplot as plt
from attack.spectrum_transform import Spectrum_Trans_

class Model_visualization(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # load trials and data list
        # if os.path.exists(self.hparams.trials_path):
        #     self.trials = np.loadtxt(self.hparams.trials_path, dtype=str)
        if os.path.exists(self.hparams.train_list_path):
            df = pd.read_csv(self.hparams.train_list_path)
            speaker = np.unique(df["utt_spk_int_labels"].values)
            self.hparams.num_classes = len(speaker)
            print("Number of Training Speaker classes is: {}".format(self.hparams.num_classes))

        #########################
        ### Network Structure ###
        #########################

        # 1. Acoustic Feature
        self.mel_trans = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, 
                    win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=self.hparams.n_mels)
                )
        self.instancenorm = nn.InstanceNorm1d(self.hparams.n_mels)

        # 2. Speaker_Encoder
        Speaker_Encoder = importlib.import_module('speaker_trainer.nnet.'+self.hparams.nnet_type).__getattribute__('MainModel')
        self.__S__ = Speaker_Encoder(**dict(self.hparams))

        # 3. Loss / Classifier
        if not self.hparams.evaluate:
            LossFunction = importlib.import_module('speaker_trainer.loss.'+self.hparams.loss_type).__getattribute__('LossFunction')
            self.loss = LossFunction(**dict(self.hparams))
        # self.enroll_embedding = kwargs['enroll_embedding']

    def forward(self, x, label):
        x = self.extract_speaker_embedding(x)
        x = x.reshape(-1, self.hparams.nPerSpeaker, self.hparams.embedding_dim)
        loss, acc = self.loss(x, label)
        return loss.mean(), acc

    def extract_speaker_embedding(self, data):
        # x = data.reshape(-1, data.size()[-1])
        # x = self.mel_trans(x) + 1e-6
        # x = x.log()
        # x = self.instancenorm(x)
        x = data
        if self.hparams.nnet_type == 'ECAPATDNN':
            x = self.__S__(x, aug = False)
        else:
            x = self.__S__(x)
        return x

    def training_step(self, batch, batch_idx):
        data, label = batch
        loss, acc = self(data, label)
        tqdm_dict = {"acc":acc}
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
            })
        return output

    def train_dataloader(self):
        frames_len = np.random.randint(self.hparams.min_frames, self.hparams.max_frames)
        print("Chunk size is: ", frames_len)
        print("Augment Mode: ", self.hparams.augment)
        train_dataset = Train_Dataset(self.hparams.train_list_path, self.hparams.augment, 
                musan_list_path=self.hparams.musan_list_path, rirs_list_path=self.hparams.rirs_list_path,
                max_frames=frames_len)
        train_sampler = Train_Sampler(train_dataset, self.hparams.nPerSpeaker,
                self.hparams.max_seg_per_spk, self.hparams.batch_size)
        loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                sampler=train_sampler,
                pin_memory=False,
                drop_last=False,
                )
        return loader

    def test_dataloader(self, trials, **kwargs):
        enroll_list = np.unique(trials.T[1])
        test_list = np.unique(trials.T[2])
        eval_list = np.unique(np.append(enroll_list, test_list))
        print("number of eval: ", len(eval_list))
        print("number of enroll: ", len(enroll_list))
        print("number of test: ", len(test_list))

        test_dataset = Test_Dataset(data_list=eval_list, eval_frames=self.hparams.eval_frames)
        loader = DataLoader(test_dataset, num_workers=self.hparams.num_workers, batch_size=1)
        return loader

    def test_dataloader_for_classification(self):
        print(self.hparams.test_list_path)
        test_dataset = Test_classification_Dataset(
            data_list_path=self.hparams.test_list_path, eval_frames=self.hparams.eval_frames, num_eval=0)
        #! 为了测试同一张图片的可视化，暂时不shuffle
        loader = DataLoader(
            test_dataset, num_workers=self.hparams.num_workers, batch_size=1,shuffle=False)
        return loader

    def cosine_evaluate(self, trials=None):
        if trials is None:
            trials = self.trials
        eval_loader = self.test_dataloader(trials)
        index_mapping = {}
        eval_vectors = [[] for _ in range(len(eval_loader))]
        print("extract eval speaker embedding...")
        self.__S__.eval()
        with torch.no_grad():
            for idx, (data, label) in enumerate(tqdm(eval_loader)):
                data = data.reshape(-1, data.size()[-1]).cuda()
                label = list(label)[0]
                index_mapping[label] = idx
                embedding = self.extract_speaker_embedding(data, False)
                # embedding = torch.mean(embedding, axis=0)
                embedding = embedding.cpu().detach().numpy()
                eval_vectors[idx] = embedding
        eval_vectors = np.array(eval_vectors)
        print("scoring...")
        score_file = self.hparams.trials_path[:-4] + "_victim_" + self.hparams.nnet_type + "_score.lst"
        eer, th = save_cosine_score(trials, index_mapping, eval_vectors, score_file)
        print("Cosine EER: {:.3f}%".format(eer*100))
        print("threshold: {:.3f}".format(th))
        self.log('cosine_eer', eer*100)
        self.log('threshold', th)
        return eer, th

    def evaluate(self):
        dev_dataset = Dev_Dataset(data_list_path=self.hparams.train_list_path, eval_frames=self.hparams.min_frames, num_eval=10)
        dev_loader = DataLoader(dev_dataset, num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size)

        # first we extract dev speaker embedding
        dev_vectors = []
        dev_labels = []
        print("extract dev speaker embedding...")
        for data, label in tqdm(dev_loader):
            length = len(data)
            embedding = self.extract_speaker_embedding(data.cuda())
            embedding = embedding.reshape(length, 10, -1)
            embedding = torch.mean(embedding, axis=1)
            embedding = embedding.cpu().detach().numpy()
            dev_vectors.append(embedding)
            label = label.cpu().detach().numpy()
            dev_labels.append(label)

        dev_vectors = np.vstack(dev_vectors).reshape(-1, self.hparams.embedding_dim)
        dev_labels = np.hstack(dev_labels)
        print("dev vectors shape:", dev_vectors.shape)
        print("dev labels shape:", dev_labels.shape)

        eval_loader = self.test_dataloader()
        index_mapping = {}
        eval_vectors = [[] for _ in range(len(eval_loader))]
        print("extract eval speaker embedding...")
        for idx, (data, label) in enumerate(tqdm(eval_loader)):
            data = data.permute(1, 0, 2).cuda()
            label = list(label)[0]
            index_mapping[label] = idx
            embedding = self.extract_speaker_embedding(data)
            embedding = torch.mean(embedding, axis=0)
            embedding = embedding.cpu().detach().numpy()
            eval_vectors[idx] = embedding
        eval_vectors = np.array(eval_vectors)
        print("eval_vectors shape is: ", eval_vectors.shape)

        print("scoring...")
        eer, th = cosine_score(self.trials, index_mapping, eval_vectors)
        print("Cosine EER: {:.3f}%".format(eer*100))

        # PCA
        for dim in [32, 64, 128, 150, 200, 250, 256]:
            pca = PCA(n_components=dim)
            pca.fit(dev_vectors)
            eval_vectors_trans = pca.transform(eval_vectors)
            eer, th = cosine_score(self.trials, index_mapping, eval_vectors_trans)
            print("PCA {} Cosine EER: {:.3f}%".format(dim, eer*100))

        ## LDA
        for dim in [32, 64, 128, 150, 200, 250, 256]:
            lda = LDA(n_components=dim)
            lda.fit(dev_vectors, dev_labels)
            eval_vectors_trans = lda.transform(eval_vectors)
            eer, th = cosine_score(self.trials, index_mapping, eval_vectors_trans)
            print("LDA {} Cosine EER: {:.3f}%".format(dim, eer*100))

        # PLDA
        pca = PCA(n_components=256)
        dev_vectors = pca.fit_transform(dev_vectors)
        eval_vectors = pca.fit_transform(eval_vectors)
        for dim in [32, 64, 128, 150, 200, 250, 256]:
            try:
                plda = PldaAnalyzer(n_components=dim)
                plda.fit(dev_vectors, dev_labels, num_iter=10)
                eval_vectors_trans = plda.transform(eval_vectors)
                eer, th = PLDA_score(self.trials, index_mapping, eval_vectors_trans, plda)
                print("PLDA {} EER: {:.3f}%".format(dim,eer*100)) 
            except:
                pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.lr_step_size, gamma=self.hparams.lr_gamma)
        print("init {} optimizer with learning rate {}".format("Adam", self.hparams.learning_rate))
        print("init Step lr_scheduler with step size {} and gamma {}".format(self.hparams.lr_step_size, self.hparams.lr_gamma))
        return [optimizer], [lr_scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                   optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # warm up learning_rate
        if self.trainer.global_step < 500:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
            for idx, pg in enumerate(optimizer.param_groups):
                pg['lr'] = lr_scale * self.hparams.learning_rate
        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
    
    def make_decision(self, x, enroll_embedding, threshold):
        scores = None
        test_embedding = self.extract_speaker_embedding(x).squeeze(0)
        # scores = torch.zeros((x.shape[0], len(enroll_embedding.enroll_data)), device=test_embedding.device)
        for i, enroll_embed in enumerate(enroll_embedding.enroll_data):
            # test_embedding_ = test_embedding.repeat(1, len(enroll_embedding)).repeat(-1, len(enroll_embedding), test_embedding.shape[-1])
            enroll_embed = enroll_embed['embedding'].squeeze(0)
            score = torch.mean(enroll_embed.dot(test_embedding.T))
            denorm = torch.norm(enroll_embed) * torch.norm(test_embedding)
            score = score / denorm
            score = score.unsqueeze(0)
            if scores == None:
                scores = score
            else:
                scores = torch.cat((scores, score), dim=0)
        scores = scores.reshape((1,-1))
        decisions = torch.argmax(scores, dim=1)
        max_scores = torch.max(scores, dim=1)[0]
        decisions = torch.where(max_scores > threshold, decisions,
                        torch.tensor([-1] * decisions.shape[0], dtype=torch.int64, device=decisions.device)) # -1 means reject


        return decisions, scores

    def creat_importance_map(self,speaker_classification_model,input_mel_spectrogram,true_label, pred, target, label, mel):
        target_layers_name = [self.hparams.target_layer]
        
        target_layer = find_speaker_encoder_layer(speaker_classification_model,target_layers_name[0])
        if self.hparams.visualization_type == 'CAM':
          wrapped_model = CAM(model=speaker_classification_model, loss=self.score_loss, target_layer=target_layer)
          # cam, _ = wrapped_model(input_mel_spectrogram)
          cam, _ = wrapped_model(input_mel_spectrogram,idx=label)
        if self.hparams.visualization_type == 'GradCAM':
          wrapped_model = GradCAM(model=speaker_classification_model,loss=self.score_loss, target_layer=target_layer)
          cam, _ = wrapped_model(input_mel_spectrogram,idx=label,mel=mel)
        if self.hparams.visualization_type == 'GradCAMpp':
          wrapped_model = GradCAMpp(model=speaker_classification_model,loss=self.score_loss, target_layer=target_layer)
          cam, _ = wrapped_model(input_mel_spectrogram,idx=label,mel=mel)
        if self.hparams.visualization_type == 'ScoreCAM':
          wrapped_model = ScoreCAM(model=speaker_classification_model,loss=self.score_loss, target_layer=target_layer)
          cam, _ = wrapped_model(input_mel_spectrogram,idx=label,mel=mel)
        if self.hparams.visualization_type == 'LayerCAM':
          wrapped_model = LayerCAM(model=speaker_classification_model, loss=self.score_loss, target_layer=target_layer)
          cam, _ = wrapped_model(input_mel_spectrogram,idx=label, mel=mel)

        return cam
    
    def eval_forward(self,input,label,idx,topN):
        scores = None
        # input:mel-spectrum, logits指的是embedding
        test_embedding = self.__S__(input)
        test_embedding = test_embedding.squeeze(0)
        # 这里的output指的是分类的score
        # TODO : 改成与enroll_embedding的余弦相似度得分
        for i, enroll_embed in enumerate(self.enroll_embedding.enroll_data):
            enroll_embed = enroll_embed['embedding'].squeeze(0)
            score = torch.mean(enroll_embed.dot(test_embedding.T))
            denorm = torch.norm(enroll_embed) * torch.norm(test_embedding)
            score = score / denorm
            score = score.unsqueeze(0)
            if scores == None:
                scores = score
            else:
                scores = torch.cat((scores, score), dim=0)
        scores = scores.reshape((1,-1))
        decisions = torch.argmax(scores, dim=1)
        max_scores = torch.max(scores, dim=1)[0]
        # decisions = torch.where(max_scores > 0, decisions,
        #                 torch.tensor([-1] * decisions.shape[0], dtype=torch.int64, device=decisions.device)) # -1 means reject
        return scores[0][label], label.cpu().numpy().tolist()[0], decisions, decisions, max_scores
###  ====================
        # output = self.loss(logit,label)
        # # output = self.softmax(output) # logit.size() -> [1,400]
        # output = output.cpu().detach().numpy().tolist() # max(x[0]) is the max posterior
        # label_int = label.cpu().numpy().tolist()[0]
        # score = output[0][label]
        # y_tilde = output[0].index(max(output[0]))
        # topN_score = heapq.nlargest(topN,output[0])
        # topN = heapq.nlargest(topN, range(len(output[0])), key=lambda x: output[0][x])

        # return score, label_int, y_tilde, topN, topN_score
    def score_loss(self,input,label):
        scores = None
        # input:mel-spectrum, logits指的是embedding
        # test_embedding = self.__S__(input)
        test_embedding = input
        # 这里的output指的是分类的score
        # TODO : 改成与enroll_embedding的余弦相似度得分
        for i, enroll_embed in enumerate(self.enroll_embedding.enroll_data):
            enroll_embed = enroll_embed['embedding'].squeeze(0)
            score = torch.mean(enroll_embed.dot(test_embedding.T))
            denorm = torch.norm(enroll_embed) * torch.norm(test_embedding)
            score = score / denorm
            score = score.unsqueeze(0)
            if scores == None:
                scores = score
            else:
                scores = torch.cat((scores, score), dim=0)
        return scores
    def make_decision_vis(self,input):
        scores = None
        # input:mel-spectrum, logits指的是embedding
        test_embedding = (self.__S__(input)).squeeze(0)
        # test_embedding = input
        # 这里的output指的是分类的score
        # TODO : 改成与enroll_embedding的余弦相似度得分
        for i, enroll_embed in enumerate(self.enroll_embedding.enroll_data):
            enroll_embed = enroll_embed['embedding'].squeeze(0)
            score = torch.mean(enroll_embed.dot(test_embedding.T))
            denorm = torch.norm(enroll_embed) * torch.norm(test_embedding)
            score = score / denorm
            score = score.unsqueeze(0)
            if scores == None:
                scores = score
            else:
                scores = torch.cat((scores, score), dim=0)
        decisions = torch.argmax(scores, dim=-1).tolist()
        return decisions
    
    def evaluate_visualization(self,speaker_classification_model, enroll_embedding):
        eval_loader = self.test_dataloader_for_classification()
        self.enroll_embedding = enroll_embedding
        heat_map_cow_list = [] 
        feature_map_cow_list = []
        i = 0

        # 初始化模型；初始化可视化方法
        self.__S__.eval()
        # self.loss.eval()
        # os.system('mkdir -p {}/{}/{}'.format(self.hparams.vis_result,self.hparams.visualization_type, self.hparams.use_label))
        # os.system('mkdir -p {}/input'.format(self.hparams.vis_result))
        if not os.path.exists(os.path.join(self.hparams.vis_result,self.hparams.visualization_type, self.hparams.use_label)):
            os.makedirs(os.path.join(self.hparams.vis_result,self.hparams.visualization_type, self.hparams.use_label))
        if not os.path.exists(os.path.join(self.hparams.vis_result,'CAM', self.hparams.use_label)):
            os.makedirs(os.path.join(self.hparams.vis_result,'CAM', self.hparams.use_label))
        if not os.path.exists(os.path.join(self.hparams.vis_result,'input')):
            os.makedirs(os.path.join(self.hparams.vis_result,'input'))
        if not os.path.exists(os.path.join(self.hparams.vis_result,'save_pic', self.hparams.use_label)):
            os.makedirs(os.path.join(self.hparams.vis_result,'save_pic', self.hparams.use_label))

        for idx, (data, label, name, y_target) in enumerate(tqdm(eval_loader)):
            data = data.permute(1, 0, 2).cuda()
            label, y_target = label.cuda(), y_target.cuda()
            # todo data preprocess
            x = data.reshape(-1, data.size()[-1])

            # # 对输入x进行变换
            # # =====================
            # Spectrum_Trans = Spectrum_Trans_('MDCT')
            # # for n in range(20):
            # # gauss = (torch.randn(x.shape) * 44).cuda()
            # x_spectrum = Spectrum_Trans.spectrum_T(x)
            # # x_gauss_spectrum = x_spectrum.cuda() + Spectrum_Trans.spectrum_T(gauss).cuda()
            # # mask = (torch.rand_like(x_gauss_spectrum) * 2 * 0.5 + 1 - 0.5).cuda()
            # x_i_spectrum = Spectrum_Trans.i_spectrum_T(x_spectrum).cuda()
            # x = x_i_spectrum.unsqueeze(0)
            # # ======================

            x = self.mel_trans(x) + 1e-6
            x = x.log()#! MFCC 不能提取log
            #todo test and draw mel-spectrum for input
            input_mel_spectrogram = self.instancenorm(x)   # [bs, mel_dim, T]
            mel = input_mel_spectrogram
            input_target_class_posterior, label_int, y_tilde, topN, topN_score  = self.eval_forward(input_mel_spectrogram,label,idx,self.hparams.topN)
            if self.hparams.use_label == 'use_label':
              cam_merge = self.creat_importance_map(speaker_classification_model,input_mel_spectrogram,label, y_tilde, y_target, label, mel)
            if self.hparams.use_label == 'use_pred':
              cam_merge = self.creat_importance_map(speaker_classification_model,input_mel_spectrogram,label, y_tilde, y_target, y_tilde, mel)
            if self.hparams.use_label == 'use_target':
              cam_merge = self.creat_importance_map(speaker_classification_model,input_mel_spectrogram,label, y_tilde, y_target, y_target, mel)

            if self.hparams.save_np_pic:
              heat_map_cow = cam_merge.cpu().numpy()[0,0,:,:].ravel()
              heat_map_cow_list.append(heat_map_cow.ravel())
              if len(heat_map_cow_list) == 50 or len(eval_loader) == idx+1:
                np.save('{}/{}/{}/{}_pic_{}.npy'.format(self.hparams.vis_result,self.hparams.visualization_type,self.hparams.use_label, self.hparams.visualization_type, i), heat_map_cow_list)
                heat_map_cow_list = []
                i=i+1

            if self.hparams.input_save_np_pic or len(eval_loader) == idx+1:
              feature_map_cow = input_mel_spectrogram.cpu().numpy()[0,:,:].ravel()
              feature_map_cow_list.append(feature_map_cow.ravel())
              if len(feature_map_cow_list) == 50:
                np.save('{}/input/input_pic_{}.npy'.format(self.hparams.vis_result, i), feature_map_cow_list)
                feature_map_cow_list = []
                i=i+1


    def scale(self,x,scale_gama):
        return np.tanh(scale_gama * x / x.max())

    def np2fig(self, label, input_mel_spectrogram, cam, idx, pred, target):
        # save fig class activation map
        label_int = label.cpu().numpy().tolist()[0]
        heat_map("{}_{};\nlabel:{}, pred:{}, target:{}".format(self.hparams.visualization_type, self.hparams.target_layer, int(label), int(pred), int(target)), \
        input_mel_spectrogram.cpu(), cam.cpu().numpy(), \
        save_path='{}/save_pic/{}/{}_heat_map_{}_{}_scale_{}.pdf'.format(self.hparams.vis_result,self.hparams.use_label ,idx,self.hparams.visualization_type,self.hparams.target_layer,self.hparams.scale_gama))
        plt.figure()
        plt.imshow(input_mel_spectrogram[0,:,:].cpu().numpy(), cmap='jet')
        # plt.colorbar(fraction=0.046, pad=0.04)
        # TODO:
        if self.hparams.input_save_pic == True:
            plt.savefig('{}/input/{}_input_mel_spectrogram.pdf'.format(self.hparams.vis_result,idx), bbox_inches='tight', pad_inches =0.01)

    def repocessing(self,cam, input_size, idx, label, pred, target):
        cam.reshape(input_size[0], input_size[1])
        cam_time = np.sum(cam.reshape(input_size[0], input_size[1]),axis=0)
        cam_min, cam_max = cam_time.min(), cam_time.max()
        # todo !!!!
        repocessed_arr = np.tile(((cam_time - cam_min)/(cam_max - cam_min + 1e-8)), (input_size[0],1))
        repocessed_arr = repocessed_arr.reshape(-1,1,input_size[0], input_size[1])
        print(repocessed_arr.shape)
        w_mask = np.int64((repocessed_arr-0.6) > 0)
        # threshold is set 0.6
        # heat_map("title", [], w_mask, save_path='test.png')
        heat_map("{}_{};\nlabel:{},pred:{}, target:{}".format(self.hparams.visualization_type, self.hparams.target_layer, int(label), pred, int(target)), \
        [], w_mask, \
        save_path='{}/CAM/{}/{}_heat_map_{}_{}_scale_{}.pdf'.format(self.hparams.vis_result,self.hparams.use_label, idx,self.hparams.visualization_type,self.hparams.target_layer,self.hparams.scale_gama))
        
        # import ipdb;ipdb.set_trace()
        return w_mask

    def visualization_metrics(self,speaker_classification_model, enroll_embedding):
        self.enroll_embedding = enroll_embedding
        eval_loader = self.test_dataloader_for_classification()
        insertion_h_list, deletion_h_list = [],[]
        importance_acc_list, origin_acc_list = [],[]
        i = 0

        # 初始化模型；初始化可视化方法
        self.__S__.eval()
        # self.loss.eval()
        # os.system('mkdir -p {}/{}'.format(self.hparams.vis_result,self.hparams.visualization_type))
        # os.system('mkdir -p {}/input'.format(self.hparams.vis_result))
        # os.system('mkdir -p {}/save_pic'.format(self.hparams.vis_result))

        # f1 = open('{}/{}_origin_scores_{}_{}.csv'.format(self.hparams.vis_result,self.hparams.visualization_type,self.hparams.test_type,self.hparams.threshold),'w')
        # f2 = open('{}/{}_salient_scores_{}_{}.csv'.format(self.hparams.vis_result,self.hparams.visualization_type,self.hparams.test_type,self.hparams.threshold),'w')
        

        for idx, (data, label, name, y_target) in enumerate(tqdm(eval_loader)):
            data = data.permute(1, 0, 2).cuda()
            label = label.cuda()
            # todo data preprocess
            x = data.reshape(-1, data.size()[-1])
            x = self.mel_trans(x) + 1e-6
            x = x.log()
            input_mel_spectrogram = self.instancenorm(x)
            pred = self.make_decision_vis(input_mel_spectrogram)
            cams_idx = idx % 50
            input_size = input_mel_spectrogram.squeeze(0).shape
            # save_path = {}/save_pic/{}/{}_heat_map_{}_{}_scale_{}.png'.format(self.hparams.vis_result,self.hparams.use_label ,idx,self.hparams.visualization_type,self.hparams.target_layer,self.hparams.scale_gama)
            # if os.path.exists(())
            if self.hparams.visualization_type != "_":
                if cams_idx == 0:
                    cams = np.load('{}/{}/{}/{}_pic_{}.npy'.format(self.hparams.vis_result,self.hparams.visualization_type,self.hparams.use_label, self.hparams.visualization_type, i), allow_pickle=True)
                    i = i+1
                if self.hparams.scale_gama != 0:
                    cam_np = self.scale(cams[cams_idx], self.hparams.scale_gama)
                else:
                    cam_np = cams[cams_idx]
                self.repocessing(cam_np,input_size, idx, label, pred, y_target)
                cam = torch.from_numpy(cam_np.reshape(-1,1, input_size[0], input_size[1])).cuda()

            if self.hparams.save_pic:
                self.np2fig(label, input_mel_spectrogram, cam, idx, pred, y_target)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--num_workers', type=int, default=32)
        parser.add_argument('--save_top_k', type=int, default=15)

        parser.add_argument('--loss_type', type=str, default="angleproto")
        parser.add_argument('--nnet_type', type=str, default="ResNetSE34V2")  # ['ECAPATDNN', 'ResNetSE34L', 'RawNet3', 'ResNetSE34V2']
        parser.add_argument('--pooling_type', type=str, default="SAP")

        parser.add_argument('--augment', action='store_true', default=False)
        parser.add_argument('--max_frames', type=int, default=400)
        parser.add_argument('--eval_frames', type=int, default=0)   #set max_frames=0 for evaluation. If max_frames=0, then the returned feat is a whole utterance.
        parser.add_argument('--num_eval', type=int, default=10)
        parser.add_argument('--min_frames', type=int, default=200)
        parser.add_argument('--n_mels', type=int, default=80)


        parser.add_argument('--train_list_path', type=str, default='')
        # parser.add_argument('--trials_path', type=str, default='speaker_verification/data/trials.lst')   ## 'trials.lst'
        # parser.add_argument('--trials_path', type=str, default='/home/jdyao/PycharmProjects/spot-adv-by-vocoder-main/Adversarial_examples/ECAPATDNN/S2A_data_epsilon30_it10_alpha4_srho0.5_N20_sigma8.0_DCT_MI1.0/adv_trials.lst')
        parser.add_argument('--test_list_path', type=str, default='data/test.csv')
        # parser.add_argument('--test_list_path', type=str, default='adver-audio/ECAPATDNN-CSI-Spk10_test/None/PGD/PGD-[25, 80, 6, 1.0, 0, 1]/data.csv')
        # parser.add_argument('--test_list_path', type=str, default='adver-audio/ECAPATDNN-CSI-Spk10_test/None/S2A/S2A-[10, 60, 20, 6, 44.0, 0.5, 2.0, MDCT]/data.csv')
        parser.add_argument('--musan_list_path', type=str, default='')
        parser.add_argument('--rirs_list_path', type=str, default='')
        parser.add_argument('--nPerSpeaker', type=int, default=2, help='Number of utterances per speaker per batch, only for metric learning based losses')
        parser.add_argument('--max_seg_per_spk', type=int, default=2500, help='Maximum number of utterances per speaker per epoch')

        parser.add_argument('--checkpoint_path', type=str, default='')

        parser.add_argument('--embedding_dim', type=int, default=1024)

        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--lr_step_size', type=int, default=3)
        parser.add_argument('--lr_gamma', type=float, default=0.1)

        parser.add_argument('--evaluate', action='store_true', default=True)
        parser.add_argument('--eval_interval', type=int, default=1)

        #visualization
        parser.add_argument('--visualization', action='store_true', default=True)   # TODO
        parser.add_argument('--visualization_metrics', action='store_true', default=True)   #TODO
        parser.add_argument('--visualization_type', type=str, default='LayerCAM')
        parser.add_argument('--target_layer', type=str, default='layer4')
        parser.add_argument('--vis_result', type=str, default=None)
        parser.add_argument('--topN', type=int, default=1)
        parser.add_argument('--save_pic', action='store_true', default=True)
        parser.add_argument('--save_np_pic', action='store_true', default=True)
        parser.add_argument('--input_save_np_pic', action='store_true', default=False)
        parser.add_argument('--input_save_pic', action='store_true', default=False)    # TODO
        parser.add_argument('--save_auc', action='store_true', default=False)
        parser.add_argument('--test_type', type=str, help='save_acc_multiplication,save_acc_deletion,save_acc_insertion,save_acc_random_deletion,save_acc_random_insertion,save_acc_time_deletion,save_acc_time_insertion',default='save_acc_deletion')
        parser.add_argument('--use_label', type=str, help='use_label,use_pred, use_target',default='use_label')   # TODO


        parser.add_argument('--scale_gama', type=int, default=0)
        parser.add_argument('--threshold', type=int, default=0)
        return parser

