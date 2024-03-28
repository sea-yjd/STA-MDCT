
from attack.FGSM import FGSM
from attack.utils import SEC4SR_MarginLoss
from attack.utils import resolve_loss
import torch
import math
import numpy as np

class AoA(FGSM):

    def __init__(self,model, task='CSI', targeted=True, max_iter=10, epsilon=30,
                    alpha=4, batch_size=1,
                    loss='Entropy', EOT_size=1, EOT_batch_size=1,threshold=0,
                    enroll_embedding = None, confidence=0.0,
                    verbose=1):
        self.model = model # remember to call model.eval()
        self.task = task
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.loss_name = loss
        self.targeted = targeted
        self.batch_size = batch_size
        EOT_size = max(1, EOT_size)
        EOT_batch_size = max(1, EOT_batch_size)
        assert EOT_size % EOT_batch_size == 0, 'EOT size should be divisible by EOT batch size'
        self.EOT_size = EOT_size
        self.EOT_batch_size = EOT_batch_size
        self.verbose = verbose
        self.threshold = threshold
        self.enroll_embedding = enroll_embedding
        self.confidence = confidence
        self.threshold = 0
        self.alpha = alpha
        if self.task in ['SV', 'OSI']:
            self.threshold = self.model.threshold
            print('Running white box attack for {} task, directly using the true threshold {}'.format(self.task, self.threshold))
        self.loss, self.grad_sign = resolve_loss(loss_name=self.loss_name, targeted=self.targeted, confidence=self.confidence,
                                    task=self.task, threshold=threshold, clip_max=False)
        # self.loss = SEC4SR_MarginLoss(targeted=self.targeted, confidence=self.confidence, task=self.task, threshold=self.threshold, clip_max=True)
    
    def attack_batch(self, x_batch, y_batch, lower, upper, batch_id, label):
        n_audios, _, _ = x_batch.shape
        x_batch = x_batch.reshape((-1))
        target = y_batch.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        grad = 0
        adv_wav = x_batch
        success = False
        score_best = None
        adv_wav_best = x_batch
        for t in range(self.max_iter):
            # x_batch 首先要进行mel变换, 最后一个label即是cam变换的目标
            # CAM的维度是多少？
            # adv_wav = adv_wav.permute(1, 0, 2).cuda()
            # label, y_batch = label.cuda(), y_batch.cuda()
            # todo data preprocess
            x = adv_wav.reshape(-1, adv_wav.size()[-1])
            x.requires_grad = True


            input_mel_spectrogram = self.model.instancenorm((self.model.mel_trans(x) + 1e-6).log())

            cam_label = self.model.creat_importance_map(self.model.__S__.eval(), x, label, None, y_batch, label, input_mel_spectrogram)
            cam_tgt  = self.model.creat_importance_map(self.model.__S__.eval(), x, label, None, y_batch, y_batch, input_mel_spectrogram)
            # TODO: loss的制定可以有多种方式 
            loss_cam = math.log(torch.norm(cam_label, p=1)) - math.log(torch.norm(cam_tgt, p=1))

            decision, scores = self.model.make_decision(x, self.enroll_embedding, self.threshold)
            loss_dis = self.loss(scores, y_batch)
            loss = loss_cam - loss_dis
            loss.backward()
            noise = x.grad.detach()
            
            adv_wav = (adv_wav + self.alpha*noise.sign()).clamp(-1*self.epsilon + x_batch, self.epsilon + x_batch)
            decisions, scores = self.model.make_decision(adv_wav, self.enroll_embedding, self.threshold)
            if decisions == y_batch:
                if not score_best:
                    score_best = scores[:, int(y_batch)]
                    adv_wav_best = adv_wav
                elif score_best < scores[:, int(y_batch)]:
                    score_best = scores[:, int(y_batch)]
                    adv_wav_best = adv_wav
            predict = np.array(decisions.detach().cpu())
            target = y_batch.detach().cpu().numpy()
            success = self.compare(target, predict, self.targeted)
            if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            if self.verbose:
                print('batch:{} iter:{} loss:{} predict:{} target:{}'.format(batch_id, t, loss.detach().cpu().numpy().tolist(), predict, target))
            # if success == [True]:
            #     break  
        if score_best:
            return adv_wav_best, success
        else:
            return adv_wav, success
    

    def attack(self, x, y, label):
        return super().attack(x, y, label)


