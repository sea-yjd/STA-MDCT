

from adaptive_attack.EOT import EOT
from attack.FGSM import FGSM
from attack.utils import resolve_loss
import numpy as np
from attack.utils import SEC4SR_MarginLoss
from attack.spectrum_transform import Spectrum_Trans_
import torch
from attack.utils import resolve_loss, resolve_prediction

class ACG(FGSM):
    
    def __init__(self,model, task='CSI', targeted=True, max_iter=20, epsilon=20,
                        rho=0.75, W_set=[5,10,15], alpha=2, confidence=0., loss='Margin', 
                        eta =2.0, EOT_size=1, EOT_batch_size=1,threshold=0,
                       enroll_embedding = '',batch_size=1,  verbose=1):

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
        self.rho = rho
        self.W_set = W_set
        self.alpha = alpha
        self.eta = eta
        self.confidence = confidence
        if self.task in ['SV', 'OSI']:
            self.threshold = threshold
            print('Running white box attack for {} task, directly using the true threshold {}'.format(self.task, self.threshold))
        self.loss = SEC4SR_MarginLoss(targeted=self.targeted, confidence=self.confidence, task=self.task, threshold=self.threshold, clip_max=False)
        self.alpha= -1 * alpha
    def attack_batch(self,  x_batch, y_batch, lower, upper, batch_id, label):
        eta_list = []
        loss_list = []
        nabla_list = []
        s_list = []
        x_list = []
        beta_list = []
        lower = lower.reshape(1, -1).squeeze()
        upper = upper.reshape(1, -1).squeeze()
        x_batch = x_batch.reshape(1, -1).squeeze()
        x_pre = x_adv = x_batch
        x_list.append(x_batch)
        x_batch.requires_grad = True
        target = y_batch.detach().cpu().numpy()
        # x_pre.requires_grad = True
        # x_adv.requires_grad = True
        # compute gradient
        beta_list.append(0)
        eta_list.append(self.eta)
        decision, score = self.model.make_decision(x_batch, self.enroll_embedding, self.threshold)
        loss = self.loss(score, y_batch)
        loss.backward()
        s_pre = x_batch.grad.detach()
        
        # auto conjugate gradient method
        loss_list.append(loss)
        nabla_list.append(s_pre)
        s_list.append(s_pre)

        for k in range(self.max_iter):
            x_new = (x_list[k] + self.alpha * eta_list[k] * torch.sign(s_list[k])).detach()
            x_new = torch.min(torch.max(x_new, lower), upper)
            x_list.append(x_new)
            x_new.requires_grad = True
            decision_new, score_new = self.model.make_decision(x_new, self.enroll_embedding, self.threshold)
            decision_adv, score_adv = self.model.make_decision(x_adv, self.enroll_embedding, self.threshold)
            loss_adv = self.loss(score_adv, y_batch)
            loss_new = self.loss(score_new, y_batch)
            loss_new.backward()
            loss_list.append(loss_new)

            y_list = []
            if loss_new <= loss_adv:
                x_adv = torch.min(torch.max(x_list[k+1], lower), upper)
                x_pre = x_list[k]
                s_pre = s_list[k]
            eta_list.append([])
            eta_list[k+1] = eta_list[k]
            nabla_list.append(x_new.grad.detach())

            with torch.no_grad():
                if k in self.W_set[1:]:
                    if self.judgement(k, x_list, eta_list, loss_list):
                        eta_list[k+1] = (eta_list[k] / 2)
                        x_list[k+1] = x_adv
                        x_list[k] = x_pre
                        s_list[k] = s_pre
                y_list.append(nabla_list[-1] - nabla_list[-2])
                beta_list.append(torch.dot(-nabla_list[-1], y_list[-1]) / (torch.dot(s_list[-1], y_list[-1])))
                s_list.append(nabla_list[-1] + beta_list[-1] * s_list[-1])
                decisions, scores = self.model.make_decision(x_adv, self.enroll_embedding, self.threshold)
                predict = np.array(decisions.detach().cpu())
                success = self.compare(target, predict, self.targeted)
            if self.verbose:
                print('batch:{} iter:{} loss:{} predict:{} target:{}'.format(batch_id, k, loss_list[k].detach().cpu().numpy().tolist(), predict, target))
        # Get Adversarial Attack wav
        adv_wav = x_adv
        adv_wav = adv_wav.cpu().detach().numpy()
        return adv_wav, success
    
    def attack(self, x, y, label):
        return super().attack(x, y, label)

    def judgement(self, k, x, eta_list, loss_list):
        flag = False
        count = 0
        j = self.W_set.index(k)
        for i in range(self.W_set[j-1], k-1):
            # test_embedding_new = self.model.extract_speaker_embedding(x[i+1]).squeeze(0)
            # score_new = enroll_embedding.dot(test_embedding_new.T)
            # denom_new = torch.norm(enroll_embedding) * torch.norm(test_embedding_new)
            # score_new = score_new / denom_new

            # test_embedding_old = self.model.extract_speaker_embedding(x[i]).squeeze(0)
            # score_old = enroll_embedding.dot(test_embedding_old.T)
            # denom_old = torch.norm(enroll_embedding) * torch.norm(test_embedding_old)
            # score_old = score_old / denom_old

            if loss_list[i+1] < loss_list[i]:
                count += 1
        if count < self.rho * (k - self.W_set[j-1]):
            flag = True
        if eta_list[k] == eta_list[self.W_set[j-1]] and loss_list[k] == loss_list[self.W_set[j-1]]:
            flag = True
        return flag
