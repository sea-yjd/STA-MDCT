

from adaptive_attack.EOT import EOT
from attack.FGSM import FGSM
from attack.utils import resolve_loss
import numpy as np
from attack.utils import SEC4SR_MarginLoss
from attack.spectrum_transform import Spectrum_Trans_
import torch
from attack.utils import resolve_loss, resolve_prediction

class S2A(FGSM):
    
    def __init__(self,model, task='CSI', targeted=True, max_iter=10, epsilon=30,
                    N=20, s_rho=0.5, momentum=0.9, alpha=4, sigma=8, 
                    spectrum_transform_type='DCT', attack_transform='I',
                    batch_size=1,
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
        self.N = N
        self.s_rho = s_rho
        self.momentum = momentum
        
        self.sigma = sigma
        self.spectrum_transform_type = spectrum_transform_type
        self.attack_transform = attack_transform
        self.confidence = confidence
        if self.task in ['SV', 'OSI']:
            self.threshold = threshold
            print('Running white box attack for {} task, directly using the true threshold {}'.format(self.task, self.threshold))
        self.loss = SEC4SR_MarginLoss(targeted=self.targeted, confidence=self.confidence, task=self.task, threshold=self.threshold, clip_max=False)
        self.alpha= -1 * alpha
    def attack_batch(self,  x_batch, y_batch, lower, upper, batch_id, label):
        n_audios, _, _ = x_batch.shape
        x_batch = x_batch.reshape((-1))
        target = y_batch.detach().cpu().numpy()
        Spectrum_Trans = Spectrum_Trans_(self.spectrum_transform_type)
        grad = 0
        adv_wav = x_batch
        success = False
        for t in range(self.max_iter):
            noise = 0
            for n in range(self.N):
                gauss = (torch.randn(adv_wav.shape) * self.sigma).cuda()
                x_spectrum = Spectrum_Trans.spectrum_T(adv_wav)
                x_gauss_spectrum = x_spectrum.cuda() + Spectrum_Trans.spectrum_T(gauss).cuda()
                mask = (torch.rand_like(x_gauss_spectrum) * 2 * self.s_rho + 1 - self.s_rho).cuda()
                x_i_spectrum = Spectrum_Trans.i_spectrum_T(x_gauss_spectrum * mask).cuda()
                if x_i_spectrum.shape < adv_wav.shape:
                    x_i_spectrum = torch.nn.functional.pad(x_i_spectrum, (0, len(adv_wav)-len(x_i_spectrum)))
                elif x_i_spectrum.shape > adv_wav.shape:
                    x_i_spectrum = x_i_spectrum[:,:, :adv_wav.shape[-1]]
                x_i_spectrum.requires_grad = True

                decision, scores = self.model.make_decision(x_i_spectrum, self.enroll_embedding, self.threshold)
                loss = self.loss(scores, y_batch)
                loss.backward()
                noise += x_i_spectrum.grad.detach()
                x_i_spectrum.grad.zero_()

                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            noise = noise / self.N

            if self.attack_transform == 'MI':
                noise = noise / torch.abs(noise)
                noise = self.momentum * grad + noise
                grad = noise
            
            # if args.attack_transform == 'TI':
            #     noise = F.con
            
            adv_wav = (adv_wav + self.alpha*noise.sign()).clamp(-1*self.epsilon + x_batch, self.epsilon + x_batch)
            decisions, scores = self.model.make_decision(adv_wav, self.enroll_embedding, self.threshold)
            predict = np.array(decisions.detach().cpu())
            target = y_batch.detach().cpu().numpy()
            success = self.compare(target, predict, self.targeted)
            if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            if self.verbose:
                print('batch:{} iter:{} loss:{} predict:{} target:{}'.format(batch_id, t, loss.detach().cpu().numpy().tolist(), predict, target))
            # if success == [True]:
            #     break  
        return adv_wav, success
    
    def attack(self, x, y, label):
        return super().attack(x, y, label)