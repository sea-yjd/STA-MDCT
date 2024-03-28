from pickle import FALSE
import torch
import torch.nn.functional as F
import sys
from statistics import mode, mean
"""Ref:https://github.com/yiskw713/SmoothGradCAMplusplus"""

"""
为了验证aamsoftmax 是由于s还是m造成的影响
这里重新写了cam_give_gt.py
目的是,原始cam不需要给定 label。但是如果要使用aamsoftmax或者amsoftmax
都需要给出对应的类别
"""
class SaveValues():
    def __init__(self, m):
        # register a hook to save values of activations and gradients
        self.activations = None
        self.gradients = None
        self.forward_hook = m.register_forward_hook(self.hook_fn_act)
        self.backward_hook = m.register_full_backward_hook(self.hook_fn_grad)

    def hook_fn_act(self, module, input, output):
        self.activations = output

    def hook_fn_grad(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


class CAM(object):
    """ Class Activation Mapping """

    def __init__(self, model, loss, target_layer):
        """
        Args:
            model: a base model to get CAM which have global pooling and fully connected layer.
            target_layer: conv_layer before Global Average Pooling
        """

        self.model = model
        self.loss = loss
        self.target_layer = target_layer

        # save values of activations and gradients in target_layer
        self.values = SaveValues(self.target_layer)

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
        Return:
            heatmap: class activation mappings of the predicted class
        """

        # object classification
        embedding = self.model(x)
        embedding = embedding.unsqueeze(0)
        score = self.loss(embedding)
        # print(" ids {}\t probability {}, loss {}".format(idx, score[0,idx], loss))

        # prob = F.softmax(score, dim=1)
        
        # if idx is None:
        #     prob, idx = torch.max(prob, dim=1)
        #     idx = idx.item()
        #     prob = prob.item()
        #     print("predicted class ids {}\t probability {}".format(idx, prob))
        # cam can be calculated from the weights of linear layer and activations
        # weight_fc = list(
        #     self.model._modules.get('fc').parameters())[0].to('cpu').data

        cam = self.getCAM(x, self.values, 0, idx)

        return cam, idx

    def __call__(self, x,idx):
        return self.forward(x,idx)

    def getCAM(self,x, values, weight_fc, idx):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W)
        weight_fc: the weight of fully connected layer.  shape => (num_classes, C)
        idx: predicted class id
        cam: class activation map.  shape => (1, num_classes, H, W)
        '''

        # cam = F.conv2d(values.activations, weight=weight_fc[:, :, None, None])
        #todo replace average
        c, h, w = x.size()  # c:1,h:mel-scale,w:time

        b, k, u, v = values.activations.shape
        cam = torch.zeros((1, 1, h, w))
        if torch.cuda.is_available():
          cam = cam.cuda()
        for i in range(k):
            # upsampling
            saliency_map = torch.unsqueeze(values.activations[:, i, :, :], 1)
            saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
            if saliency_map.max() == saliency_map.min():
              continue
            cam += saliency_map
        _, _, h, w = cam.shape

        # class activation mapping only for the predicted class
        # cam is normalized with min-max.
        cam = cam[:, idx, :, :]
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        cam = cam.view(1, 1, h, w)

        return cam.data


class GradCAM(CAM):
    """ Grad CAM """

    def __init__(self, model, loss, target_layer):
        super().__init__(model, loss, target_layer)
        """
        Args:
            model: a base model to get CAM, which need not have global pooling and fully connected layer.
            target_layer: conv_layer you want to visualize
        """

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
            idx: ground truth index => (1, C)
        Return:
            heatmap: class activation mappings of the predicted class
        """

        # anomaly detection
        embedding = self.model(x)
        embedding = embedding.squeeze(0)
        score = self.loss(embedding,idx)
        prob = F.softmax(score, dim=-1)
        if idx is None:
            prob, idx = torch.max(prob, dim=1)
            idx = idx.item()
            prob = prob.item()
            print("predicted class ids {}\t probability {}".format(idx, prob))

        # caluculate cam of the predicted class
        cam = self.getGradCAM(x, self.values, score, idx)

        return cam, idx

    def __call__(self, x,idx):
        return self.forward(x,idx)

    def getGradCAM(self, x, values, score, idx):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W)
        score: the output of the model before softmax
        idx: predicted class id
        cam: class activation map.  shape=> (1, 1, H, W)
        '''
        _, h, w = x.size()  # c:1,h:mel-scale,w:time

        self.model.zero_grad()
        self.loss.zero_grad()
        score[0, idx].backward(retain_graph=True)

        activations = values.activations
        gradients = values.gradients
        with torch.no_grad():
          n, c, _, _ = gradients.shape
          alpha = gradients.view(n, c, -1).mean(2)
          alpha = alpha.view(n, c, 1, 1)

          cam = torch.zeros((1, 1, h, w)).cuda()

          for i in range(c):
              # upsampling
              saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
              saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
              if saliency_map.max() == saliency_map.min():
                continue
              cam += saliency_map * alpha[:,i,:,:]
          # _, _, h, w = cam.shape
          # shape => (1, 1, H', W')
          # cam = (alpha * activations).sum(dim=1, keepdim=True)
          
          cam = F.relu(cam)
          cam -= torch.min(cam)
          cam /= torch.max(cam)

        return cam.data


class GradCAMpp(CAM):
    """ Grad CAM plus plus """

    def __init__(self, model, loss, target_layer):
        super().__init__(model, loss, target_layer)
        """
        Args:
            model: a base model
            target_layer: conv_layer you want to visualize
        """

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
        Return:
            heatmap: class activation mappings of predicted classes
        """

        # object classification
        embedding = self.model(x)
        embedding = embedding.squeeze(0)
        score = self.loss(embedding,idx)

        prob = F.softmax(score, dim=-1)

        if idx is None:
            prob, idx = torch.max(prob, dim=1)
            idx = idx.item()
            prob = prob.item()
            print("predicted class ids {}\t probability {}".format(idx, prob))

        # caluculate cam of the predicted class
        cam = self.getGradCAMpp(x, self.values, score, idx)

        return cam, idx

    def __call__(self, x,idx):
        return self.forward(x,idx)

    def getGradCAMpp(self, x, values, score, idx):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W)
        score: the output of the model before softmax. shape => (1, n_classes)
        idx: predicted class id
        cam: class activation map.  shape=> (1, 1, H, W)
        '''

        _, h, w = x.size()  # c:1,h:mel-scale,w:time

        self.model.zero_grad()
        # self.loss.zero_grad()

        score[idx].backward(retain_graph=True)

        activations = values.activations
        gradients = values.gradients
        n, c, _, _ = gradients.shape
        
        with torch.no_grad():
          # calculate alpha
          numerator = gradients.pow(2)
          denominator = 2 * gradients.pow(2)
          ag = activations * gradients.pow(3)
          denominator += ag.view(n, c, -1).sum(-1, keepdim=True).view(n, c, 1, 1)
          denominator = torch.where(
              denominator != 0.0, denominator, torch.ones_like(denominator))
          alpha = numerator / (denominator + 1e-7)

          relu_grad = F.relu(score[idx].exp() * gradients)
          weights = (alpha * relu_grad).reshape(n, c, -1).sum(-1).reshape(n, c, 1, 1)

          # shape => (1, 1, H', W')
          # cam = (weights * activations).sum(1, keepdim=True)
          cam = torch.zeros((1, 1, h, w)).cuda()

          for i in range(c):
              # upsampling
              saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
              saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
              if saliency_map.max() == saliency_map.min():
                continue
              cam += saliency_map * weights[:,i,:,:]

          cam = F.relu(cam)
          cam -= torch.min(cam)
          cam /= torch.max(cam)

        return cam.data


class ScoreCAM(CAM):
    """ Grad CAM """

    def __init__(self, model, loss, target_layer):
        super().__init__(model, loss, target_layer)
        """
        Args:
            model: a base model to get CAM, which need not have global pooling and fully connected layer.
            target_layer: conv_layer you want to visualize
        """

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
            idx: ground truth index => (1, C)
        Return:
            heatmap: class activation mappings of the predicted class
        """

        # anomaly detection
        embedding = self.model(x).cuda()
        embedding = embedding.squeeze(0)
        score = self.loss(embedding,idx)
        prob = F.softmax(score, dim=1)

        if idx is None:
            prob, idx = torch.max(prob, dim=-1)
            idx = idx.item()
            prob = prob.item()
            print("predicted class ids {}\t probability {}".format(idx, prob))

        # caluculate cam of the predicted class
        cam = self.getScoreCAM(x, self.values, score, idx)

        return cam, idx
    def __call__(self, x,idx):
        return self.forward(x,idx)
    def getScoreCAM(self, x, values, score, idx):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W)
        score: the output of the model before softmax
        idx: predicted class id
        cam: class activation map.  shape=> (1, 1, H, W)
        '''
        _, h, w = x.size()  # c:1,h:mel-scale,w:time

        self.model.zero_grad()
        self.loss.zero_grad()
        activations = values.activations
        b, k, u, v = activations.size()
        cam = torch.zeros((1, 1, h, w))
        if torch.cuda.is_available():
          activations = activations.cuda()
          cam = cam.cuda()
        with torch.no_grad():
          for i in range(k):

              # upsampling
              saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
              
              saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
              
              if saliency_map.max() == saliency_map.min():
                continue

              # normalize to 0-1
              norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

              # how much increase if keeping the highlighted region
              # predication on masked input
              norm_saliency_map = torch.squeeze(norm_saliency_map,1)
              embedding = self.model(x * norm_saliency_map).cuda()
              embedding = embedding.unsqueeze(0)
              score = self.loss(embedding,idx)
              # output = self.model_arch(input * norm_saliency_map)
             
              score = F.softmax(score,1)
              score = score[0][idx]

              cam +=  score * saliency_map
             
        cam = F.relu(cam)
        score_saliency_map_min, score_saliency_map_max = cam.min(), cam.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None

        cam = (cam - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data

        return cam

class LayerCAM(CAM):
    """ Grad CAM """

    def __init__(self, model, loss, target_layer):
        super().__init__(model, loss, target_layer)
        """
        Args:
            model: a base model to get CAM, which need not have global pooling and fully connected layer.
            target_layer: conv_layer you want to visualize
        """
    def forward(self, x, idx, mel):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
            idx: ground truth index => (1, C)
        Return:
            heatmap: class activation mappings of the predicted class
        """
        # anomaly detection
        # x = x.reshape(-1, x.size()[-1])   # is_mel=False   ecapatdnn
        # x = x.unsqueeze(0)   # is_mel=True       resnet
        embedding = self.model(x).cuda()   # TODO: aims to ecapatdnn model
        embedding = embedding.squeeze(0)
        score = self.loss(embedding,idx)
        prob = F.softmax(score, dim=-1)

        if idx is None:
            prob, idx = torch.max(prob, dim=1)
            idx = idx.item()
            prob = prob.item()
            print("predicted class ids {}\t probability {}".format(idx, prob))

        # caluculate cam of the predicted class
        cam = self.getLayerCAM(x, self.values, score, idx, mel)  

        return cam, idx

    def __call__(self, x, idx, mel):
        return self.forward(x, idx, mel)

    def getLayerCAM(self, x, values, score, idx ,mel):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W)
        score: the output of the model before softmax
        idx: predicted class id
        cam: class activation map.  shape=> (1, 1, H, W)
        '''
        # if len(x.size()) < 3:
        #     x = x.unsqueeze(0)
        # if len(x.size()) > 3:
        #     x = x.squeeze(0)

        _, h, w = mel.size()  # c:1,h:mel-scale,w:time   [1, 80, 497]

        one_hot_output = torch.FloatTensor(1, score.size()[-1]).zero_()
        one_hot_output[0][idx] = 1
        one_hot_output = one_hot_output.cuda(non_blocking=True)
        # Zero grads
        self.model.zero_grad()
        # self.loss.zero_grad()
        score = score.reshape((1, -1))
        score.backward(gradient=one_hot_output, retain_graph=True)
        
        # score[0, idx].backward(retain_graph=True)
        activations = values.activations.clone().detach()
        gradients = values.gradients.clone().detach()

        # activations = self.activations['value'].clone().detach()
        # gradients = self.gradients['value'].clone().detach()
        
        # ResNet
        b, k, u, v = activations.size()   

        with torch.no_grad():
            activation_maps = activations * F.relu(gradients)
            cam = torch.sum(activation_maps, dim=1).unsqueeze(0)      
            cam = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)      
            cam_min, cam_max = cam.min(), cam.max()
            norm_cam = (cam - cam_min).div(cam_max - cam_min + 1e-8).data
        return norm_cam   
