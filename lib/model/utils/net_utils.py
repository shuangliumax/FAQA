# coding:utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable,Function
import numpy as np
import torchvision.models as models
from model.utils.config import cfg
# from model.roi_crop.functions.roi_crop import RoICropFunction
import cv2
import pdb
import random
from torch.utils.data.sampler import Sampler
from kornia.losses import ssim_loss
from torch.nn.functional import mse_loss
from torch.nn.utils import spectral_norm
from torch.nn.init import xavier_uniform_




class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

#############################################################
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)
        # print(m,"\n")

def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))

class Self_Attn(nn.Module):
    # Self attention Layer"""

    def __init__(self, in_channels, st):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels
        # print("\nchannels:",self.in_channels)
        # with torch.no_grad():
        cuda3 = torch.device('cuda:3')
        cuda2 = torch.device('cuda:2')
        self.st = st
        # self.dom = domain
        if st == 'source':
            cud = cuda2
        else:
            cud = cuda3

        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1, stride=1,
                                        padding=0).cuda(cud)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1, stride=1,
                                      padding=0).cuda(cud)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1,
                                    padding=0).cuda(cud)
        self.snconv1x1_attn = snconv2d(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=1, stride=1,
                                       padding=0).cuda(cud)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0).cuda(cud)
        self.softmax = nn.Softmax(dim=-1).cuda(cud)
        self.sigma = nn.Parameter(torch.zeros(1)).cuda(cud)

    def forward(self, x):
        cuda3 = torch.device('cuda:3')
        cuda2 = torch.device('cuda:2')

        if self.st == 'source':
            cud = cuda2
        else:
            cud = cuda3

        x = x.cuda(cud)
        # with torch.no_grad():

        self.apply(init_weights)

        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch // 8, h * w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch // 8, h * w // 4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch // 2, h * w // 4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        # attn_g = attn_g.cuda()
        attn_g = attn_g.view(-1, ch // 1, h, w)
        attn_g = self.snconv1x1_attn(attn_g)
        # Out
        out = x + self.sigma * attn_g

        del self.snconv1x1_theta,
        self.snconv1x1_phi,
        self.snconv1x1_g,
        self.snconv1x1_attn,
        self.maxpool, self.softmax,
        self.sigma

        del x, theta, phi, attn, g
        del self
        del cud
        torch.cuda.empty_cache()
        # return -torch.mul(domain, attn_g)
        return out

def get_gc_discriminator(n_classes, ndf=64):
    return nn.Sequential(
        nn.Conv2d(n_classes, ndf, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf + 8, 1, kernel_size=4, stride=2, padding=1),
        )

def self_entropy(prob, softmax):
    # prob.size() = [N, C(2)]   C: Number of categories

    if softmax:
        prob = F.softmax(prob, 1)
    prob = prob.clamp(1e-6,1)
    log_prob = torch.log(prob)
    H = - torch.sum(prob * log_prob, dim = 1) # [N]
    H_mean = H.mean()

    return H, H_mean

def global_attention(features, d):
    # d.size() =[1, 2]
    _, H = self_entropy(d, softmax=True)
    features_attention = (1 + H) * features
    return features_attention

def prob2entropy(prob):
    # convert prob prediction maps to weighted self-information maps
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) #/ np.log2(c)

def prob2entropy2(prob,st):
    # convert prob prediction maps to weighted self-information maps
    cuda0 = torch.device('cuda:0')
    cuda2 = torch.device('cuda:2')
    cuda3 = torch.device('cuda:3')
    if st=='source' :
        cud = cuda2
    else :
        cud = cuda3
    prob.cuda(cud)
    #domain.cuda(cud)
    n, c, h, w = prob.size()
    #prob = Variable(prob, requires_grad=True)
    attn1 = Self_Attn(c,st)
    out = attn1(prob)
    out1 = out.cuda(cuda0)
    #out1 = out.cuda(cuda0)
    #out1 = -torch.mul(domain_p1, out)
    #out2 = out1.cuda(cuda0)
    del out, attn1, prob, cud
    #cudaDeviceReset()
    #torch.cuda.de_init()
    #return out1
    return out1

def prob2entropy3(prob):
    # convert prob prediction maps to weighted self-information maps
    # n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) #/ np.log2(c)

def CrossEntropy(output, label):
    criteria = torch.nn.CrossEntropyLoss()
    loss = criteria(output, label)
    return loss

class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

def SoftTarget(out_s, out_t):
    loss = F.kl_div(F.log_softmax(out_s / 1, dim=1),
                    F.softmax(out_t / 1, dim=1),
                    reduction='batchmean') * 1 * 1
    return loss

class EFocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(EFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1) * 1.0)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        # print(N)
        C = inputs.size(1)
        # inputs = F.sigmoid(inputs)
        P = F.softmax(inputs)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)
        batch_loss = -alpha * torch.exp(-self.gamma * probs) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True,sigmoid=False,reduce=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1) * 1.0)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.sigmoid = sigmoid
        self.reduce = reduce
    def forward(self, inputs, targets):
        N = inputs.size(0)
        # print(N)
        C = inputs.size(1)
        if self.sigmoid:
            P = F.sigmoid(inputs)
            #F.softmax(inputs)
            if targets == 0:
                probs = 1 - P#(P * class_mask).sum(1).view(-1, 1)
                log_p = probs.log()
                batch_loss = - (torch.pow((1 - probs), self.gamma)) * log_p
            if targets == 1:
                probs = P  # (P * class_mask).sum(1).view(-1, 1)
                log_p = probs.log()
                batch_loss = - (torch.pow((1 - probs), self.gamma)) * log_p
        else:
            #inputs = F.sigmoid(inputs)
            P = F.softmax(inputs, dim = 1).clamp(1e-6,1)

            class_mask = inputs.data.new(N, C).fill_(0)
            class_mask = Variable(class_mask)
            ids = targets.view(-1, 1)
            class_mask.scatter_(1, ids.data, 1.)
            # print(class_mask)


            if inputs.is_cuda and not self.alpha.is_cuda:
                self.alpha = self.alpha.cuda()
            alpha = self.alpha[ids.data.view(-1)]

            probs = (P * class_mask).sum(1).view(-1, 1)

            log_p = probs.log()
            # print('probs size= {}'.format(probs.size()))
            # print(probs)

            batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
            # print('-----bacth_loss------')
            # print(batch_loss)

        if not self.reduce:
            return batch_loss
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
class FocalPseudo(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True,threshold=0.8):
        super(FocalPseudo, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1)*1.0)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.threshold = threshold

    def forward(self, inputs):
        N = inputs.size(0)
        C = inputs.size(1)
        inputs = inputs[0,:,:]
        #print(inputs)
        #pdb.set_trace()
        inputs,ind = torch.max(inputs,1)
        ones = torch.ones(inputs.size()).cuda()
        value = torch.where(inputs>self.threshold,inputs,ones)
        #
        #pdb.set_trace()
        #ind
        #print(value)
        try:
            ind = value.ne(1)
            indexes = torch.nonzero(ind)
            #value2 = inputs[indexes]
            inputs = inputs[indexes]
            log_p = inputs.log()
            # print('probs size= {}'.format(probs.size()))
            # print(probs)
            if not self.gamma == 0:
                batch_loss = - (torch.pow((1 - inputs), self.gamma)) * log_p
            else:
                batch_loss = - log_p
        except:
            #inputs = inputs#[indexes]
            log_p = value.log()
            # print('probs size= {}'.format(probs.size()))
            # print(probs)
            if not self.gamma == 0:
                batch_loss = - (torch.pow((1 - inputs), self.gamma)) * log_p
            else:
                batch_loss = - log_p
        # print('-----bacth_loss------')
        # print(batch_loss)
        #batch_loss = batch_loss #* weight
        if self.size_average:
            try:
                loss = batch_loss.mean() #+ 0.1*balance
            except:
                pdb.set_trace()
        else:
            loss = batch_loss.sum()
        return loss

class GradReverseFunction(torch.autograd.Function):
    @staticmethod
    # def __init__(self, lambd):
    #     self.lambd = lambd
    #
    # def forward(self, x):
    #     return x.view_as(x)
    #
    # def backward(self, grad_output):
    #     #pdb.set_trace()
    #     return (grad_output * -self.lambd)
    def forward(ctx, x, lambd):
        ctx.save_for_backward(x)
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.neg() * ctx.lambd
        return grad_input, None

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        #pdb.set_trace()
        return (grad_output * -self.lambd)

def grad_reverse(x, lambd=1.0):
    return GradReverseFunction.apply(x, lambd)
def SP(fm_s, fm_t):
    fm_s = fm_s.view(fm_s.size(0), -1)
    G_s = torch.mm(fm_s, fm_s.t())
    norm_G_s = F.normalize(G_s, p=2, dim=1)

    fm_t = fm_t.view(fm_t.size(0), -1)
    G_t = torch.mm(fm_t, fm_t.t())
    norm_G_t = F.normalize(G_t, p=2, dim=1)

    loss = F.mse_loss(norm_G_s, norm_G_t)

    return loss
def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())

def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)

def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = torch.sqrt(totalnorm).item()

    norm = (clip_norm / max(totalnorm, clip_norm))
    #print(norm)
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)

def vis_detections(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:

            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 255, 255), 2)
            # cv2.putText(im, '%s:%.2f' % (class_name, score), (bbox[0] - 5, bbox[1] + (-5)), cv2.FONT_HERSHEY_DUPLEX,
            #             0.3, (0, 255, 255), thickness=1)
            # cv2.putText(im, '%s:%.2f' % (class_name, score), (bbox[0] - 15, bbox[1] + (-5)), cv2.FONT_HERSHEY_PLAIN,
            #             0.65, (0, 255, 255), thickness=1)
        # if score > thresh:
        #     cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
        #     cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
        #                 1.0, (0, 0, 255), thickness=1)
    return im


def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']

def calc_supp(iter,iter_total=80000):
    p = float(iter) / iter_total
    #print(math.exp(-10*p))
    return 2 / (1 + math.exp(-10*p)) - 1
# def adjust_learning_rate(optimizer, decay=0.1,lr_init = 0.001):
#     """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = decay * lr_init#param_group['lr']


def save_checkpoint(state, filename):
    torch.save(state, filename)

def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
      loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box

def _crop_pool_layer(bottom, rois, max_pool=True):
    # code modified from 
    # https://github.com/ruotianluo/pytorch-faster-rcnn
    # implement it using stn
    # box to affine
    # input (x1,y1,x2,y2)
    """
    [  x2-x1             x1 + x2 - W + 1  ]
    [  -----      0      ---------------  ]
    [  W - 1                  W - 1       ]
    [                                     ]
    [           y2-y1    y1 + y2 - H + 1  ]
    [    0      -----    ---------------  ]
    [           H - 1         H - 1      ]
    """
    rois = rois.detach()
    batch_size = bottom.size(0)
    D = bottom.size(1)
    H = bottom.size(2)
    W = bottom.size(3)
    roi_per_batch = rois.size(0) / batch_size
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = bottom.size(2)
    width = bottom.size(3)

    # affine theta
    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([\
      (x2 - x1) / (width - 1),
      zero,
      (x1 + x2 - width + 1) / (width - 1),
      zero,
      (y2 - y1) / (height - 1),
      (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    if max_pool:
      pre_pool_size = cfg.POOLING_SIZE * 2
      grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, pre_pool_size, pre_pool_size)))
      bottom = bottom.view(1, batch_size, D, H, W).contiguous().expand(roi_per_batch, batch_size, D, H, W)\
                                                                .contiguous().view(-1, D, H, W)
      crops = F.grid_sample(bottom, grid)
      crops = F.max_pool2d(crops, 2, 2)
    else:
      grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, cfg.POOLING_SIZE, cfg.POOLING_SIZE)))
      bottom = bottom.view(1, batch_size, D, H, W).contiguous().expand(roi_per_batch, batch_size, D, H, W)\
                                                                .contiguous().view(-1, D, H, W)
      crops = F.grid_sample(bottom, grid)
    
    return crops, grid

def _affine_grid_gen(rois, input_size, grid_size):

    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([\
      (x2 - x1) / (width - 1),
      zero,
      (x1 + x2 - width + 1) / (width - 1),
      zero,
      (y2 - y1) / (height - 1),
      (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, grid_size, grid_size)))

    return grid

def _affine_theta(rois, input_size):

    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())

    # theta = torch.cat([\
    #   (x2 - x1) / (width - 1),
    #   zero,
    #   (x1 + x2 - width + 1) / (width - 1),
    #   zero,
    #   (y2 - y1) / (height - 1),
    #   (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    theta = torch.cat([\
      (y2 - y1) / (height - 1),
      zero,
      (y1 + y2 - height + 1) / (height - 1),
      zero,
      (x2 - x1) / (width - 1),
      (x1 + x2 - width + 1) / (width - 1)], 1).view(-1, 2, 3)

    return theta

class ContrastiveMLP(nn.Module):
    def __init__(self, in_channel=2048, out_channel=128):
        super(ContrastiveMLP, self).__init__()
        self.fc1 = nn.Linear(2048, 512)  #
        self.fc2 = nn.Linear(512, 256)  #
        self.fc3 = nn.Linear(256, 128)  #
        self.relu = nn.ReLU()

    def forward(self, fg_feats, bg_feats):
        fg_feats = self.relu(self.fc1(fg_feats))  #
        fg_feats = self.relu(self.fc2(fg_feats))  #
        fg_feats = self.fc3(fg_feats)  #

        bg_feats = self.relu(self.fc1(bg_feats))  #
        bg_feats = self.relu(self.fc2(bg_feats))  #
        bg_feats = self.fc3(bg_feats)  #

        return fg_feats, bg_feats































