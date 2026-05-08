import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from torchvision.utils import save_image

import torch.fft as fft
from model.roi_layers import ROIAlign, ROIPool
# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade_aug import _ProposalTargetLayer_aug
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer

import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.raw_strength = nn.Parameter(torch.tensor(1.0))
        self.w_alpha = 1
        self.consistency_a = 0.1
        self.consistency_b = 0.1
        self.contrastive_c = 0.1

        self.print_counter = 0
        self.print_interval = 500

        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_proposal_target_aug = _ProposalTargetLayer_aug(self.n_classes)

        #####################################################################
        feat_dim = 2048
        projection_dim = 128
        self.contrastive_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, projection_dim)
        )
        #####################################################################
        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0) ###4，3，800，800
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        #########################################################################
        #Fourier-based Perturbation Augmentation (FPA)
        #########################################################################
        #im_data.shape torch.Size([4, 3, 800, 800])
        if self.training:

            # print('im_data', im_data)
            B, C, H, W = im_data.shape
            #
            fft = torch.fft.fftn(im_data, dim=(2, 3))  #
            fft_shifted = torch.fft.fftshift(fft, dim=(2, 3))  #
            #
            amplitude = torch.abs(fft_shifted)
            phase = torch.angle(fft_shifted)
            ##
            center_x = W // 2
            center_y = H // 2
            min_radius = 1  #
            max_radius = H // 32  #

            if max_radius < min_radius:
                max_radius = min_radius
            radius = torch.randint(min_radius, max_radius + 1, (1,), device=im_data.device).item()
            #
            y = torch.arange(H, device=im_data.device)
            x = torch.arange(W, device=im_data.device)

            grid_y, grid_x = torch.meshgrid(y, x)

            dist_sq = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2
            #
            mask = (dist_sq <= radius ** 2).float()
            mask = mask.unsqueeze(0).unsqueeze(0)

            effective_strength = F.softplus(self.raw_strength)
            perturbation = 1 + effective_strength * torch.randn(B, C, H, W, device=im_data.device)            # 对掩码区域施加扰动（乘性方式）
            perturbed_amplitude = amplitude * (1 + mask * (perturbation - 1))
            #################################################################
            self.print_counter += 1
            if self.print_counter % self.print_interval == 0:
                print(f"[Batch {self.print_counter}] "
                      f"[effective_strength {effective_strength}] "
                      f"Center: ({center_x}, {center_y}) | Radius: {radius}")

            #
            fft_shifted_perturbed = perturbed_amplitude * torch.exp(1j * phase)
            #
            fft_perturbed = torch.fft.ifftshift(fft_shifted_perturbed, dim=(2, 3))
            im_data_augmented = torch.fft.ifftn(fft_perturbed, dim=(2, 3)).float()
        base_feat = self.RCNN_base(im_data)  ###4，1024，50，50
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
        #rois[4,2000,5]
        consistency_loss_cls = 0
        consistency_loss_loc = 0
        contrastive_loss = 0
        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            ###############################################################
            base_feat_aug = self.RCNN_base(im_data_augmented)
            roi_data = self.RCNN_proposal_target_aug(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws, max_ious = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
            max_ious = Variable(max_ious.view(-1))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        if self.training:
            if cfg.POOLING_MODE == 'align':
                pooled_feat_aug = self.RCNN_roi_align(base_feat_aug, rois.view(-1, 5))
            elif cfg.POOLING_MODE == 'pool':
                pooled_feat_aug = self.RCNN_roi_pool(base_feat_aug, rois.view(-1, 5))
        # feed pooled features to top model
        ######256，2048
        pooled_feat = self._head_to_tail(pooled_feat)
        if self.training:
            pooled_feat_aug = self._head_to_tail(pooled_feat_aug)
        # compute bbox offset

        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training:
            bbox_pred_aug = self.RCNN_bbox_pred(pooled_feat_aug)

        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

            bbox_pred_view_aug = bbox_pred_aug.view(bbox_pred_aug.size(0), int(bbox_pred_aug.size(1) / 4), 4)
            bbox_pred_select_aug = torch.gather(bbox_pred_view_aug, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred_aug = bbox_pred_select_aug.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        if self.training:
            cls_score_aug = self.RCNN_cls_score(pooled_feat_aug)
            cls_prob_aug = F.softmax(cls_score_aug, 1)
#======================================================================================================
#                                Quality-Aware Invariance Learning (QAIL)
#======================================================================================================
        if self.training:
            ##batch_size1=512,num_classes=2
            batch_size1, num_classes = cls_prob.shape
            ##256，2
            class_mask = torch.zeros_like(cls_prob)
            class_mask[torch.arange(batch_size1), rois_label] = 1.0
            p_cls = (cls_prob * class_mask).sum(dim=1)
            fg_mask = rois_label > 0
            h = (p_cls ** 0.5) * (max_ious ** (1 - 0.5))
            h = torch.clamp(h, min=1e-5, max=1.0)
###########################################################
            h_fore = h[fg_mask]
            w = torch.exp(self.w_alpha * h_fore) - 1
            # print('w', w)
            ######512
            w = w.detach()
# ======================================================================================================
            kl_loss = F.kl_div(cls_prob_aug[fg_mask].log(), cls_prob[fg_mask], reduction='none').sum(dim=1)  # [N]
            consistency_loss_cls = self.consistency_a * (w * kl_loss).mean()
# ======================================================================================================
            pos_mask = rois_label > 0
            ####    Tensor([ True,  True,  True, False, False, False, False, False, False, False,xxxxxx, ,,,,,]
            consistency_loss_loc = 0
            if pos_mask.sum() == 0:
                return torch.tensor(0.0, device=bbox_pred.device)

            bbox_pred_orig_pos = bbox_pred[pos_mask]
            bbox_pred_aug_pos = bbox_pred_aug[pos_mask]
            weights_pos = w
            loc_loss = F.mse_loss(bbox_pred_aug_pos, bbox_pred_orig_pos, reduction='none').sum(dim=1)   # [N]
            consistency_loss_loc = self.consistency_b * (loc_loss * weights_pos).mean()
# ======================================================================================================
            pos_mask = rois_label > 0  #  [512]
            bg_mask = rois_label == 0  #  [512]
            # 2.
            def apply_projection(feats):
                projected = self.contrastive_proj(feats)
                return F.normalize(projected, p=2, dim=1)

            #
            orig_pos_feats = apply_projection(pooled_feat[pos_mask]) if pos_mask.sum() > 0 else None
            aug_pos_feats = apply_projection(pooled_feat_aug[pos_mask]) if pos_mask.sum() > 0 else None
            orig_bg_feats = apply_projection(pooled_feat[bg_mask]) if bg_mask.sum() > 0 else None
            aug_bg_feats = apply_projection(pooled_feat_aug[bg_mask]) if bg_mask.sum() > 0 else None

            #
            contrastive_loss = torch.tensor(0.0, device=pooled_feat.device)

            #
            if orig_pos_feats is not None and orig_pos_feats.size(0) > 0:
                #
                w_pos = w  # [N_pos]

                ###6，128
                all_fore_feats = torch.cat([orig_pos_feats, aug_pos_feats], dim=0)  # [2*N_pos, D]
                ##
                intra_sim_matrix = torch.mm(all_fore_feats, all_fore_feats.t())  # [2*N_pos, 2*N_pos]
                #
                target_matrix = torch.ones_like(intra_sim_matrix)  #
                base_pull_loss = F.mse_loss(intra_sim_matrix, target_matrix, reduction='none')

                expanded_weights = torch.cat([w_pos, w_pos], dim=0)  # [2*N_pos]


                weight_matrix = expanded_weights.view(-1, 1) * expanded_weights.view(1, -1)

                weighted_pull_loss = (weight_matrix * base_pull_loss).mean()

                if orig_bg_feats is not None and orig_bg_feats.size(0) > 0 or \
                        aug_bg_feats is not None and aug_bg_feats.size(0) > 0:
                    all_bg_feats_list = []
                    if orig_bg_feats is not None and orig_bg_feats.size(0) > 0:
                        all_bg_feats_list.append(orig_bg_feats)
                    if aug_bg_feats is not None and aug_bg_feats.size(0) > 0:
                        all_bg_feats_list.append(aug_bg_feats)
                    all_bg_feats = torch.cat(all_bg_feats_list, dim=0)  # [M, D]

                    inter_sim_matrix = torch.mm(all_fore_feats, all_bg_feats.t())  # [2*N_pos, M]


                    target_bg = -torch.ones_like(inter_sim_matrix)


                    push_loss = F.smooth_l1_loss(inter_sim_matrix, target_bg, reduction='mean')
                else:
                    push_loss = torch.tensor(0.0, device=pooled_feat.device)
                contrastive_loss = self.contrastive_c * (weighted_pull_loss + push_loss)


            contrastive_loss = contrastive_loss
        else:
            contrastive_loss = torch.tensor(0.0, device=pooled_feat.device)


        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        RCNN_loss_cls_aug = 0
        RCNN_loss_bbox_aug = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            RCNN_loss_cls_aug = F.cross_entropy(cls_score_aug, rois_label)
            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
            RCNN_loss_bbox_aug = _smooth_l1_loss(bbox_pred_aug, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        if self.training:
            cls_prob_aug = cls_prob_aug.view(batch_size, rois.size(1), -1)
            bbox_pred_aug = bbox_pred_aug.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, \
               RCNN_loss_cls_aug, RCNN_loss_bbox_aug, consistency_loss_cls, consistency_loss_loc, contrastive_loss

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
