import os
import numbers
import numpy as np
from tqdm import trange
import torch
from im2mesh.ptf import models
from im2mesh import encoder
from torch.nn import functional as F
from torch import distributions as dist

from im2mesh.utils.focalloss import FocalLoss

from im2mesh.common import (
    compute_iou, make_3d_grid
)
from im2mesh.training import BaseTrainer
from collections import defaultdict

IPNet_parts = ['head', 'left_foot', 'left_forearm', 'left_leg', 'left_midarm', 'left_upperarm', 'right_foot', 'right_forearm', 'right_leg', 'right_midarm', 'right_upperarm', 'torso', 'upper_left_leg', 'upper_right_leg']

class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        skinning_weight (float): weight for skinning loss
        use_corr_loss_pred (bool): whether to use correspondence loss with predicted labels
        corr_weight (float): weight for correspondences loss
        device (device): pytorch device
        input_type (str): input type
        threshold (float): threshold value for evaluating IoU
        num_joints (int): number of joints, should be either 14 or 24
        max_operator (str): max operator used for final occupancy value of IP-Net and PTF.
            For PTF-piecewise, it can be either 'lse' or 'max'; for IP-Net and PTF it must
            be 'softmax'.
        occ_loss_type (str): occupancy loss type, currenlty only support binary cross-entropy (ce)
        occ_loss_type (str): skinning loss type, currenlty only support binary cross-entropy (ce)

    '''

    def __init__(self, model, optimizer, skinning_weight,
                 use_corr_loss_pred, corr_weight=1.0, device=None,
                 input_type='pointcloud', threshold=0.5,
                 num_joints=24, max_operator='lse', occ_loss_type='ce', skin_loss_type='ce',
                 **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.threshold = threshold
        self.num_joints = num_joints

        self.use_corr_loss_pred = use_corr_loss_pred

        self.skinning_weight = skinning_weight
        self.corr_weight = corr_weight

        self.model_counter = defaultdict(int)

        self.max_operator = max_operator
        self.occ_loss_type = occ_loss_type
        self.skin_loss_type = skin_loss_type

        # Focal loss is experimental, not used in current version
        if self.occ_loss_type == 'fl':
            fl_gamma = kwargs['fl_gamma']
            fl_alpha = kwargs['fl_alpha']
            self.focal_loss = FocalLoss(gamma=fl_gamma, alpha=fl_alpha, device=device)

        try:
            self.model_type = self.model.module.model_type
            self.tf_type = self.model.module.tf_type
        except:
            self.model_type = self.model.model_type
            self.tf_type = self.model.tf_type

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss_dict = self.compute_loss(data)
        loss_dict['total_loss'].backward()
        self.optimizer.step()
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}

    def eval_step(self, data, model_dict=None):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)
        root_locs = data.get('points_iou.root_loc').to(device)
        trans = data.get('points_iou.trans').to(device)
        loc = data.get('points_iou.loc').to(device)
        # bone_transforms = data.get('points_iou.bone_transforms').to(device)
        # bone_transforms_inv = data.get('points_iou.bone_transforms_inv').to(device)    # B x num_joints x 4 x 4
        batch_size, T, D = points_iou.size()

        occ_iou = occ_iou[:, :]

        kwargs = {}
        scale = data.get('points_iou.scale').to(device)
        kwargs.update({'scale': scale.view(-1, 1, 1)}) #, 'bone_transforms_inv': bone_transforms_inv})

        with torch.no_grad():
            # Encoder inputs
            inputs = data.get('inputs', torch.empty(1, 1, 0)).to(device)
            mask = torch.ones(batch_size, T, dtype=points_iou.dtype, device=points_iou.device)

            # Decode occupancies
            out_dict = self.model(points_iou, inputs, **kwargs)
            logits = out_dict['logits']

            if len(logits.shape) == 4:
                # PTF-piecewise predictions
                logits = torch.max(logits, dim=1)[0]
                p_out = dist.Multinomial(logits=logits.transpose(1, 2))
            elif len(logits.shape) == 3:
                # IPNet/PTF predictions
                p_out = dist.Multinomial(logits=logits.transpose(1, 2))
            else:
                raise ValueError('Wrong logits shape')

        # Compute iou
        occ_iou_np = ((occ_iou >= 0.5) * mask).cpu().numpy()
        if len(logits.shape) == 3:
            # IoU for outer surface; we just want an easy-to-compute indicator for model selection
            occ_iou_hat_np = ((p_out.probs[:, :, 1:].sum(-1) >= threshold) * mask).cpu().numpy()
        else:
            raise ValueError('Wrong logits shape')

        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        eval_dict['iou'] = iou

        return eval_dict

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p = data.get('points').to(device)
        n_p_occ = p.size(1)

        p_corr = data.get('points.points_corr')

        if p_corr is not None:
            p_corr = p_corr.to(device)
            n_p_corr = p_corr.size(1)
            p = torch.cat([p, p_corr], dim=1)
        else:
            n_p_corr = 0

        batch_size = p.size(0)
        occ = data.get('points.occ').to(device)
        # bone_transforms = data.get('points.bone_transforms').to(device)    # B x num_joints x 4 x 4
        # bone_transforms_inv = data.get('points.bone_transforms_inv').to(device)    # B x num_joints x 4 x 4
        root_locs = data.get('points.root_loc').to(device)
        trans = data.get('points.trans').to(device)
        loc = data.get('points.loc').to(device)

        kwargs = {}
        scale = data.get('points.scale').to(device)
        kwargs.update({'scale': scale.view(-1, 1, 1)}) #, 'bone_transforms_inv': bone_transforms_inv})

        inputs = data.get('inputs', torch.empty(1, 1, 0)).to(device)

        z = None    # no prior

        loss_dict = {}

        if self.model_type in ['ipnet', 'deformable']:
            # Models for IP-Net and PTF
            # Get labels for skinning loss
            s_targets = data.get('points.skinning_inds').to(self.device).long()
            s_targets_ipnet = data.get('points.skinning_inds_ipnet').to(self.device).long()
            s_targets_smpl = data.get('points.skinning_inds_smpl').to(self.device).long()
            # Network computation
            out_dict = self.model(p, inputs, **kwargs)
            logits = out_dict['logits']
            p_hat = out_dict['p_hat']
            p_hat_template = out_dict.get('p_hat_template', None)   # get predicted offset in canonical space, if available

            if n_p_corr > 0:
                logits = logits[:, :, :n_p_occ]
                p_hat = p_hat[:, :, n_p_occ:]
                if p_hat_template is not None:
                    p_hat_template = p_hat_template[:, :, n_p_occ:]

            # Occupancy loss
            if self.max_operator == 'lse':
                occ_logits = torch.logsumexp(logits, dim=1)    # smoothed-max over logits, for PTF-piecewise only
            elif self.max_operator == 'max':
                occ_logits = torch.max(logits, dim=1)[0]    # max over logits, for PTF-piecewise only
            elif self.max_operator == 'softmax':
                occ_logits = logits     # do nothing, softmax was already applied inside IP-Net/PTF decoder
            else:
                raise ValueError('Max operator type {} is not supported'.format(self.max_operator))

            if self.occ_loss_type == 'ce':
                if len(occ_logits.shape) == 3:
                    # Double-layer prediction
                    loss_occ = F.cross_entropy(
                        occ_logits, occ, reduction='none').sum(-1).mean()
                else:
                    # Single-layer prediction
                    loss_occ = F.binary_cross_entropy_with_logits(
                        occ_logits, occ, reduction='none').sum(-1).mean()
            else:
                raise ValueError('Occupancy loss type {} is not supported'.format(self.occ_loss_type))

            # Compute IoUs for this batch
            if len(occ_logits.shape) == 3:
                # Double-layer prediction
                occ_hat_np = ((F.softmax(occ_logits, dim=1)[:, 1:, :].sum(1)) > 0.5).detach().cpu().numpy()
            else:
                # Single-layer prediction
                occ_hat_np = (torch.sigmoid(occ_logits) > 0.5).detach().cpu().numpy()

            occ_np = (occ > 0.5).detach().cpu().numpy()
            ious = compute_iou(occ_np, occ_hat_np).flatten()
            loss_dict['iou'] = ious

            # Skinning loss with PTFs predictions
            if self.max_operator == 'softmax':
                # For IP-Net/PTF
                out_cls = out_dict['out_cls']
                if n_p_corr > 0:
                    out_cls = out_cls[:, :, n_p_occ:]

                if self.skin_loss_type == 'ce':
                    loss_skin = F.cross_entropy(out_cls, s_targets, reduction='none').sum(-1).mean()
                else:
                    raise ValueError('Skinning loss type {} is not supported'.format(self.skin_loss_type))
            elif self.max_operator in ['lse']:
                # For PTF-piecewise
                if len(logits.shape) == 4:
                    # Double layer prediction
                    skin_logits = torch.logsumexp(logits, dim=2)    # smoothed-max over class (i.e. inside, in-between, outside) dimension
                else:
                    # Single layer prediction
                    skin_logits = logits

                if self.skin_loss_type == 'ce':
                    loss_skin = F.cross_entropy(skin_logits, s_targets, reduction='none').sum(-1).mean()
                else:
                    raise ValueError('Skinning loss type {} is not supported'.format(self.skin_loss_type))
            else:
                raise ValueError('Max operator type {} is not supported'.format(self.max_operator))

            if self.use_corr_loss_pred: # and self.tf_type is not None:
                parts_softmax = out_dict.get('parts_softmax', None)
                # Compute p_hat
                if parts_softmax is not None:
                    # For IP-Net/PTF, who has a separate branch to predict part probabilities
                    if n_p_corr > 0:
                        parts_softmax = parts_softmax[:, :, n_p_occ:]

                    p_hat = p_hat.view(batch_size, -1, 3, n_p_corr if n_p_corr > 0 else n_p_occ)
                    p_hat = p_hat * parts_softmax.view(batch_size, -1, 1, n_p_corr if n_p_corr > 0 else n_p_occ)
                    if p_hat_template is not None:
                        p_hat_template = p_hat_template.view(batch_size, -1, 3, n_p_corr if n_p_corr > 0 else n_p_occ)
                        p_hat_template = p_hat_template * parts_softmax.view(batch_size, -1, 1, n_p_corr if n_p_corr > 0 else n_p_occ)
                else:
                    # For PTF-piecewise, who does not predict part probabilities directly
                    p_hat = p_hat.view(batch_size, -1, 3, n_p_corr if n_p_corr > 0 else n_p_occ)
                    p_hat = p_hat * F.softmax(skin_logits, dim=1).view(batch_size, -1, 1, n_p_corr if n_p_corr > 0 else n_p_occ)
                    if p_hat_template is not None:
                        p_hat_template = p_hat_template.view(batch_size, -1, 3, n_p_corr if n_p_corr > 0 else n_p_occ)
                        p_hat_template = p_hat_template * F.softmax(skin_logits, dim=1).view(batch_size, -1, 1, n_p_corr if n_p_corr > 0 else n_p_occ)

                p_hat = p_hat.sum(1).transpose(1, 2)
                if p_hat_template is not None:
                    p_hat_template = p_hat_template.sum(1).transpose(1, 2)

            # Correspondence loss with predicted skinning inds
            loss_corr_dict = {}
            if self.use_corr_loss_pred: # and self.tf_type is not None:
                p_a_pose = data.get('points.pts_a_pose').to(device)
                p_template = data.get('points.pts_template')

                if p_template is not None:
                    p_template = p_template.to(device)

                loss_corr_pred_all = torch.norm(p_hat - p_a_pose, 2, dim=-1)
                loss_corr_pred = loss_corr_pred_all.sum(-1).mean()
                if p_hat_template is not None:
                    loss_template_pred = torch.norm(p_hat_template - p_template, 2, dim=-1).sum(-1).mean()
                else:
                    loss_template_pred = torch.zeros(1, device=self.device)
            else:
                loss_corr_pred = torch.zeros(1, device=self.device)
                loss_template_pred = torch.zeros(1, device=self.device)

            loss_dict['occ_loss'] = loss_occ
            loss_dict['skinning_loss'] = loss_skin
            loss_dict['corr_loss'] = loss_corr_pred
            loss_dict['template_loss'] = loss_template_pred
            loss_dict.update(loss_corr_dict)

            # Total weighted sum
            loss = loss_occ + self.skinning_weight * loss_skin + self.corr_weight * (loss_corr_pred + loss_template_pred)
            loss_dict['total_loss'] = loss
        else:
            raise ValueError('Supported model type: ipnet, deformable, got {}'.format(self.model_type))

        return loss_dict
