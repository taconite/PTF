import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

SMPL2IPNET_IDX = np.array([11, 12, 13, 11, 3, 8, 11, 1, 6, 11, 1, 6, 0, 11, 11, 0, 5, 10, 4, 9, 2, 7, 2, 7])
IPNET2SMPL_IDX = np.array([12, 7, 20, 4, 18, 16, 8, 21, 5, 19, 17, 0, 1, 2])
SMPL_parents = np.array([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
                    16, 17, 18, 19, 20, 21], dtype=np.int32)
IPNet_parents = np.array([11, 3, 4, 12, 5, 11, 8, 9, 13, 10, 11, -1, 11, 11], dtype=np.int32)
IPNet_parents_in_SMPL = np.array([9, 4, 18, 1, 16, 13, 5, 19, 2, 17, 14, -1, 0, 0], dtype=np.int32)
IPNet_kinematic_tree_order = np.array([11, 0, 12, 13, 5, 10, 3, 8, 4, 9, 1, 6, 2, 7], dtype=np.int32)

class PiecewiseTransformationField(nn.Module):
    ''' Piecewise Transformation Field network class.

    It maps input points together with (optional) conditioned
    codes c and latent codes z to the respective A-pose space for every bone.

    Args:
        in_dim (int): input dimension of points concatenated with the time axis
        out_dim (int): output dimension of motion vectors
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size_full (int): hidden size of part classifier
        hidden_size_part (int): hidden size of piecewise transformation fields
        num_joints (int): number of joints/piecewise transformation fields
        full_smpl (bool): if set ot True, merge 24-dimensional output from part classifiers
            into 14 parts
        non_piecewise: if set to True, use a single MLP to predict translational vectors for
            all parts
        residual: if set to True, predict translations as residuals to input coordinates
    '''

    def __init__(self, in_dim=3, out_dim=3, z_dim=128, c_dim=128,
                 hidden_size_full=256, hidden_size_part=128,
                 num_joints=24, full_smpl=False,
                 non_piecewise=False, residual=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.num_joints = num_joints
        self.residual = residual
        self.full_smpl = full_smpl

        self.fc_parts_0 = nn.Conv1d(c_dim, hidden_size_full, 1)
        self.fc_parts_1 = nn.Conv1d(hidden_size_full, hidden_size_full, 1)
        self.fc_parts_out = nn.Conv1d(hidden_size_full, 24 if full_smpl else num_joints, 1)
        self.fc_parts_softmax = nn.Softmax(1)

        # per-part regressor
        self.non_piecewise = non_piecewise
        if non_piecewise:
            self.part_0 = nn.Conv1d(c_dim, hidden_size_part, 1)
            self.part_1 = nn.Conv1d(hidden_size_part, hidden_size_part, 1)
            self.part_2 = nn.Conv1d(hidden_size_part, hidden_size_part, 1)
            self.part_out = nn.Conv1d(hidden_size_part, 3, 1)
        else:
            self.part_0 = nn.Conv1d(c_dim, hidden_size_part * num_joints, 1)
            self.part_1 = nn.Conv1d(hidden_size_part * num_joints, hidden_size_part * num_joints, 1, groups=num_joints)
            self.part_2 = nn.Conv1d(hidden_size_part * num_joints, hidden_size_part * num_joints, 1, groups=num_joints)
            self.part_out = nn.Conv1d(hidden_size_part * num_joints, num_joints * 3, 1, groups=num_joints)

        self.actvn = lambda x: F.relu(x, True)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, _, T = p.size()

        assert len(c.shape) == 3

        net_parts = self.actvn(self.fc_parts_0(c))
        net_parts = self.actvn(self.fc_parts_1(net_parts))
        out_parts = self.fc_parts_out(net_parts)

        parts_softmax = self.fc_parts_softmax(out_parts)

        if self.full_smpl:
            assert (self.num_joints == 14)
            index = torch.from_numpy(SMPL2IPNET_IDX).to(parts_softmax.device).long().view(1, -1, 1).repeat(batch_size, 1, T)
            parts_softmax_ = torch.zeros(batch_size, self.num_joints, T, device=parts_softmax.device, dtype=parts_softmax.dtype)
            parts_softmax_.scatter_add_(1, index, parts_softmax)
            parts_softmax = parts_softmax_

        net_t = self.actvn(self.part_0(c))
        net_t = self.actvn(self.part_1(net_t))
        net_t = self.actvn(self.part_2(net_t))

        net_t = self.part_out(net_t)

        if self.non_piecewise:
            net_t = net_t.repeat(1, self.num_joints, 1)

        if self.residual:
            net_t = net_t + p.repeat(1, self.num_joints, 1)

        out_dict = {'p_hat': net_t, 'out_cls': out_parts, 'parts_softmax': parts_softmax}

        return out_dict

class PiecewiseFullTransformationField(nn.Module):
    ''' Fully Piecewise Transformation Field network class.

    It maps input points together with (optional) conditioned
    codes c and latent codes z to the respective A-pose space for every bone.

    Args:
        in_dim (int): input dimension of points concatenated with the time axis
        out_dim (int): output dimension of motion vectors
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): size of the hidden dimension
        num_joints (int): number of joints/piecewise transformation fields
        use_coord (bool): concatenate coordinates to input features
        bottleneck_size (int): size of the first layer after input features
        residual: if set to True, predict translations as residuals to input coordinates
    '''

    def __init__(self, in_dim=3, out_dim=3, z_dim=128, c_dim=128,
                 hidden_size=256, num_joints=24,
                 use_coord=True,
                 bottleneck_size=4, residual=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.num_joints = num_joints
        self.residual = residual
        self.use_coord = use_coord

        # Submodules
        if bottleneck_size > 0:
            self.fc_p = nn.Conv1d((in_dim + bottleneck_size) * num_joints if use_coord else bottleneck_size * num_joints, hidden_size, 1, groups=num_joints)
        else:
            self.fc_p = nn.Conv1d((in_dim + c_dim) * num_joints if use_coord else c_dim * num_joints, hidden_size, 1, groups=num_joints)

        self.block0 = nn.Conv1d(hidden_size, hidden_size, 1, groups=num_joints)
        self.block1 = nn.Conv1d(hidden_size, hidden_size, 1, groups=num_joints)

        self.fc_out = nn.Conv1d(hidden_size, num_joints * out_dim, 1, groups=num_joints)

        if bottleneck_size > 0:
            self.proj = nn.Conv1d(num_joints * c_dim, num_joints * bottleneck_size, 1, groups=num_joints)
        else:
            self.proj = None

        self.actvn = F.relu

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2).repeat(1, self.num_joints, 1)
        batch_size, _, T = p.size()
        p_in = p.view(batch_size, self.num_joints, -1, T)
        if len(c.shape) < 3:
            c = c.repeat(1, self.num_joints).unsqueeze(-1)  # B x c_dim -> B x (num_joints * c_dim) x 1
            if self.proj is not None:
                c_proj = self.proj(c).view(batch_size, self.num_joints, -1, 1).repeat(1, 1, 1, T)   # B x (num_joints * c_dim) x 1 -> B x (num_joints * bottleneck_size) x 1 -> B x num_joints x bottleneck_size x T
            else:
                c_proj = c.view(batch_size, self.num_joints, -1, 1).repeat(1, 1, 1, T)
        else:
            c = c.repeat(1, self.num_joints, 1)  # B x c_dim x T -> B x (num_joints * c_dim) x T
            if self.proj is not None:
                c_proj = self.proj(c).view(batch_size, self.num_joints, -1, T)   # B x (num_joints * c_dim) x T -> B x (num_joints * bottleneck_size) x T -> B x num_joints x bottleneck_size x T
            else:
                c_proj = c.view(batch_size, self.num_joints, -1, T)

        if self.use_coord:
            p_in = torch.cat([p_in, c_proj], dim=2).view(batch_size, -1, T) # B x num_joints x (3 + bottleneck_size) x T -> B x (num_joints * (3 + bottleneck_size)) x T
        else:
            p_in = c_proj.view(batch_size, -1, T)

        net = self.actvn(self.fc_p(p_in))

        net = self.actvn(self.block0(net))
        net = self.actvn(self.block1(net))

        out = self.fc_out(net)  # B x (num_joints * 3) x T

        if self.residual:
            return {'p_hat': out + p}
        else:
            return {'p_hat': out}
