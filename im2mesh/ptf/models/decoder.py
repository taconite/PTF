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

class IPNetDecoder(nn.Module):
    ''' IPNet decoder.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size for both part classifier and occupancy classifier
        num_joints (int): number of joints/occupancy classifiers
        predict_ptfs (bool): predict additional translational vectors to A-pose
        use_coord (bool): concatenate coordinates to input features
        double_layer (bool): use double layer occupancy
    '''

    def __init__(self, dim=3, z_dim=128, c_dim=128,
                 hidden_size=256, num_joints=14,
                 predict_ptfs=False,
                 use_coord=False, double_layer=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.num_joints = num_joints
        self.double_layer = double_layer
        self.use_coord = use_coord

        self.fc_parts_0 = nn.Conv1d(c_dim + 3 if use_coord else c_dim, hidden_size, 1)
        self.fc_parts_1 = nn.Conv1d(hidden_size, hidden_size, 1)
        self.fc_parts_out = nn.Conv1d(hidden_size, num_joints, 1)
        self.fc_parts_softmax = nn.Softmax(1)

        # per-part classifiers
        self.part_0 = nn.Conv1d(c_dim + 3 if use_coord else c_dim, hidden_size * num_joints, 1)
        self.part_1 = nn.Conv1d(hidden_size * num_joints, hidden_size * num_joints, 1, groups=num_joints)
        self.part_2 = nn.Conv1d(hidden_size * num_joints, hidden_size * num_joints, 1, groups=num_joints)

        self.predict_ptfs = predict_ptfs
        if predict_ptfs:
            self.part_out = nn.Conv1d(hidden_size * num_joints, num_joints * 6 if double_layer else num_joints * 4, 1,
                                      groups=num_joints)  # predict (multi-class) occupancy and translational vectors
        else:
            self.part_out = nn.Conv1d(hidden_size * num_joints, num_joints * 3 if double_layer else num_joints, 1,
                                      groups=num_joints)  # predict (multi-class) occupancy

        self.actvn = nn.ReLU()

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        assert len(c.shape) == 3

        if self.use_coord:
            c = torch.cat([p, c], dim=1)

        net_parts = self.actvn(self.fc_parts_0(c))
        net_parts = self.actvn(self.fc_parts_1(net_parts))
        out_parts = self.fc_parts_out(net_parts)

        parts_softmax = self.fc_parts_softmax(out_parts)

        net_full = self.actvn(self.part_0(c))
        net_full = self.actvn(self.part_1(net_full))
        net_full = self.actvn(self.part_2(net_full))

        if self.double_layer:
            net_full = self.part_out(net_full).view(batch_size, self.num_joints, -1, T)
            if self.predict_ptfs:
                p_hat = net_full[:, :, 3:, :]
                net_full = net_full[:, :, :3, :]

            net_full *= parts_softmax.view(batch_size, self.num_joints, 1, -1)
        else:
            if self.predict_ptfs:
                raise NotImplementedError('PTFs prediction for binary ONet is not implemented yet')

            net_full = self.part_out(net_full).view(batch_size, self.num_joints, -1)
            net_full *= parts_softmax.view(batch_size, self.num_joints, -1)

        out_full = net_full.mean(1)

        if self.predict_ptfs:
            out = {'logits': out_full, 'out_cls': out_parts, 'p_hat': p_hat, 'parts_softmax': parts_softmax}
        else:
            out = {'logits': out_full, 'out_cls': out_parts, 'p_hat': torch.zeros(batch_size, 3, T, device=p.device, dtype=p.dtype)}

        return out


class PTFDecoder(nn.Module):
    ''' PTF Decoder.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size_full (int): hidden size of part classifier
        hidden_size_part (int): hidden size of piecewise occupancy classifiers
        num_joints (int): number of joints/occupancy classifiers
        double_layer (bool): use double layer occupancy
    '''

    def __init__(self, dim=3, z_dim=128, c_dim=128,
                 hidden_size_full=256, hidden_size_part=128,
                 num_joints=14, double_layer=False):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.num_joints = num_joints
        self.double_layer = double_layer

        # per-part classifier
        self.part_0 = nn.Conv1d((c_dim + dim) * num_joints, hidden_size_part * num_joints, 1, groups=num_joints)
        self.part_1 = nn.Conv1d(hidden_size_part * num_joints, hidden_size_part * num_joints, 1, groups=num_joints)
        self.part_2 = nn.Conv1d(hidden_size_part * num_joints, hidden_size_part * num_joints, 1, groups=num_joints)
        self.part_out = nn.Conv1d(hidden_size_part * num_joints, num_joints * 3 if double_layer else num_joints, 1,
                                  groups=num_joints)  # we now predict 3 labels: out, between, in

        self.actvn = lambda x: F.relu(x, True)

    def forward(self, p, z, c, **kwargs):
        batch_size, _, T = p.size()
        p_in = p.view(batch_size, self.num_joints, -1, T)

        parts_softmax = kwargs['parts_softmax'] # softmax probabilities from PTFs

        assert (len(c.shape) == 3)

        # Per-part classifier, this is fully piecewse, i.e. no inter-connections between
        # parts
        c_in = c.repeat(1, self.num_joints, 1).view(batch_size, self.num_joints, -1, T)
        c_in = torch.cat([p_in, c_in], dim=2).view(batch_size, -1, T) # B x num_joints x (3 + c_dim) x T -> B x (num_joints * (3 + c_dim)) x T

        net_occ = self.actvn(self.part_0(c_in))
        net_occ = self.actvn(self.part_1(net_occ))
        net_occ = self.actvn(self.part_2(net_occ))

        net_occ = self.part_out(net_occ)

        if self.double_layer:
            net_occ = net_occ.view(batch_size, self.num_joints, 3, -1)
            net_occ *= parts_softmax.view(batch_size, self.num_joints, 1, -1)
        else:
            net_occ *= parts_softmax

        out_occ = net_occ.mean(1) # sum or mean, difference is just a constant scale

        out_dict = {'logits': out_occ}

        return out_dict


class PTFPiecewiseDecoder(nn.Module):
    ''' PTF-piecewise decoder.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size for both part classifier and occupancy classifier
        num_joints (int): number of joints/occupancy classifiers
        double_layer (bool): use double layer occupancy
        bottleneck_size (int): size of the first layer after input features
    '''

    def __init__(self, dim=3, z_dim=128, c_dim=128,
                 hidden_size=256, num_joints=24,
                 double_layer=False, bottleneck_size=4):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.dim = dim
        self.num_joints = num_joints
        self.double_layer = double_layer

        # Submodules
        if bottleneck_size > 0:
            self.fc_p = nn.Conv1d((dim + bottleneck_size) * num_joints, hidden_size, 1, groups=num_joints)
        else:
            self.fc_p = nn.Conv1d((dim + c_dim) * num_joints, hidden_size, 1, groups=num_joints)

        self.block0 = nn.Conv1d(hidden_size, hidden_size, 1, groups=num_joints)
        self.block1 = nn.Conv1d(hidden_size, hidden_size, 1, groups=num_joints)
        self.block2 = nn.Conv1d(hidden_size, hidden_size, 1, groups=num_joints)

        self.fc_out = nn.Conv1d(hidden_size, num_joints * 3 if double_layer else num_joints, 1, groups=num_joints)

        if bottleneck_size > 0:
            self.proj = nn.Conv1d(num_joints * c_dim, num_joints * bottleneck_size, 1, groups=num_joints)
        else:
            self.proj = None

        self.actvn = F.relu

    def forward(self, p, z, c, **kwargs):
        batch_size, _, T = p.size()
        p_in = p.view(batch_size, self.num_joints, -1, T)
        D = p_in.size(2)
        p_in = p_in[:, :, D - self.dim:, :]

        if len(c.shape) < 3:
            c = c.repeat(1, self.num_joints).view(batch_size, -1, 1)  # B x c_dim -> B x (num_joints * c_dim) x 1
            if self.proj is not None:
                c_proj = self.proj(c).view(batch_size, self.num_joints, -1, 1).repeat(1, 1, 1, T)   # B x (num_joints * c_dim) x 1 -> B x (num_joints * bottleneck_size) x 1 -> B x num_joints x bottleneck_size x T
            else:
                c_proj = c.view(batch_size, self.num_joints, -1, 1).repeat(1, 1, 1, T)
        else:
            c = c.repeat(1, self.num_joints, 1)  # B x c_dim x T -> B x (num_joints * c_dim) x T
            if self.proj is not None:
                c_proj = self.proj(c).view(batch_size, self.num_joints, -1, T)    # B x (num_joints * c_dim) x T -> B x (num_joints * bottleneck_size) x T
            else:
                c_proj = c.view(batch_size, self.num_joints, -1, T)

        p_in_ = torch.cat([p_in, c_proj], dim=2).view(batch_size, -1, T) # B x num_joints x (3 + bottleneck_size) x T -> B x (num_joints * (3 + bottleneck_size)) x T

        net = self.actvn(self.fc_p(p_in_))

        net = self.actvn(self.block0(net))
        net = self.actvn(self.block1(net))
        net = self.actvn(self.block2(net))

        out = self.fc_out(net)  # B x num_joints x T or B x (num_joints * 3) x T

        if self.double_layer:
            out = out.view(batch_size, self.num_joints, -1, T)

        return {'logits': out}
