import torch
import torch.nn as nn


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class ResnetBlockGroupConv1d(nn.Module):
    ''' Fully connected ResNet Block imeplemented with group convolutions.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, groups, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Conv1d(size_in, size_h, 1, groups=groups)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1, groups=groups)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False, groups=groups)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class ResnetBlockGroupNormConv1d(nn.Module):
    ''' Fully connected ResNet Block imeplemented with group convolutions and group normalizations.

    Args:
        size_in (int): input dimension
        groups (int): number of groups for group convolutions
        gn_groups (int): number of groups for group normalizations
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, groups, gn_groups=4, size_out=None, size_h=None, dropout_prob=0.0, leaky=False):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        if dropout_prob > 0.0:
            self.dropout = nn.Dropout(dropout_prob, inplace=True)
        else:
            self.dropout = None

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.gn_0 = GroupNorm1d(groups * gn_groups, size_in)
        self.gn_1 = GroupNorm1d(groups * gn_groups, size_h)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1, groups=groups, bias=False)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1, groups=groups, bias=False)
        if not leaky:
            self.actvn = nn.ReLU()
        else:
            self.actvn = nn.LeakyReLU(0.1)

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False, groups=groups)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        if self.dropout is not None:
            net = self.fc_0(self.dropout(self.actvn(self.gn_0(x))))
            dx = self.fc_1(self.dropout(self.actvn(self.gn_1(net))))
        else:
            net = self.fc_0(self.actvn(self.gn_0(x)))
            dx = self.fc_1(self.actvn(self.gn_1(net)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class ResnetBlockGroupNormShallowConv1d(nn.Module):
    ''' Fully connected ResNet Block imeplemented with group convolutions and group normalizations.

    Args:
        size_in (int): input dimension
        groups (int): number of groups for group convolutions
        gn_groups (int): number of groups for group normalizations
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, groups, gn_groups=4, size_out=None, size_h=None, dropout_prob=0.0, leaky=False):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        if dropout_prob > 0.0:
            self.dropout = nn.Dropout(dropout_prob, inplace=True)
        else:
            self.dropout = None

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.gn_0 = GroupNorm1d(groups * gn_groups, size_in)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1, groups=groups, bias=False)
        if not leaky:
            self.actvn = nn.ReLU()
        else:
            self.actvn = nn.LeakyReLU(0.1)

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False, groups=groups)

    def forward(self, x):
        if self.dropout is not None:
            dx = self.fc_0(self.dropout(self.actvn(self.gn_0(x))))
        else:
            dx = self.fc_0(self.actvn(self.gn_0(x)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class ResnetBlockInplaceNormShallowConv1d(nn.Module):
    ''' Fully connected ResNet Block imeplemented with group convolutions and weight/spectral normalizations.

    Args:
        size_in (int): input dimension
        groups (int): number of groups for group convolutions
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, groups, norm_method='weight_norm', size_out=None, size_h=None, dropout_prob=0.0, leaky=False):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        if dropout_prob > 0.0:
            self.dropout = nn.Dropout(dropout_prob, inplace=True)
        else:
            self.dropout = None

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        fc_0 = nn.Conv1d(size_in, size_h, 1, groups=groups, bias=False)
        if norm_method == 'weight_norm':
            self.fc_0 = nn.utils.weight_norm(fc_0)
        elif norm_method == 'spectral_norm':
            self.fc_0 = nn.utils.spectral_norm(fc_0)
        else:
            raise ValueError('Normalization method {} not supported.'.format(norm_method))

        if not leaky:
            self.actvn = nn.ReLU()
        else:
            self.actvn = nn.LeakyReLU(0.1)

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False, groups=groups)

    def forward(self, x):
        if self.dropout is not None:
            dx = self.fc_0(self.dropout(self.actvn(x)))
        else:
            dx = self.fc_0(self.actvn(x))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class ResnetBlockGroupBatchNormConv1d(nn.Module):
    ''' Fully connected ResNet Block imeplemented with group convolutions and group batch normalizations.

    Args:
        size_in (int): input dimension
        groups (int): number of groups for group convolutions
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, groups, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.bn_0 = GBatchNorm1d(size_in, groups)
        self.bn_1 = GBatchNorm1d(size_h, groups)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1, groups=groups, bias=False)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1, groups=groups, bias=False)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False, groups=groups)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(self.bn_0(x)))
        dx = self.fc_1(self.actvn(self.bn_1(net)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class CResnetBlockConv1d(nn.Module):
    ''' Conditional batch normalization-based Resnet block class.

    Args:
        c_dim (int): dimension of latend conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        norm_method (str): normalization method
        legacy (bool): whether to use legacy blocks
    '''

    def __init__(self, c_dim, size_in, size_h=None, size_out=None,
                 norm_method='batch_norm', legacy=False):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        if not legacy:
            self.bn_0 = CBatchNorm1d(
                c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d(
                c_dim, size_h, norm_method=norm_method)
        else:
            self.bn_0 = CBatchNorm1d_legacy(
                c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d_legacy(
                c_dim, size_h, norm_method=norm_method)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class ResnetBlockConv1d(nn.Module):
    ''' 1D-Convolutional ResNet block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_h=None, size_out=None):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.bn_0 = nn.BatchNorm1d(size_in)
        self.bn_1 = nn.BatchNorm1d(size_h)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)

        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(self.bn_0(x)))
        dx = self.fc_1(self.actvn(self.bn_1(net)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


# Utility modules
class AffineLayer(nn.Module):
    ''' Affine layer class.

    Args:
        c_dim (tensor): dimension of latent conditioned code c
        dim (int): input dimension
    '''

    def __init__(self, c_dim, dim=3):
        super().__init__()
        self.c_dim = c_dim
        self.dim = dim
        # Submodules
        self.fc_A = nn.Linear(c_dim, dim * dim)
        self.fc_b = nn.Linear(c_dim, dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.fc_A.weight)
        nn.init.zeros_(self.fc_b.weight)
        with torch.no_grad():
            self.fc_A.bias.copy_(torch.eye(3).view(-1))
            self.fc_b.bias.copy_(torch.tensor([0., 0., 2.]))

    def forward(self, x, p):
        assert(x.size(0) == p.size(0))
        assert(p.size(2) == self.dim)
        batch_size = x.size(0)
        A = self.fc_A(x).view(batch_size, 3, 3)
        b = self.fc_b(x).view(batch_size, 1, 3)
        out = p @ A + b
        return out


class CBatchNorm1d(nn.Module):
    ''' Conditional batch normalization layer class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''

    def __init__(self, c_dim, f_dim, norm_method='batch_norm'):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.conv_gamma = nn.Conv1d(c_dim, f_dim, 1)
        self.conv_beta = nn.Conv1d(c_dim, f_dim, 1)
        if norm_method == 'batch_norm':
            self.bn = nn.BatchNorm1d(f_dim, affine=False)
        elif norm_method == 'instance_norm':
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == 'group_norm':
            self.bn = nn.GroupNorm1d(f_dim, affine=False)
        else:
            raise ValueError('Invalid normalization method!')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x, c):
        assert(x.size(0) == c.size(0))
        assert(c.size(1) == self.c_dim)

        # c is assumed to be of size batch_size x c_dim x T
        if len(c.size()) == 2:
            c = c.unsqueeze(2)

        # Affine mapping
        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)

        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out


class GBatchNorm1d(nn.Module):
    ''' Group batch normalization layer class.

    Args:
        f_dim (int): feature dimension
    '''

    def __init__(self, f_dim, groups):
        super().__init__()
        self.f_dim = f_dim
        self.groups = groups
        assert (f_dim % groups == 0)
        # Submodules
        bn = [nn.BatchNorm1d(f_dim // groups) for _ in range(groups)]
        self.bn = nn.ModuleList(bn)

    def forward(self, x):
        net = torch.split(x, self.f_dim // self.groups, 1)
        out = torch.cat([self.bn[idx](net[idx]) for idx in range(len(self.bn))], dim=1)

        return out


class GroupNorm1d(nn.Module):
    ''' Group normalization that does per-point group normalization.

    Args:
        groups (int): number of groups
        f_dim (int): feature dimension, mush be divisible by groups
    '''

    def __init__(self, groups, f_dim, eps=1e-5, affine=True):
        super().__init__()
        self.groups = groups
        self.f_dim = f_dim
        self.affine = affine
        self.eps = eps
        assert (f_dim % groups == 0)
        # Affine parameters
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, f_dim, 1))
            self.beta = nn.Parameter(torch.zeros(1, f_dim, 1))

    def forward(self, x):
        batch_size, D, T = x.size()
        net = x.view(batch_size, self.groups, D // self.groups, T)

        means = net.mean(2, keepdim=True)
        variances = net.var(2, keepdim=True)

        net = (net - means) / (variances + self.eps).sqrt()

        net = net.view(batch_size, D, T)

        if self.affine:
            return net * self.gamma + self.beta
        else:
            return net


class CBatchNorm1d_legacy(nn.Module):
    ''' Conditional batch normalization legacy layer class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''

    def __init__(self, c_dim, f_dim, norm_method='batch_norm'):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.fc_gamma = nn.Linear(c_dim, f_dim)
        self.fc_beta = nn.Linear(c_dim, f_dim)
        if norm_method == 'batch_norm':
            self.bn = nn.BatchNorm1d(f_dim, affine=False)
        elif norm_method == 'instance_norm':
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == 'group_norm':
            self.bn = nn.GroupNorm1d(f_dim, affine=False)
        else:
            raise ValueError('Invalid normalization method!')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.fc_gamma.weight)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.ones_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, x, c):
        batch_size = x.size(0)
        # Affine mapping
        gamma = self.fc_gamma(c)
        beta = self.fc_beta(c)
        gamma = gamma.view(batch_size, self.f_dim, 1)
        beta = beta.view(batch_size, self.f_dim, 1)
        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out
