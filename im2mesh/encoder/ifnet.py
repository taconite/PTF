import torch
import torch.nn as nn
import torch.nn.functional as F

class IFNet(nn.Module):
    """
    Input: IFNet for feature extraction.
    """

    def __init__(self, tex=False, **kwargs):
        super(IFNet, self).__init__()

        self.tex = tex

        if tex:
            self.conv_00 = nn.Conv3d(3, 32, 3, padding=1)  # out: 128
        else:
            self.conv_00 = nn.Conv3d(1, 32, 3, padding=1)  # out: 128

        self.conv_01 = nn.Conv3d(32, 32, 3, padding=1)  # out: 128
        self.bn_01 = torch.nn.BatchNorm3d(32)

        self.conv_10 = nn.Conv3d(32, 64, 3, padding=1)  # out: 128
        self.conv_11 = nn.Conv3d(64, 64, 3, padding=1, stride=2)  # out: 64
        self.bn_11 = torch.nn.BatchNorm3d(64)

        self.conv_20 = nn.Conv3d(64, 64, 3, padding=1)  # out: 64
        self.conv_21 = nn.Conv3d(64, 64, 3, padding=1, stride=2)  # out: 32
        self.bn_21 = torch.nn.BatchNorm3d(64)

        self.conv_30 = nn.Conv3d(64, 128, 3, padding=1)  # out: 32
        self.conv_31 = nn.Conv3d(128, 128, 3, padding=1, stride=2)  # out: 16
        self.bn_31 = torch.nn.BatchNorm3d(128)

        self.conv_40 = nn.Conv3d(128, 128, 3, padding=1)  # out: 16
        self.conv_41 = nn.Conv3d(128, 128, 3, padding=1, stride=2)  # out: 8
        self.bn_41 = torch.nn.BatchNorm3d(128)

        # if tex:
        #     feature_size_parts = (3 + 32 + 64 + 64 + 128 + 128) * 7
        # else:
        #     feature_size_parts = (1 + 32 + 64 + 64 + 128 + 128) * 7

        # self.fc_parts_0 = nn.Conv1d(feature_size_parts, hidden_dim, 1)
        # self.fc_parts_1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        # self.fc_parts_out = nn.Conv1d(hidden_dim, num_parts, 1)
        # self.fc_parts_softmax = nn.Softmax(1)

        # if tex:
        #     feature_size = (3 + 32 + 64 + 64 + 128 + 128) * 7
        # else:
        #     feature_size = (1 + 32 + 64 + 64 + 128 + 128) * 7

        # # per-part classifiers
        # self.part_0 = nn.Conv1d(feature_size, hidden_dim * num_parts, 1)
        # self.part_1 = nn.Conv1d(hidden_dim * num_parts, hidden_dim * num_parts, 1, groups=num_parts)
        # self.part_2 = nn.Conv1d(hidden_dim * num_parts, hidden_dim * num_parts, 1, groups=num_parts)
        # self.part_out = nn.Conv1d(hidden_dim * num_parts, 3 * num_parts, 1,
        #                           groups=num_parts)  # we now predict 3 labels: in, between, out

        self.actvn = nn.ReLU()

        # displacment = 0.0722
        # displacments = [[0, 0, 0]]
        # for x in range(3):
        #     for y in [-1, 1]:
        #         input = [0, 0, 0]
        #         input[x] = y * displacment
        #         displacments.append(input)

        # self.displacments = torch.Tensor(displacments).cuda()

    def forward(self, x, **kwargs):
        if not self.tex:
            x = x.unsqueeze(1)

        # p = p.unsqueeze(1).unsqueeze(1)
        # p = torch.cat([p + d for d in self.displacments], dim=2)  # (B,1,7,num_samples,3)
        # full_0 = F.grid_sample(x, p)  # out : (B,C (of x), 1,1,sample_num)

        features = []
        features.append(x)

        net = self.actvn(self.conv_00(x))
        net = self.actvn(self.conv_01(net))
        net = self.bn_01(net)
        # full_1 = F.grid_sample(net, p)  # out : (B,C (of x), 1,1,sample_num)
        features.append(net)

        net = self.actvn(self.conv_10(net))
        net = self.actvn(self.conv_11(net))
        net = self.bn_11(net)
        # full_2 = F.grid_sample(net, p)  # out : (B,C (of x), 1,1,sample_num)
        features.append(net)

        net = self.actvn(self.conv_20(net))
        net = self.actvn(self.conv_21(net))
        net = self.bn_21(net)
        # full_3 = F.grid_sample(net, p)  # out : (B,C (of x), 1,1,sample_num)
        features.append(net)

        net = self.actvn(self.conv_30(net))
        net = self.actvn(self.conv_31(net))
        net = self.bn_31(net)
        # full_4 = F.grid_sample(net, p)
        features.append(net)

        net = self.actvn(self.conv_40(net))
        net = self.actvn(self.conv_41(net))
        net = self.bn_41(net)
        # full_5 = F.grid_sample(net, p)
        features.append(net)

        return features

        # full = torch.cat((full_0, full_1, full_2, full_3, full_4, full_5), dim=1)  # (B, features, 1,7,sample_num)
        # shape = full.shape
        # full = torch.reshape(full,
        #                      (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)

        # net_parts = self.actvn(self.fc_parts_0(full))
        # net_parts = self.actvn(self.fc_parts_1(net_parts))
        # out_parts = self.fc_parts_out(net_parts)

        # parts_softmax = self.fc_parts_softmax(out_parts)

        # net_full = self.actvn(self.part_0(full))
        # net_full = self.actvn(self.part_1(net_full))
        # net_full = self.actvn(self.part_2(net_full))

        # batch_sz = net_full.shape[0]
        # net_full = self.part_out(net_full).view(batch_sz, 3, self.num_parts, -1)
        # net_full *= parts_softmax.view(batch_sz, 1, self.num_parts, -1)
        # out_full = net_full.mean(2)

        # out = {'out': out_full, 'parts': out_parts}
        # return out
