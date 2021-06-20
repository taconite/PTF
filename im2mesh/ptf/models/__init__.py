import torch
import torch.nn as nn
import numbers
# from torch import distributions as dist
from im2mesh import encoder
from im2mesh.encoder.pointnet import normalize_coordinate, normalize_3d_coordinate
from im2mesh.ptf.models import (
    decoder, transform_field)

# Decoder dictionary
decoder_dict = {
    'ipnet': decoder.IPNetDecoder,
    'ptf': decoder.PTFDecoder,
    'ptf_piecewise': decoder.PTFPiecewiseDecoder,
}

# TransformField dictionary
transform_field_dict = {
    'ptf': transform_field.PiecewiseTransformationField,
    'ptf_piecewise': transform_field.PiecewiseFullTransformationField,
}


class PTF(nn.Module):
    ''' PTF model class.

    It consists of a decoder and an encoder.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        transform_field (nn.Module): transform field network
        device (device): PyTorch device

    '''

    def __init__(
        self, decoder, encoder=None,
            transform_field=None,
            device=None, **kwargs):
        super().__init__()

        self.device = device

        self.decoder = decoder

        if transform_field is None:
            self.tf_type = None
        elif isinstance(transform_field, transform_field_dict['ptf']) or isinstance(transform_field, transform_field_dict['ptf_piecewise']):
            self.tf_type = 'ptf'
        else:
            raise NotImplementedError('Transform field model not supported yet')

        if isinstance(decoder, decoder_dict['ptf']) or isinstance(decoder, decoder_dict['ptf_piecewise']):
            self.model_type = 'deformable'
        elif isinstance(decoder, decoder_dict['ipnet']):
            self.model_type = 'ipnet'
        else:
            raise NotImplementedError('Decoder model not supported yet')

        self.encoder = encoder
        self.transform_field = transform_field

        # Displacements for IFNet encoder
        displacement = 0.0722
        displacements = [[0, 0, 0]]
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacement
                displacements.append(input)

        self.displacements = torch.Tensor(displacements).to(self.device)

    def forward(self, p, inputs, **kwargs):
        ''' Makes a forward pass through the network.

        Args:
            p (tensor): points tensor
            time_val (tensor): time values
            inputs (tensor): input tensor
        '''
        batch_size, T, _ = p.size()

        c = self.encode_inputs(inputs, **kwargs)
        z = None
        c_p = self.get_point_features(p, c=c, **kwargs)

        if self.tf_type not in [None, 'unstructured_ptf', 'unstructured_tf', 'ipnet']:
            tf_dict = self.transform_points(p, z=z, c=c_p, **kwargs)
        else:
            p_hat = p

        if self.model_type in ['unstructured_ptf', 'unstructured_tf', 'ipnet']:
            out_dict = self.decode(p_hat, c=c_p, z=z, **kwargs)
        else:
            p_hat = tf_dict['p_hat']
            if 'parts_softmax' in tf_dict.keys():
                kwargs.update({'parts_softmax': tf_dict['parts_softmax']})
            #     p_hat_out = p_hat.view(batch_size, -1, 3, T)
            #     p_hat_out = p_hat_out * tf_dict['parts_softmax'].view(batch_size, -1, 1, T)
            #     p_hat_out = p_hat_out.sum(1).transpose(1, 2)

            out_dict = self.decode(p_hat, c=c_p, z=z, **kwargs)
            out_dict.update(tf_dict)

            # out_dict = {'logits': occ_dict['logits'], 'p_hat': p_hat, 'out_cls': tf_dict['out_cls']}

        return out_dict

    def encode_inputs(self, inputs, **kwargs):
        ''' Returns the encoding.

        Args:
            inputs (tensor): input tensor)
        '''
        batch_size = inputs.shape[0]
        device = self.device

        if self.encoder is not None:
            c = self.encoder(inputs, **kwargs)
        else:
            c = torch.empty(batch_size, 0).to(device)

        return c

    def transform_points(self, p, z=None, c=None, **kwargs):
        ''' Returns transformed points p_hat from input points p.

        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c (For OFlow, this is
                c_spatial)
        '''
        if self.tf_type in ['tf', 'ptf']:
            # Different transformation field model:
            # tf - a single transformation field for all points
            # ptf - piecewise transformation fields
            out_dict = self.transform_field(p, z=z, c=c, **kwargs)
        elif self.tf_type in ['unstructured_ptf', 'unstructured_tf']:
            # For these, we don't have a separate transformation field model.
            # The translational vectors are predicted by the decoder.
            out_dict = self.decoder(p, z=z, c=c, **kwargs)
        else:
            raise ValueError('Transformation field type not supported: {}.'.format(self.tf_type))

        return out_dict

    def get_point_features(self, p, c=None, **kwargs):
        ''' Returns point-aligned features from convolutional encoder or global feature from global encoder.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''
        # Supported encoder types:
        # pointnet_conv - ConvONet encoder
        # ifnet - IFNet encoder
        # for other cases, we assume it's some global point-cloud encoder (e.g. PointNet)
        if isinstance(self.encoder, encoder.encoder_dict['pointnet_conv']):
            # ConvONet features
            point_features = []
            if self.encoder.normalized_scale:
                scale = 1.0
            else:
                scale = kwargs['scale']

            for k, v in c.items():
                if k in ['xz', 'xy', 'yz']:
                    projected = normalize_coordinate(p.clone(), scale, plane=k, padding=self.encoder.padding) # normalize to the range of (0, 1)
                    projected = (projected * 2 - 1).unsqueeze(2)    # grid_sample accepts inputs in range [-1, 1]
                    if isinstance(v, list):
                        # In case of multi-resolution networks, e.g. HRNet
                        for v_ in v:
                            point_features.append(nn.functional.grid_sample(v_, projected, align_corners=True).squeeze(-1))
                    else:
                        point_features.append(nn.functional.grid_sample(v, projected, align_corners=True).squeeze(-1))
                elif k in ['grid']:
                    # Note: this part should work, but not fully tested
                    p_nor = normalize_3d_coordinate(p.clone(), scale, padding=self.encoder.padding) # normalize to the range of (0, 1)
                    p_nor = p_nor.unsqueeze(1).unsqueeze(1)
                    vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
                    point_features.append(nn.functional.grid_sample(v, vgrid, align_corners=True).squeeze(2).squeeze(2))
                else:
                    raise ValueError('Wrong type of convolutional feature: {}.'.format(k))

            return torch.cat(point_features, dim=1)    # B x c_dim x T
        elif isinstance(self.encoder, encoder.encoder_dict['ifnet']):
            # IF-Net features
            point_features = []

            batch_size, n_pts, _ = p.size()
            p = p.clone()   # We DON'T swap x and z because the voxels were constructed as a (W, H, D) tensor during preprocessing
            # p[:, :, 0], p[:, :, 2] = p[:, :, 2], p[:, :, 0]
            p = p.unsqueeze(1).unsqueeze(1)
            p = torch.cat([p + d for d in self.displacements], dim=2)  # (B,1,7,num_samples,3)

            for c_ in c:
                point_features.append(
                        nn.functional.grid_sample(c_, p, align_corners=True).squeeze(2).view(batch_size, -1, n_pts)
                    )

            return torch.cat(point_features, dim=1)    # B x c_dim x T
        else:
            # Global feature, do nothing and just return the input feature
            return c

    def decode(self, p, z=None, c=None, **kwargs):
        ''' Returns occupancy values for the points p.

        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        logits = self.decoder(p, z=z, c=c, **kwargs)
        return logits
