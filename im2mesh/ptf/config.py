import torch
import torch.distributions as dist
import numpy as np
from torchvision import transforms
from torch import nn
import os
from im2mesh.encoder import encoder_dict
from im2mesh.ptf import models, training, generation
from im2mesh import data
from im2mesh import config


def get_decoder(cfg, device, dim=3, c_dim=0, z_dim=0):
    ''' Returns a decoder instance.

    Args:
        cfg (yaml config): yaml config object
        device (device): PyTorch device
        dim (int): points dimension
        c_dim (int): dimension of conditioned code c
        z_dim (int): dimension of latent code z
    '''
    decoder = cfg['model']['decoder']
    decoder_kwargs = cfg['model']['decoder_kwargs']

    if decoder:
        decoder = models.decoder_dict[decoder](
            dim=dim, z_dim=z_dim, c_dim=c_dim,
            **decoder_kwargs).to(device)
    else:
        decoder = None

    return decoder


def get_encoder(cfg, device, c_dim=0):
    ''' Returns an encoder instance.

    Args:
        cfg (yaml config): yaml config object
        device (device): PyTorch device
        c_dim (int): dimension of conditioned code c
    '''
    encoder = cfg['model']['encoder']
    encoder_kwargs = cfg['model']['encoder_kwargs']

    encoder_kwargs.update({'normalized_scale': cfg['data']['normalized_scale']})

    if encoder is not None:
        if encoder in ['pointnet_conv']:
            # For ConvONet encoder, the output c_dim is concatenation of features from 3 planes
            encoder = encoder_dict[encoder](
                c_dim=c_dim // 3, **encoder_kwargs).to(device)
        else:
            encoder = encoder_dict[encoder](**encoder_kwargs).to(device)
    else:
        encoder = None

    return encoder


def get_transform_field(cfg, device, dim=3, c_dim=0, z_dim=0):
    ''' Returns a transformation field instance.

    Args:
        cfg (yaml config): yaml config object
        device (device): PyTorch device
        dim (int): points dimension
        c_dim (int): dimension of conditioned code c
        z_dim (int): dimension of latent code z
    '''
    transform_field = cfg['model']['transform_field']
    transform_field_kwargs = cfg['model']['transform_field_kwargs']

    if transform_field:
        transform_field = models.transform_field_dict[transform_field](
            out_dim=dim, z_dim=z_dim,
            c_dim=c_dim, **transform_field_kwargs
        ).to(device)
    else:
        transform_field = None

    return transform_field


def get_model(cfg, device=None, **kwargs):
    ''' Return the PTF model.

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    encoder = cfg['model']['encoder']
    decoder = cfg['model']['decoder']
    dim = cfg['data']['dim']
    tf_dim = cfg['data']['tf_dim']
    use_global_trans = cfg['data']['use_global_trans']
    z_dim = cfg['model']['z_dim']
    c_dim = cfg['model']['c_dim']

    decoder = get_decoder(cfg, device, dim, c_dim, z_dim)
    encoder = get_encoder(cfg, device, c_dim)
    transform_field = get_transform_field(cfg, device, tf_dim, c_dim, z_dim)

    # Get full PTF model
    model = models.PTF(
        decoder=decoder, encoder=encoder,
        transform_field=transform_field,
        device=device)

    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    input_type = cfg['data']['input_type']
    batch_size = cfg['training']['batch_size']
    model_type = cfg['model']['decoder']
    num_joints = cfg['model']['num_joints']

    skinning_weight = cfg['training']['skinning_weight']
    corr_weight = cfg['training']['corr_weight']
    use_corr_loss_pred = cfg['training']['use_corr_loss_pred']
    occ_loss_type = cfg['training']['occ_loss_type']
    skin_loss_type = cfg['training']['skin_loss_type']

    max_operator = cfg['training']['max_operator']

    kwargs = {}
    if occ_loss_type == 'fl':
        kwargs.update({
            'fl_gamma': cfg['training']['fl_gamma'],
            'fl_alpha': cfg['training']['fl_alpha'],
        })

    trainer = training.Trainer(
        model, optimizer,
        skinning_weight, use_corr_loss_pred,
        device=device, input_type=input_type,
        hreshold=threshold,
        max_operator=max_operator,
        occ_loss_type=occ_loss_type,
        skin_loss_type=skin_loss_type,
        corr_weight=corr_weight,
        num_joints=num_joints,
        **kwargs
    )

    return trainer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    double_layer = cfg['model']['decoder_kwargs'].get('double_layer', False)

    generator = generation.Generator3D(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        input_type=cfg['data']['input_type'],
        num_joints=cfg['model']['num_joints'],
        double_layer=double_layer,
    )

    return generator
