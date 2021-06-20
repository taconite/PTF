import yaml
from im2mesh import data
from im2mesh import ptf

method_dict = {
    'ptf': ptf,
}

# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


# Models
def get_model(cfg, device=None, dataset=None):
    ''' Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    method = cfg['method']
    model = method_dict[method].config.get_model(
        cfg, device=device, dataset=dataset)
    return model


# Trainer
def get_trainer(model, optimizer, cfg, device):
    ''' Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    trainer = method_dict[method].config.get_trainer(
        model, optimizer, cfg, device)
    return trainer


# Generator for final mesh extraction
def get_generator(model, cfg, device):
    ''' Returns a generator instance.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    generator = method_dict[method].config.get_generator(model, cfg, device)
    return generator


# Datasets
def get_dataset(mode, cfg, sequence_idx=None, subject_idx=None):
    ''' Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        sequence_idx (int or list of int): which sequence(s) to use, None means using all sequences
        subject_idx (int or list of int): which subject(s) to use, None means using all subjects
    '''
    method = cfg['method']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    use_global_trans = cfg['data']['use_global_trans']
    use_aug = cfg['data']['use_aug']
    normalized_scale = cfg['data']['normalized_scale']
    input_type = cfg['data']['input_type']

    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
    }

    split = splits[mode]

    # Create dataset
    if dataset_type == 'cape':
        # Improved version with online point sampling
        input_pointcloud_n = cfg['data']['input_pointcloud_n']
        input_pointcloud_noise = cfg['data']['input_pointcloud_noise']
        points_subsample = cfg['data']['points_subsample']
        points_uniform_ratio = cfg['data']['points_uniform_ratio']
        num_joints = cfg['data']['num_joints']
        voxel_res = cfg['data']['voxel_res']
        use_abs_bone_transforms = cfg['data']['use_abs_bone_transforms']
        query_on_clothed = cfg['data']['query_on_clothed']
        double_layer = cfg['model']['decoder_kwargs'].get('double_layer', False)
        use_v_template = cfg['model']['decoder_kwargs'].get('pred_template', False)

        dataset = data.CAPEDataset(
            dataset_folder,
            subjects=split,
            mode=mode,
            use_aug=use_aug,
            use_v_template=use_v_template,
            double_layer=double_layer,
            num_joints=num_joints,
            input_pointcloud_n=input_pointcloud_n,
            input_pointcloud_noise=input_pointcloud_noise,
            points_size=points_subsample,
            points_uniform_ratio=points_uniform_ratio,
            use_global_trans=use_global_trans,
            normalized_scale=normalized_scale,
            use_abs_bone_transforms=use_abs_bone_transforms,
            query_on_clothed=query_on_clothed,
            input_type=input_type,
            voxel_res=voxel_res,
            sequence_idx=sequence_idx,
            subject_idx=subject_idx,
        )
    elif dataset_type == 'cape_sv':
        # Improved version with online point sampling
        input_pointcloud_n = cfg['data']['input_pointcloud_n']
        input_pointcloud_noise = cfg['data']['input_pointcloud_noise']
        points_subsample = cfg['data']['points_subsample']
        points_uniform_ratio = cfg['data']['points_uniform_ratio']
        num_joints = cfg['data']['num_joints']
        voxel_res = cfg['data']['voxel_res']
        use_abs_bone_transforms = cfg['data']['use_abs_bone_transforms']
        query_on_clothed = cfg['data']['query_on_clothed']
        double_layer = cfg['model']['decoder_kwargs'].get('double_layer', False)
        use_v_template = cfg['model']['decoder_kwargs'].get('pred_template', False)

        dataset = data.CAPESingleViewDataset(
            dataset_folder,
            subjects=split,
            mode=mode,
            use_aug=use_aug,
            use_v_template=use_v_template,
            double_layer=double_layer,
            num_joints=num_joints,
            input_pointcloud_n=input_pointcloud_n,
            input_pointcloud_noise=input_pointcloud_noise,
            points_size=points_subsample,
            points_uniform_ratio=points_uniform_ratio,
            use_global_trans=use_global_trans,
            normalized_scale=normalized_scale,
            use_abs_bone_transforms=use_abs_bone_transforms,
            query_on_clothed=query_on_clothed,
            input_type=input_type,
            voxel_res=voxel_res,
            sequence_idx=sequence_idx,
            subject_idx=subject_idx,
        )
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])

    return dataset
