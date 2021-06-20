import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import torch
import trimesh
import argparse
from tqdm import tqdm
import time
from collections import defaultdict
from collections import OrderedDict
import numpy as np
from im2mesh import config
from im2mesh.checkpoints import CheckpointIO

parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite already generated mesh or not')

parser.add_argument('--subject-idx', type=int, default=-1,
                    help='Which subject in the validation set to test')
parser.add_argument('--sequence-idx', type=int, default=-1,
                    help='Which sequence in the validation set to test')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = cfg['training']['out_dir']
generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
out_time_file = os.path.join(generation_dir, 'time_generation_full.pkl')
out_time_file_class = os.path.join(generation_dir, 'time_generation.pkl')

batch_size = cfg['generation']['batch_size']
input_type = cfg['data']['input_type']
vis_n_outputs = cfg['generation']['vis_n_outputs']
if vis_n_outputs is None:
    vis_n_outputs = -1

# Dataset
if cfg['data']['input_type'] == 'img':
    cfg['model']['encoder_kwargs'].update({'pretrained': ''})

if args.subject_idx >= 0 and args.sequence_idx >= 0:
    dataset = config.get_dataset('test', cfg, subject_idx=args.subject_idx, sequence_idx=args.sequence_idx)
else:
    dataset = config.get_dataset('test', cfg)

# Model
model = config.get_model(cfg, device=device, dataset=dataset)

checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])

# Generator
generator = config.get_generator(model, cfg, device=device)

# Determine what to generate
generate_mesh = cfg['generation']['generate_mesh']
generate_pointcloud = cfg['generation']['generate_pointcloud']

if generate_mesh and not hasattr(generator, 'generate_mesh'):
    generate_mesh = False
    print('Warning: generator does not support mesh generation.')

if generate_pointcloud and not hasattr(generator, 'generate_pointcloud'):
    generate_pointcloud = False
    print('Warning: generator does not support pointcloud generation.')


# Loader
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, num_workers=1, shuffle=False)

# Statistics
time_dicts = []

# Generate
model.eval()

# Count how many models already created
model_counter = defaultdict(int)

part_inds = list(range(22)) + [25, 40]   # SMPLH to SMPL

faces = np.load('body_models/misc/faces.npz')['faces']

for it, data in enumerate(tqdm(test_loader)):
    # Output folders
    mesh_dir = os.path.join(generation_dir, 'meshes')
    label_dir = os.path.join(generation_dir, 'labels')
    pointcloud_dir = os.path.join(generation_dir, 'pointcloud')
    in_dir = os.path.join(generation_dir, 'input')
    generation_vis_dir = os.path.join(generation_dir, 'vis', )

    # Get index etc.
    idx = data['idx'].item()
    model_dict = dataset.get_model_dict(idx)

    if input_type == 'pointcloud':
        subset = model_dict['subset']
        subject = model_dict['subject']
        sequence = model_dict['sequence']
        gender = model_dict['gender']
        data_path = model_dict['data_path']
        filebase = os.path.basename(data_path)[:-4]
    else:
        raise ValueError('Unknown input type: {}'.format(input_type))

    folder_name = os.path.join(subset, subject, sequence)
    generation_vis_dir = os.path.join(generation_vis_dir, folder_name)
    in_dir = os.path.join(in_dir, folder_name)

    mesh_dir = os.path.join(mesh_dir, folder_name)
    label_dir = os.path.join(label_dir, folder_name)

    # Create directories if necessary
    if vis_n_outputs >= 0 and not os.path.exists(generation_vis_dir):
        os.makedirs(generation_vis_dir)

    if generate_mesh and not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)

    if generate_mesh and not os.path.exists(label_dir):
        os.makedirs(label_dir)

    if generate_pointcloud and not os.path.exists(pointcloud_dir):
        os.makedirs(pointcloud_dir)

    if not os.path.exists(in_dir):
        os.makedirs(in_dir)

    # Timing dict
    time_dict = {
        'idx': idx,
        'subset': subset,
        'subject': subject,
        'sequence': sequence,
    }
    time_dicts.append(time_dict)

    # Generate outputs
    out_file_dict = {}

    c_it = model_counter[sequence]

    mesh_out_file = os.path.join(mesh_dir, filebase + '.minimal.posed.ply')

    if not args.overwrite and os.path.exists(mesh_out_file):
        continue

    if cfg['generation']['copy_input']:
        # Save inputs
        if input_type == 'pointcloud':
            inp = data['inputs'].squeeze(0).cpu().numpy()
            loc = data['points.loc'].squeeze(0).cpu().numpy()
            scale = data['points.scale'].squeeze(0).cpu().numpy()
            inp = inp * scale / 1.5 + loc

            pc_file_name = os.path.join(in_dir, filebase + '.input_pc.npy')
            np.save(pc_file_name, inp)
        else:
            raise ValueError('Supported input type: pointcloud, got {}'.format(input_type))

    if generate_mesh:
        t0 = time.time()
        out = generator.generate_mesh(data)
        time_dict['mesh'] = time.time() - t0

        # Get statistics
        try:
            mesh, stats_dict = out
        except TypeError:
            mesh, stats_dict = out, {}
        time_dict.update(stats_dict)
        # print ('Time for marching cubes: {:06f}'.format(time_dict['time (marching cubes)']))
        # print ('Time for eval points: {:06f}'.format(time_dict['time (eval points)']))

        # Write output
        if isinstance(mesh, dict):
            # Posed
            mesh_out_file = os.path.join(mesh_dir, filebase + '.minimal.posed.ply')
            mesh['minimal_posed'].export(mesh_out_file)
            mesh_out_file = os.path.join(mesh_dir, filebase + '.cloth.posed.ply')
            mesh['cloth_posed'].export(mesh_out_file)
            # Labels
            label_out_file = os.path.join(label_dir,  filebase + '.minimal.npz')
            part_labels = mesh['minimal_part_labels']
            np.savez(label_out_file,
                part_labels=mesh['minimal_part_labels'],
            )
            label_out_file = os.path.join(label_dir,  filebase + '.cloth.npz')
            part_labels = mesh['cloth_part_labels']
            np.savez(label_out_file,
                part_labels=mesh['cloth_part_labels'],
            )
            # Unposed
            if 'minimal_unposed' in mesh.keys() and 'cloth_unposed' in mesh.keys():
                mesh_out_file = os.path.join(mesh_dir, filebase + '.minimal.unposed.ply')
                mesh['minimal_unposed'].export(mesh_out_file)
                mesh_out_file = os.path.join(mesh_dir, filebase + '.cloth.unposed.ply')
                mesh['cloth_unposed'].export(mesh_out_file)
        else:
            raise ValueError('mesh must be a dict')

    model_counter[sequence] += 1
