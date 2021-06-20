import os
import glob
import numpy as np
import torch
import yaml
import trimesh
import numbers

import pickle as pkl

from torch.utils import data

from scipy.spatial import cKDTree as KDTree
from scipy.spatial.transform import Rotation as R
# from human_body_prior.mesh import MeshViewer
from im2mesh.utils.libmesh import check_mesh_contains

SMPL2IPNET_IDX = np.array([11, 12, 13, 11, 3, 8, 11, 1, 6, 11, 1, 6, 0, 11, 11, 0, 5, 10, 4, 9, 2, 7, 2, 7])
IPNET2SMPL_IDX = np.array([12, 7, 20, 4, 18, 16, 8, 21, 5, 19, 17, 0, 1, 2])
SMPL_parents = np.array([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
                    16, 17, 18, 19, 20, 21], dtype=np.int32)
IPNet_parents = np.array([11, 3, 4, 12, 5, 11, 8, 9, 13, 10, 11, -1, 11, 11], dtype=np.int32)
IPNet_parents_in_SMPL = np.array([9, 4, 18, 1, 16, 13, 5, 19, 2, 17, 14, -1, 0, 0], dtype=np.int32)

class CAPEDataset(data.Dataset):
    ''' CAPE dataset class.
    '''

    def __init__(self, dataset_folder,
                 subjects=['00032', '00096', '00122', '00127', '00134', '00145', '00159', '00215', '02474', '03223', '03284', '03331', '03375', '03383', '03394'],
                 mode='train',
                 input_type='pointcloud',
                 voxel_res=128,
                 double_layer=False,
                 use_aug=False,
                 use_v_template=False,
                 use_abs_bone_transforms=False,
                 num_joints=24,
                 input_pointcloud_n=5000,
                 input_pointcloud_noise=0.001,
                 points_size=2048,
                 points_uniform_ratio=0.5,
                 use_global_trans=False,
                 normalized_scale=False,
                 points_sigma=0.05,
                 query_on_clothed=False,
                 sequence_idx=None,
                 subject_idx=None):
        ''' Initialization of the the CAPE dataset.

        Args:
            dataset_folder (str): dataset folder
            subjects (list of strs): list of subjects to use
            mode (str): can be either 'train', 'val' or 'test'
            input_type (str): only pointcloud is supported for now
            voxel_res (int): voxel resolution, currently deprecated
            double_layer (bool): use double layer prediction for cloth
            use_aug (bool): use data augmentation or not
            use_v_template (bool): also extract correspondences on the mean-shape template
            use_abs_bone_transforms (bool): use absolute bone transforms to get A-pose points
                as was done in the original NASA
            num_joints (int): number of joints of the articulated body
            input_pointcloud_n (int): number of points to sample for encoder inputs
            input_pointcloud_noise (float): noise for input pointcloud
            points_size (int): number of occupancy query points to sample
            points_uniform_ratio (float): ratio of uniformly sampled points in query points
            use_global_trans (bool): subtract GT global translation to query points in the A-pose
            normalized_scale (bool): normalize all points into [-1, 1]
            points_sigma (float): standard deviation for sampling query points around mesh surfaces
            query_on_clothed (bool): use registered clothed surface to compute GT skinning weights
                of query points
            sequence_idx (int): use only the speficied sequence. If None, use all sequences
            subject_idx (int): use only the speficied subject. If None, use all subjects
        '''
        # Attributes
        self.cape_path = '/cluster/home/shawang/Datasets/CAPE'
        self.dataset_folder = dataset_folder
        self.subjects = subjects
        self.use_global_trans = use_global_trans
        self.use_aug = use_aug
        self.mode = mode
        self.normalized_scale = normalized_scale
        self.input_type = input_type
        self.voxel_res = voxel_res
        self.double_layer = double_layer
        self.num_joints = num_joints
        self.query_on_clothed = query_on_clothed
        self.use_abs_bone_transforms = use_abs_bone_transforms

        # if normalized_scale:
        #     assert ( not use_global_trans )

        self.points_size = points_size if self.mode in ['train', 'test'] else 100000
        # self.points_padding = 0.1
        self.points_padding = 1 / 3    # For IPNet, mesh is normalized to [-0.75, 0.75] while sampling space is [-1, 1]
        self.points_uniform_ratio = points_uniform_ratio if self.mode in ['train', 'test'] else 0.5
        self.points_sigma = points_sigma    # 5cm standard deviation for surface points

        if input_type == 'pointcloud':
            self.input_pointcloud_n = input_pointcloud_n
            self.input_pointcloud_noise = input_pointcloud_noise
        else:
            self.input_pointcloud_n = self.input_pointcloud_noise = 0

        self.faces = np.load('body_models/misc/faces.npz')['faces']
        self.skinning_weights = dict(np.load('body_models/misc/skinning_weights_all.npz'))
        self.posedirs = dict(np.load('body_models/misc/posedirs_all.npz'))
        if self.use_abs_bone_transforms:
            self.J_regressors = dict(np.load('body_models/misc/J_regressors.npz'))

        with open('body_models/misc/smpl_parts_dense.pkl', 'rb') as f:
            part_labels = pkl.load(f)

        labels = np.zeros(6890, dtype=np.int32)
        for idx, k in enumerate(part_labels):
            labels[part_labels[k]] = idx

        self.part_labels = labels

        self.use_v_template = use_v_template
        if use_v_template:
            self.v_templates = dict(np.load('body_models/misc/v_templates.npz'))

        # Get all data
        self.data = []
        if subject_idx is not None:
            subjects = [subjects[subject_idx]]

        with open(os.path.join(self.cape_path, 'cape_release/misc/subj_genders.pkl'), 'rb') as f:
            genders = pkl.load(f)

        for subject in subjects:
            subject_dir = os.path.join(dataset_folder, subject)
            sequence_dirs = glob.glob(os.path.join(subject_dir, '*'))
            sequences = set()
            for sequence_dir in sequence_dirs:
                sequences.add(os.path.basename(sequence_dir).split('.')[0])

            sequences = sorted(list(sequences))

            if sequence_idx is not None:
                sequences = [sequences[sequence_idx]]

            for sequence in sequences:
                points_dir = os.path.join(subject_dir, sequence)

                points_files = sorted(glob.glob(os.path.join(points_dir, '*.npz')))

                self.data += [
                        {'subset': 'cape',
                         'subject': subject,
                         'gender': genders[subject],
                         'sequence': sequence,
                         'data_path': points_file}
                        for points_file in points_files
                    ]

    def augm_params(self):
        """Get augmentation parameters."""
        if self.mode == 'train' and self.use_aug:
            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            # Roll
            rot_x = min(2*90,
                    max(-2*90, np.random.randn()*90))

            sn, cs = np.sin(np.pi / 180 * rot_x), np.cos(np.pi / 180 * rot_x)
            rot_x = np.eye(4)
            rot_x[1, 1] = cs
            rot_x[1, 2] = -sn
            rot_x[2, 1] = sn
            rot_x[2, 2] = cs

            rot_y = min(2*90,
                    max(-2*90, np.random.randn()*90))

            # Pitch
            sn, cs = np.sin(np.pi / 180 * rot_y), np.cos(np.pi / 180 * rot_y)
            rot_y = np.eye(4)
            rot_y[0, 0] = cs
            rot_y[0, 2] = sn
            rot_y[2, 0] = -sn
            rot_y[2, 2] = cs

            rot_z = min(2*90,
                    max(-2*90, np.random.randn()*90))

            # Yaw
            sn, cs = np.sin(np.pi / 180 * rot_z), np.cos(np.pi / 180 * rot_z)
            rot_z = np.eye(4)
            rot_z[0, 0] = cs
            rot_z[0, 1] = -sn
            rot_z[1, 0] = sn
            rot_z[1, 1] = cs

            rot_mat = np.dot(rot_x, np.dot(rot_y, rot_z))

            # but it is identity with probability 3/5
            if np.random.uniform() <= 0.6:
                rot_mat = np.eye(4)

        else:
            rot_mat = np.eye(4)


        return rot_mat

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.data)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        data_path = self.data[idx]['data_path']
        subject = self.data[idx]['subject']
        gender = self.data[idx]['gender']
        data = {}

        aug_rot = self.augm_params().astype(np.float32)

        points_dict = np.load(data_path)

        # 3D models and points
        loc = points_dict['loc'].astype(np.float32)
        trans = points_dict['trans'].astype(np.float32)
        root_loc = points_dict['Jtr'][0].astype(np.float32)
        scale = points_dict['scale'].astype(np.float32)

        # Also get GT SMPL poses
        pose_body = points_dict['pose_body']
        pose_hand = points_dict['pose_hand']
        pose = np.concatenate([pose_body, pose_hand], axis=-1)
        pose = R.from_rotvec(pose.reshape([-1, 3]))

        body_mesh_a_pose = points_dict['a_pose_mesh_points']
        # Break symmetry if given in float16:
        if body_mesh_a_pose.dtype == np.float16:
            body_mesh_a_pose = body_mesh_a_pose.astype(np.float32)
            body_mesh_a_pose += 1e-4 * np.random.randn(*body_mesh_a_pose.shape)
        else:
            body_mesh_a_pose = body_mesh_a_pose.astype(np.float32)

        n_smpl_points = body_mesh_a_pose.shape[0]

        bone_transforms = points_dict['bone_transforms'].astype(np.float32)
        # Apply rotation augmentation to bone transformations
        bone_transforms_aug = np.matmul(np.expand_dims(aug_rot, axis=0), bone_transforms)
        bone_transforms_aug[:, :3, -1] += root_loc - trans - np.dot(aug_rot[:3, :3], root_loc - trans)
        bone_transforms = bone_transforms_aug
        # Get augmented posed-mesh
        skinning_weights = self.skinning_weights[gender]
        if self.use_abs_bone_transforms:
            J_regressor = self.J_regressors[gender]

        T = np.dot(skinning_weights, bone_transforms.reshape([-1, 16])).reshape([-1, 4, 4])

        homogen_coord = np.ones([n_smpl_points, 1], dtype=np.float32)
        a_pose_homo = np.concatenate([body_mesh_a_pose - trans, homogen_coord], axis=-1).reshape([n_smpl_points, 4, 1])
        body_mesh = np.matmul(T, a_pose_homo)[:, :3, 0].astype(np.float32) + trans

        # Get extents of model.
        bb_min = np.min(body_mesh, axis=0)
        bb_max = np.max(body_mesh, axis=0)
        # total_size = np.sqrt(np.square(bb_max - bb_min).sum())
        total_size = (bb_max - bb_min).max()
        # Scales all dimensions equally.
        scale = max(1.6, total_size)    # 1.6 is the magic number from IPNet
        loc = np.array(
            [(bb_min[0] + bb_max[0]) / 2,
             (bb_min[1] + bb_max[1]) / 2,
             (bb_min[2] + bb_max[2]) / 2],
            dtype=np.float32
        )

        posed_trimesh = trimesh.Trimesh(vertices=body_mesh, faces=self.faces)
        # a_pose_trimesh = trimesh.Trimesh(vertices=(body_mesh_a_pose - trans) * 1.0 / scale * 1.5, faces=self.faces)

        n_points_uniform = int(self.points_size * self.points_uniform_ratio)
        n_points_surface = self.points_size - n_points_uniform

        boxsize = 1 + self.points_padding
        points_uniform = np.random.rand(n_points_uniform, 3)
        points_uniform = boxsize * (points_uniform - 0.5)
        # Scale points in (padded) unit box back to the original space
        points_uniform *= scale
        points_uniform += loc
        # Sample points around posed-mesh surface
        n_points_surface_cloth = n_points_surface // 2 if self.double_layer else n_points_surface
        points_surface = posed_trimesh.sample(n_points_surface_cloth + self.input_pointcloud_n)
        if self.input_type == 'pointcloud':
            input_pointcloud = points_surface[n_points_surface_cloth:]
            noise = self.input_pointcloud_noise * np.random.randn(*input_pointcloud.shape)
            input_pointcloud = (input_pointcloud + noise).astype(np.float32)

        points_surface = points_surface[:n_points_surface_cloth]
        points_surface += np.random.normal(scale=self.points_sigma, size=points_surface.shape)

        if self.double_layer:
            n_points_surface_minimal = n_points_surface // 2

            posedir = self.posedirs[gender]
            minimal_shape_path = os.path.join(self.cape_path, 'cape_release', 'minimal_body_shape', subject, subject + '_minimal.npy')
            minimal_shape = np.load(minimal_shape_path)
            pose_mat = pose.as_matrix()
            ident = np.eye(3)
            pose_feature = (pose_mat - ident).reshape([207, 1])
            pose_offsets = np.dot(posedir.reshape([-1, 207]), pose_feature).reshape([6890, 3])
            minimal_shape += pose_offsets

            if self.use_abs_bone_transforms:
                Jtr_cano = np.dot(J_regressor, minimal_shape)
                Jtr_cano = Jtr_cano[IPNET2SMPL_IDX, :]

            a_pose_homo = np.concatenate([minimal_shape, homogen_coord], axis=-1).reshape([n_smpl_points, 4, 1])
            minimal_body_mesh = np.matmul(T, a_pose_homo)[:, :3, 0].astype(np.float32) + trans
            minimal_posed_trimesh = trimesh.Trimesh(vertices=minimal_body_mesh, faces=self.faces)

            # Sample points around minimally clothed posed-mesh surface
            points_surface_minimal = minimal_posed_trimesh.sample(n_points_surface_minimal)
            points_surface_minimal += np.random.normal(scale=self.points_sigma, size=points_surface_minimal.shape)

            points_surface = np.vstack([points_surface, points_surface_minimal])

        # Check occupancy values for sampled ponits
        query_points = np.vstack([points_uniform, points_surface]).astype(np.float32)
        if self.double_layer:
            # Double-layer occupancies, as was done in IPNet
            # 0: outside, 1: between body and cloth, 2: inside body mesh
            occupancies_cloth = check_mesh_contains(posed_trimesh, query_points)
            occupancies_minimal = check_mesh_contains(minimal_posed_trimesh, query_points)
            occupancies = occupancies_cloth.astype(np.int64)
            occupancies[occupancies_minimal] = 2
        else:
            occupancies = check_mesh_contains(posed_trimesh, query_points).astype(np.float32)

        # Skinning inds by querying nearest SMPL vertex on the clohted mesh
        kdtree = KDTree(body_mesh if self.query_on_clothed else minimal_body_mesh)
        _, p_idx = kdtree.query(query_points)
        pts_W = skinning_weights[p_idx, :]
        skinning_inds_ipnet = self.part_labels[p_idx] # skinning inds (14 parts)
        skinning_inds_smpl = pts_W.argmax(1)   # full skinning inds (24 parts)
        if self.num_joints == 14:
            skinning_inds = skinning_inds_ipnet
        else:
            skinning_inds = skinning_inds_smpl

        # Invert LBS to get query points in A-pose space
        T = np.dot(pts_W, bone_transforms.reshape([-1, 16])).reshape([-1, 4, 4])
        T = np.linalg.inv(T)

        homogen_coord = np.ones([self.points_size, 1], dtype=np.float32)
        posed_homo = np.concatenate([query_points - trans, homogen_coord], axis=-1).reshape([self.points_size, 4, 1])
        query_points_a_pose = np.matmul(T, posed_homo)[:, :3, 0].astype(np.float32) + trans

        if self.use_abs_bone_transforms:
            assert (not self.use_v_template and self.num_joints == 24)
            query_points_a_pose -= Jtr_cano[SMPL2IPNET_IDX[skinning_inds], :]

        if self.use_v_template:
            v_template = self.v_templates[gender]
            pose_shape_offsets = v_template - minimal_shape
            query_points_template = query_points_a_pose + pose_shape_offsets[p_idx, :]

        sc_factor = 1.0 / scale * 1.5 if self.normalized_scale else 1.0 # 1.5 is the magic number from IPNet
        offset = loc

        bone_transforms_inv = bone_transforms.copy()
        bone_transforms_inv[:, :3, -1] += trans - loc
        bone_transforms_inv = np.linalg.inv(bone_transforms_inv)
        bone_transforms_inv[:, :3, -1] *= sc_factor

        data = {
            None: (query_points - offset) * sc_factor,
            'occ': occupancies,
            'trans': trans,
            'root_loc': root_loc,
            'pts_a_pose': (query_points_a_pose - (trans if self.use_global_trans else offset)) * sc_factor,
            'skinning_inds': skinning_inds,
            'skinning_inds_ipnet': skinning_inds_ipnet,
            'skinning_inds_smpl': skinning_inds_smpl,
            'loc': loc,
            'scale': scale,
            'bone_transforms': bone_transforms,
            'bone_transforms_inv': bone_transforms_inv,
        }

        if self.use_v_template:
            data.update({'pts_template': (query_points_template - (trans if self.use_global_trans else offset)) * sc_factor})

        if self.mode in ['test']:
            data.update({'smpl_vertices': body_mesh, 'smpl_a_pose_vertices': body_mesh_a_pose})
            if self.double_layer:
                data.update({'minimal_smpl_vertices': minimal_body_mesh})

        data_out = {}
        field_name = 'points' if self.mode in ['train', 'test'] else 'points_iou'
        for k, v in data.items():
            if k is None:
                data_out[field_name] = v
            else:
                data_out['%s.%s' % (field_name, k)] = v

        if self.input_type == 'pointcloud':
            data_out.update(
                {'inputs': (input_pointcloud - offset) * sc_factor,
                 'idx': idx,
                }
            )
        elif self.input_type == 'voxel':
            voxels = np.unpackbits(points_dict['voxels_occ']).astype(np.float32)
            voxels = np.reshape(voxels, [self.voxel_res] * 3)
            data_out.update(
                {'inputs': voxels,
                 'idx': idx,
                }
            )
        else:
            raise ValueError('Unsupported input type: {}'.format(self.input_type))

        return data_out

    def get_model_dict(self, idx):
        return self.data[idx]
