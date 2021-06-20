"""
Code to fit SMPL (pose, shape) to IPNet predictions using pytorch, kaolin.
"""
import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import torch
import trimesh
import argparse
import numpy as np
import pickle as pkl
from kaolin.rep import TriangleMesh as tm
from kaolin.metrics.mesh import laplacian_loss
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial.transform import Rotation
from im2mesh.utils import icp

from human_body_prior.body_model.body_model import BodyModel
from lib.th_smpl_prior import get_prior
from lib.th_SMPL import th_batch_SMPL, th_batch_SMPL_split_params
from lib.mesh_distance import chamfer_distance, batch_point_to_surface
from im2mesh import config, data
from im2mesh.utils.logs import create_logger

SMPL2IPNET_IDX = np.array([11, 12, 13, 11, 3, 8, 11, 1, 6, 11, 1, 6, 0, 11, 11, 0, 5, 10, 4, 9, 2, 7, 2, 7], dtype=np.int64)

parser = argparse.ArgumentParser('Register SMPL meshes for NASA+PTFs predictions.')
parser.add_argument('config', type=str, help='Path to config file.')

parser.add_argument('--num-joints', type=int, default=14,
                    help='Number of joints to use for SMPL (14 for IPNet, 24 for NASA+PTFs).')
parser.add_argument('--subject-idx', type=int, default=-1,
                    help='Which subject in the validation set to test')
parser.add_argument('--sequence-idx', type=int, default=-1,
                    help='Which sequence in the validation set to test')
parser.add_argument('--use-raw-scan', action='store_true',
                    help='Whether to use raw scan to fit SMPLD')
parser.add_argument('--use-parts', action='store_true',
                    help='Whether to use part losses or not')
parser.add_argument('--init-pose', action='store_true', help='Whether to initialize pose or not. Only valid for PTFs.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

NUM_PARTS = 14  # number of parts that the smpl is segmented into.

smpl_faces = np.load('body_models/misc/faces.npz')['faces']

def backward_step(loss_dict, weight_dict, it):
    w_loss = dict()
    for k in loss_dict:
        w_loss[k] = weight_dict[k](loss_dict[k], it)

    tot_loss = list(w_loss.values())
    tot_loss = torch.stack(tot_loss).sum()
    return tot_loss

def get_loss_weights_SMPL():
    """Set loss weights"""

    loss_weight = {'s2m': lambda cst, it: 10. ** 2 * cst * (1 + it),
                   'm2s': lambda cst, it: 10. ** 2 * cst / (1 + it),
                   'betas': lambda cst, it: 10. ** 0 * cst / (1 + it),
                   'offsets': lambda cst, it: 10. ** -1 * cst / (1 + it),
                   'pose_pr': lambda cst, it: 10. ** -5 * cst / (1 + it),
                   'lap': lambda cst, it: cst / (1 + it),
                   'part': lambda cst, it: 10. ** 2 * cst / (1 + it)
                   }
    return loss_weight

def get_loss_weights_SMPLD():
    """Set loss weights"""

    loss_weight = {'s2m': lambda cst, it: 10. ** 2 * cst * (1 + it),
                   'm2s': lambda cst, it: 10. ** 2 * cst, #/ (1 + it),
                   'lap': lambda cst, it: 10. ** 4 * cst / (1 + it),
                   'offsets': lambda cst, it: 10. ** 1 * cst / (1 + it)}
    return loss_weight


def forward_step_SMPL(th_scan_meshes, smpl, scan_part_labels, smpl_part_labels, args):
    """
    Performs a forward step, given smpl and scan meshes.
    Then computes the losses.
    """
    # Get pose prior
    prior = get_prior(smpl.gender, precomputed=True)

    # forward
    verts, _, _, _ = smpl()
    th_smpl_meshes = [tm.from_tensors(vertices=v,
                                      faces=smpl.faces) for v in verts]

    scan_verts = [sm.vertices for sm in th_scan_meshes]
    smpl_verts = [sm.vertices for sm in th_smpl_meshes]

    # losses
    loss = dict()
    loss['s2m'] = batch_point_to_surface(scan_verts, th_smpl_meshes)
    loss['m2s'] = batch_point_to_surface(smpl_verts, th_scan_meshes)
    loss['betas'] = torch.mean(smpl.betas ** 2, axis=1)
    loss['pose_pr'] = prior(smpl.pose)

    # if args.num_joints == 14:
    if args.use_parts:
        loss['part'] = []
        for n, (sc_v, sc_l) in enumerate(zip(scan_verts, scan_part_labels)):
            tot = 0
            # for i in range(args.num_joints):  # we currently use 14 parts
            for i in range(14):  # we currently use 14 parts
                if i not in sc_l:
                    continue
                ind = torch.where(sc_l == i)[0]
                sc_part_points = sc_v[ind].unsqueeze(0)
                sm_part_points = smpl_verts[n][torch.where(smpl_part_labels[n] == i)[0]].unsqueeze(0)
                dist = chamfer_distance(sc_part_points, sm_part_points, w1=1., w2=1.)
                tot += dist
            # loss['part'].append(tot / args.num_joints)
            loss['part'].append(tot / 14)

        loss['part'] = torch.stack(loss['part'])

    return loss

def forward_step_SMPLD(th_scan_meshes, smpl, init_smpl_meshes, args):
    """
    Performs a forward step, given smpl and scan meshes.
    Then computes the losses.
    """

    # forward
    verts, _, _, _ = smpl()
    th_smpl_meshes = [tm.from_tensors(vertices=v,
                                      faces=smpl.faces) for v in verts]

    # losses
    loss = dict()
    loss['s2m'] = batch_point_to_surface([sm.vertices for sm in th_scan_meshes], th_smpl_meshes)
    loss['m2s'] = batch_point_to_surface([sm.vertices for sm in th_smpl_meshes], th_scan_meshes)
    loss['lap'] = torch.stack([laplacian_loss(sc, sm) for sc, sm in zip(init_smpl_meshes, th_smpl_meshes)])
    loss['offsets'] = torch.mean(torch.mean(smpl.offsets**2, axis=1), axis=1)
    return loss


def optimize_pose_shape(th_scan_meshes, smpl, iterations, steps_per_iter, scan_part_labels, smpl_part_labels,
                        display, args):
    """
    Optimize SMPL.
    :param display: if not None, pass index of the scan in th_scan_meshes to visualize.
    """
    # Optimizer
    optimizer = torch.optim.Adam([smpl.trans, smpl.betas, smpl.pose], 0.02, betas=(0.9, 0.999))

    # Get loss_weights
    weight_dict = get_loss_weights_SMPL()

    for it in range(iterations):
        loop = tqdm(range(steps_per_iter))
        loop.set_description('Optimizing SMPL')
        for i in loop:
            optimizer.zero_grad()
            # Get losses for a forward pass
            loss_dict = forward_step_SMPL(th_scan_meshes, smpl, scan_part_labels, smpl_part_labels, args)
            # Get total loss for backward pass
            tot_loss = backward_step(loss_dict, weight_dict, it)
            tot_loss.backward()
            optimizer.step()

            l_str = 'Iter: {}'.format(i)
            for k in loss_dict:
                l_str += ', {}: {:0.4f}'.format(k, weight_dict[k](loss_dict[k], it).mean().item())
                loop.set_description(l_str)

    print('** Optimised smpl pose and shape **')


def optimize_pose_only(th_scan_meshes, smpl, iterations, steps_per_iter, scan_part_labels, smpl_part_labels,
                       display, args):
    """
    Initially we want to only optimize the global rotation of SMPL. Next we optimize full pose.
    We optimize pose based on the 3D keypoints in th_pose_3d.
    :param  th_pose_3d: array containing the 3D keypoints.
    """

    batch_sz = smpl.pose.shape[0]
    split_smpl = th_batch_SMPL_split_params(batch_sz, top_betas=smpl.betas.data[:, :2],
                                            other_betas=smpl.betas.data[:, 2:],
                                            global_pose=smpl.pose.data[:, :3], other_pose=smpl.pose.data[:, 3:],
                                            faces=smpl.faces, gender=smpl.gender).to('cuda')
    optimizer = torch.optim.Adam([split_smpl.trans, split_smpl.top_betas, split_smpl.global_pose], 0.02,
                                 betas=(0.9, 0.999))

    # Get loss_weights
    weight_dict = get_loss_weights_SMPL()

    iter_for_global = 1
    for it in range(iter_for_global + iterations):
        loop = tqdm(range(steps_per_iter))
        if it < iter_for_global:
            # Optimize global orientation
            print('Optimizing SMPL global orientation')
            loop.set_description('Optimizing SMPL global orientation')
        elif it == iter_for_global:
            # Now optimize full SMPL pose
            print('Optimizing SMPL pose only')
            loop.set_description('Optimizing SMPL pose only')
            optimizer = torch.optim.Adam([split_smpl.trans, split_smpl.top_betas, split_smpl.global_pose,
                                          split_smpl.other_pose], 0.02, betas=(0.9, 0.999))
        else:
            loop.set_description('Optimizing SMPL pose only')

        for i in loop:
            optimizer.zero_grad()
            # Get losses for a forward pass
            loss_dict = forward_step_SMPL(th_scan_meshes, split_smpl, scan_part_labels, smpl_part_labels, args)
            # Get total loss for backward pass
            tot_loss = backward_step(loss_dict, weight_dict, it)
            tot_loss.backward()
            optimizer.step()

            l_str = 'Iter: {}'.format(i)
            for k in loss_dict:
                l_str += ', {}: {:0.4f}'.format(k, weight_dict[k](loss_dict[k], it).mean().item())
                loop.set_description(l_str)

    # Put back pose, shape and trans into original smpl
    smpl.pose.data = split_smpl.pose.data
    smpl.betas.data = split_smpl.betas.data
    smpl.trans.data = split_smpl.trans.data

    print('** Optimised smpl pose **')


def optimize_offsets(th_scan_meshes, smpl, init_smpl_meshes, iterations, steps_per_iter, args):
    # Optimizer
    optimizer = torch.optim.Adam([smpl.offsets, smpl.pose, smpl.trans, smpl.betas], 0.005, betas=(0.9, 0.999))

    # Get loss_weights
    weight_dict = get_loss_weights_SMPLD()

    for it in range(iterations):
        loop = tqdm(range(steps_per_iter))
        loop.set_description('Optimizing SMPL+D')
        for i in loop:
            optimizer.zero_grad()
            # Get losses for a forward pass
            loss_dict = forward_step_SMPLD(th_scan_meshes, smpl, init_smpl_meshes, args)
            # Get total loss for backward pass
            tot_loss = backward_step(loss_dict, weight_dict, it)
            tot_loss.backward()
            optimizer.step()

            l_str = 'Lx100. Iter: {}'.format(i)
            for k in loss_dict:
                l_str += ', {}: {:0.4f}'.format(k, loss_dict[k].mean().item()*100)
            loop.set_description(l_str)


def compute_poses(all_posed_vertices, all_unposed_vertices, all_labels, parents, args):
    all_thetas = []

    for posed_vertices, unposed_vertices, labels in zip(all_posed_vertices, all_unposed_vertices, all_labels):
        labels = labels.detach().cpu().numpy()
        bone_transforms_ransac = []
        for j_idx in range(0, args.num_joints):
            v_posed = posed_vertices[labels == j_idx, :]
            v_unposed = unposed_vertices[labels == j_idx, :]
            if v_unposed.shape[0] < 6:
                if j_idx == 0:
                    bone_transform_ransac = np.eye(4).astype(np.float32)
                else:
                    bone_transform_ransac = bone_transforms_ransac[parents[j_idx]].copy()
            else:
                bone_transform_ransac = icp.estimate_rigid_transform_3D(v_unposed, v_posed, 500, 0.7, 0.005)

            bone_transforms_ransac.append(bone_transform_ransac)

        # Now, factor out rotations that are relative to parents
        Rs = [bone_transforms_ransac[0][:3, :3].copy()]
        for j_idx in range(1, args.num_joints):
            R = bone_transforms_ransac[j_idx][:3, :3].copy()
            Rp = bone_transforms_ransac[parents[j_idx]][:3, :3].copy()
            R = np.dot(np.linalg.inv(Rp), R)
            Rs.append(R)

        # Convert to anxis-angle representation
        thetas = np.concatenate([Rotation.from_matrix(R).as_rotvec() for R in Rs], axis=-1)
        all_thetas.append(thetas)

    poses = np.stack(all_thetas, axis=0)

    return poses

def SMPLD_register(args):
    cfg = config.load_config(args.config, 'configs/default.yaml')
    out_dir = cfg['training']['out_dir']
    generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")

    if args.subject_idx >= 0 and args.sequence_idx >= 0:
        logger, _ = create_logger(generation_dir, phase='reg_subject{}_sequence{}'.format(args.subject_idx, args.sequence_idx), create_tf_logs=False)
    else:
        logger, _ = create_logger(generation_dir, phase='reg_all', create_tf_logs=False)

    # Get dataset
    if args.subject_idx >= 0 and args.sequence_idx >= 0:
        dataset = config.get_dataset('test', cfg, sequence_idx=args.sequence_idx, subject_idx=args.subject_idx)
    else:
        dataset = config.get_dataset('test', cfg)

    batch_size = cfg['generation']['batch_size']

    # Loader
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=1, shuffle=False)

    model_counter = defaultdict(int)

    # Set optimization hyper parameters
    iterations, pose_iterations, steps_per_iter, pose_steps_per_iter = 3, 2, 30, 30

    inner_dists = []
    outer_dists = []

    for it, data in enumerate(tqdm(test_loader)):
        idxs = data['idx'].cpu().numpy()
        loc = data['points.loc'].cpu().numpy()
        batch_size = idxs.shape[0]
        # Directories to load corresponding informations
        mesh_dir = os.path.join(generation_dir, 'meshes')   # directory for posed and (optionally) unposed implicit outer/inner meshes
        label_dir = os.path.join(generation_dir, 'labels')   # directory for part labels
        register_dir = os.path.join(generation_dir, 'registrations')   # directory for part labels

        if args.use_raw_scan:
            scan_dir = dataset.dataset_folder   # this is the folder that contains CAPE raw scans
        else:
            scan_dir = None

        all_posed_minimal_meshes = []
        all_posed_cloth_meshes = []
        all_posed_vertices = []
        all_unposed_vertices = []
        scan_part_labels = []

        for idx in idxs:
            model_dict = dataset.get_model_dict(idx)

            subset = model_dict['subset']
            subject = model_dict['subject']
            sequence = model_dict['sequence']
            gender = model_dict['gender']
            filebase = os.path.basename(model_dict['data_path'])[:-4]

            folder_name = os.path.join(subset, subject, sequence)
            # TODO: we assume batch size stays the same if one resumes the job
            # can be more flexible to support different batch sizes before and
            # after resume
            register_file = os.path.join(register_dir, folder_name, filebase + 'minimal.registered.ply')
            if os.path.exists(register_file):
                # batch already computed, break
                break

            # points_dict = np.load(model_dict['data_path'])
            # gender = str(points_dict['gender'])

            mesh_dir_ = os.path.join(mesh_dir, folder_name)
            label_dir_ = os.path.join(label_dir, folder_name)

            if scan_dir is not None:
                scan_dir_ = os.path.join(scan_dir, subject, sequence)

            # Load part labels and vertex translations
            label_file_name = filebase + '.minimal.npz'
            label_dict = dict(np.load(os.path.join(label_dir_, label_file_name)))
            labels = torch.tensor(label_dict['part_labels'].astype(np.int64)).to(device)   # part labels for each vertex (14 or 24)
            scan_part_labels.append(labels)

            # Load minimal implicit surfaces
            mesh_file_name = filebase + '.minimal.posed.ply'
            # posed_mesh = Mesh(filename=os.path.join(mesh_dir_, mesh_file_name))
            posed_mesh = trimesh.load(os.path.join(mesh_dir_, mesh_file_name), process=False)
            posed_vertices = np.array(posed_mesh.vertices)
            all_posed_vertices.append(posed_vertices)

            posed_mesh = tm.from_tensors(torch.tensor(posed_mesh.vertices.astype('float32'), requires_grad=False, device=device),
                    torch.tensor(posed_mesh.faces.astype('int64'), requires_grad=False, device=device))
            all_posed_minimal_meshes.append(posed_mesh)

            mesh_file_name = filebase + '.minimal.unposed.ply'
            if os.path.exists(os.path.join(mesh_dir_, mesh_file_name)) and args.init_pose:
                # unposed_mesh = Mesh(filename=os.path.join(mesh_dir_, mesh_file_name))
                unposed_mesh = trimesh.load(os.path.join(mesh_dir_, mesh_file_name), process=False)
                unposed_vertices = np.array(unposed_mesh.vertices)
                all_unposed_vertices.append(unposed_vertices)

            if args.use_raw_scan:
                # Load raw scans
                mesh_file_name = filebase + '.ply'
                # posed_mesh = Mesh(filename=os.path.join(scan_dir_, mesh_file_name))
                posed_mesh = trimesh.load(os.path.join(scan_dir_, mesh_file_name), process=False)

                posed_mesh = tm.from_tensors(torch.tensor(posed_mesh.vertices.astype('float32') / 1000, requires_grad=False, device=device),
                        torch.tensor(posed_mesh.faces.astype('int64'), requires_grad=False, device=device))
                all_posed_cloth_meshes.append(posed_mesh)
            else:
                # Load clothed implicit surfaces
                mesh_file_name = filebase + '.cloth.posed.ply'
                # posed_mesh = Mesh(filename=os.path.join(mesh_dir_, mesh_file_name))
                posed_mesh = trimesh.load(os.path.join(mesh_dir_, mesh_file_name), process=False)

                posed_mesh = tm.from_tensors(torch.tensor(posed_mesh.vertices.astype('float32'), requires_grad=False, device=device),
                        torch.tensor(posed_mesh.faces.astype('int64'), requires_grad=False, device=device))
                all_posed_cloth_meshes.append(posed_mesh)

        if args.num_joints == 24:
            bm = BodyModel(bm_path='body_models/smpl/male/model.pkl', num_betas=10, batch_size=batch_size).to(device)
            parents = bm.kintree_table[0].detach().cpu().numpy()
            labels = bm.weights.argmax(1)
            # Convert 24 parts to 14 parts
            smpl2ipnet = torch.from_numpy(SMPL2IPNET_IDX).to(device)
            labels = smpl2ipnet[labels].clone().unsqueeze(0)
            del bm
        elif args.num_joints == 14:
            with open('body_models/misc/smpl_parts_dense.pkl', 'rb') as f:
                part_labels = pkl.load(f)

            labels = np.zeros((6890,), dtype=np.int64)
            for n, k in enumerate(part_labels):
                labels[part_labels[k]] = n
            labels = torch.tensor(labels).to(device).unsqueeze(0)
        else:
            raise ValueError('Got {} joints but umber of joints can only be either 14 or 24'.format(args.num_joints))

        th_faces = torch.tensor(smpl_faces.astype('float32'), dtype=torch.long).to(device)

        # We assume loaded meshes are properly scaled and offsetted to the orignal SMPL space,
        if len(all_posed_minimal_meshes) > 0 and len(all_unposed_vertices) == 0:
            # IPNet optimization without vertex traslation
            # raise NotImplementedError('Optimization for IPNet is not implemented yet.')
            if args.num_joints == 24:
                for idx in range(len(scan_part_labels)):
                    scan_part_labels[idx] = smpl2ipnet[scan_part_labels[idx]].clone()

            prior = get_prior(gender=gender, precomputed=True)
            pose_init = torch.zeros((batch_size, 72))
            pose_init[:, 3:] = prior.mean
            betas, pose, trans = torch.zeros((batch_size, 10)), pose_init, torch.zeros((batch_size, 3))

            # Init SMPL, pose with mean smpl pose, as in ch.registration
            smpl = th_batch_SMPL(batch_size, betas, pose, trans, faces=th_faces, gender=gender).to(device)
            smpl_part_labels = torch.cat([labels] * batch_size, axis=0)

            # Optimize pose first
            optimize_pose_only(all_posed_minimal_meshes, smpl, pose_iterations, pose_steps_per_iter, scan_part_labels,
                               smpl_part_labels, None, args)

            # Optimize pose and shape
            optimize_pose_shape(all_posed_minimal_meshes, smpl, iterations, steps_per_iter, scan_part_labels, smpl_part_labels,
                                None, args)

            inner_vertices, _, _, _ = smpl()

            # Optimize vertices for SMPLD
            init_smpl_meshes = [tm.from_tensors(vertices=v.clone().detach(),
                                                faces=smpl.faces) for v in inner_vertices]
            optimize_offsets(all_posed_cloth_meshes, smpl, init_smpl_meshes, 5, 10, args)

            outer_vertices, _, _, _ = smpl()
        elif len(all_posed_minimal_meshes) > 0:
            # NASA+PTFs optimization with vertex traslations
            # Compute poses from implicit surfaces and correspondences
            # TODO: we could also compute bone-lengths if we train PTFs to predict A-pose with a global translation
            # that equals to the centroid of the pointcloud
            poses = compute_poses(all_posed_vertices, all_unposed_vertices, scan_part_labels, parents, args)
            # Convert 24 parts to 14 parts
            for idx in range(len(scan_part_labels)):
                scan_part_labels[idx] = smpl2ipnet[scan_part_labels[idx]].clone()

            pose_init = torch.from_numpy(poses).float()
            betas, pose, trans = torch.zeros((batch_size, 10)), pose_init, torch.zeros((batch_size, 3))

            # Init SMPL, pose with mean smpl pose, as in ch.registration
            smpl = th_batch_SMPL(batch_size, betas, pose, trans, faces=th_faces, gender=gender).to(device)
            smpl_part_labels = torch.cat([labels] * batch_size, axis=0)

            # Optimize pose first
            optimize_pose_only(all_posed_minimal_meshes, smpl, pose_iterations, pose_steps_per_iter, scan_part_labels,
                               smpl_part_labels, None, args)

            # Optimize pose and shape
            optimize_pose_shape(all_posed_minimal_meshes, smpl, iterations, steps_per_iter, scan_part_labels, smpl_part_labels,
                                None, args)

            inner_vertices, _, _, _ = smpl()

            # Optimize vertices for SMPLD
            init_smpl_meshes = [tm.from_tensors(vertices=v.clone().detach(),
                                                faces=smpl.faces) for v in inner_vertices]
            optimize_offsets(all_posed_cloth_meshes, smpl, init_smpl_meshes, 5, 10, args)

            outer_vertices, _, _, _ = smpl()
        else:
            inner_vertices = outer_vertices = None

        if args.use_raw_scan:
            for i, idx in enumerate(idxs):
                model_dict = dataset.get_model_dict(idx)

                subset = model_dict['subset']
                subject = model_dict['subject']
                sequence = model_dict['sequence']
                filebase = os.path.basename(model_dict['data_path'])[:-4]

                folder_name = os.path.join(subset, subject, sequence)
                register_dir_ = os.path.join(register_dir, folder_name)
                if not os.path.exists(register_dir_):
                    os.makedirs(register_dir_)

                if not os.path.exists(os.path.join(register_dir_, filebase + 'minimal.registered.ply')):
                    registered_mesh = trimesh.Trimesh(inner_vertices[i].detach().cpu().numpy().astype(np.float64), smpl_faces, process=False)
                    registered_mesh.export(os.path.join(register_dir_, filebase + 'minimal.registered.ply'))

                if not os.path.exists(os.path.join(register_dir_, filebase + 'cloth.registered.ply')):
                    registered_mesh = trimesh.Trimesh(outer_vertices[i].detach().cpu().numpy().astype(np.float64), smpl_faces, process=False)
                    registered_mesh.export(os.path.join(register_dir_, filebase + 'cloth.registered.ply'))
        else:
            # Evaluate registered mesh
            gt_smpl_mesh = data['points.minimal_smpl_vertices'].to(device)
            gt_smpld_mesh = data['points.smpl_vertices'].to(device)
            if inner_vertices is None:
                # if vertices are None, we assume they already exist due to previous runs
                inner_vertices = []
                outer_vertices = []
                for i, idx in enumerate(idxs):

                    model_dict = dataset.get_model_dict(idx)

                    subset = model_dict['subset']
                    subject = model_dict['subject']
                    sequence = model_dict['sequence']
                    filebase = os.path.basename(model_dict['data_path'])[:-4]

                    folder_name = os.path.join(subset, subject, sequence)
                    register_dir_ = os.path.join(register_dir, folder_name)

                    # registered_mesh = Mesh(filename=os.path.join(register_dir_, filebase + 'minimal.registered.ply'))
                    registered_mesh = trimesh.load(os.path.join(register_dir_, filebase + 'minimal.registered.ply'), process=False)
                    registered_v = torch.tensor(registered_mesh.vertices.astype(np.float32), requires_grad=False, device=device)
                    inner_vertices.append(registered_v)

                    # registered_mesh = Mesh(filename=os.path.join(register_dir_, filebase + 'cloth.registered.ply'))
                    registered_mesh = trimesh.load(os.path.join(register_dir_, filebase + 'cloth.registered.ply'), process=False)
                    registered_v = torch.tensor(registered_mesh.vertices.astype(np.float32), requires_grad=False, device=device)
                    outer_vertices.append(registered_v)

                inner_vertices = torch.stack(inner_vertices, dim=0)
                outer_vertices = torch.stack(outer_vertices, dim=0)

            inner_dist = torch.norm(gt_smpl_mesh - inner_vertices, dim=2).mean(-1)
            outer_dist = torch.norm(gt_smpld_mesh - outer_vertices, dim=2).mean(-1)

            for i, idx in enumerate(idxs):
                model_dict = dataset.get_model_dict(idx)

                subset = model_dict['subset']
                subject = model_dict['subject']
                sequence = model_dict['sequence']
                filebase = os.path.basename(model_dict['data_path'])[:-4]

                folder_name = os.path.join(subset, subject, sequence)
                register_dir_ = os.path.join(register_dir, folder_name)
                if not os.path.exists(register_dir_):
                    os.makedirs(register_dir_)

                logger.info('Inner distance for input {}: {} cm'.format(filebase, inner_dist[i].item()))
                logger.info('Outer distance for input {}: {} cm'.format(filebase, outer_dist[i].item()))

                if not os.path.exists(os.path.join(register_dir_, filebase + 'minimal.registered.ply')):
                    registered_mesh = trimesh.Trimesh(inner_vertices[i].detach().cpu().numpy().astype(np.float64), smpl_faces, process=False)
                    registered_mesh.export(os.path.join(register_dir_, filebase + 'minimal.registered.ply'))

                if not os.path.exists(os.path.join(register_dir_, filebase + 'cloth.registered.ply')):
                    registered_mesh = trimesh.Trimesh(outer_vertices[i].detach().cpu().numpy().astype(np.float64), smpl_faces, process=False)
                    registered_mesh.export(os.path.join(register_dir_, filebase + 'cloth.registered.ply'))

            inner_dists.extend(inner_dist.detach().cpu().numpy())
            outer_dists.extend(outer_dist.detach().cpu().numpy())

    logger.info('Mean inner distance: {} cm'.format(np.mean(inner_dists)))
    logger.info('Mean outer distance: {} cm'.format(np.mean(outer_dists)))


def main(args):
    SMPLD_register(args)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
