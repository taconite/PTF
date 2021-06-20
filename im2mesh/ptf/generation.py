import torch
import torch.optim as optim
from torch import autograd
import numpy as np
from tqdm import trange
import trimesh
import cv2
from im2mesh.ptf import models
from im2mesh.utils import libmcubes
from im2mesh.common import make_3d_grid
from im2mesh.utils.libsimplify import simplify_mesh
from im2mesh.utils.libmise import MISE
import time
import numbers

def replace_infs(x):
    if (x == float("Inf")).all():
        x[:] = 1e6
    elif (x == float("Inf")).any():
        x[x == float("Inf")] = x[x != float("Inf")].max()

    if (x == float("-Inf")).all():
        x[:] = -1e6
    elif (x == float("-Inf")).any():
        x[x == float("-Inf")] = x[x != float("-Inf")].min()

    return x

def replace_nans(x):
    x[torch.isnan(x)] = 1e6
    return x

class Generator3D(object):
    '''  Generator class for Occupancy Networks.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for iso-surface extraction
        sample (bool): whether z should be sampled
        simplify_nfaces (int): number of faces the mesh should be simplified to
        double_layer (bool): use double layer occupancy
    '''

    def __init__(self, model, points_batch_size=100000,
                 threshold=0.5, refinement_step=0, device=None,
                 resolution0=16, upsampling_steps=3, num_joints=24,
                 with_normals=False, padding=1.0, sample=False,
                 simplify_nfaces=None, input_type='pointcloud',
                 double_layer=False):
        self.model = model.to(device)
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.threshold = threshold
        self.device = device
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.num_joints = num_joints
        self.with_normals = with_normals
        self.padding = padding
        self.sample = sample
        self.simplify_nfaces = simplify_nfaces
        self.input_type = input_type
        self.double_layer = double_layer

        if not double_layer:
            raise NotImplementedError('We currently do not support iso-suface extraction for single-layer models')

        self.colors = np.load('body_models/misc/part_colors.npz')['colors']

        self.model_type = self.model.model_type
        self.tf_type = self.model.tf_type

    def generate_mesh(self, data, return_stats=True):
        ''' Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}

        # For generation, batch_size is always 1
        bone_transforms = None

        loc = data.get('points.loc').to(device)
        # root_loc = data.get('points.root_loc').to(device)
        # trans = data.get('points.trans').to(device)
        scale = data.get('points.scale').to(device)
        # bone_transforms = data.get('points.bone_transforms').to(device)
        inputs = data.get('inputs').to(device)

        # kwargs = {'scale': scale, 'bone_transforms': bone_transforms, 'trans': trans, 'loc': loc, 'root_loc': root_loc}
        kwargs = {'scale': scale, 'loc': loc}

        # Encode inputs
        t0 = time.time()
        with torch.no_grad():
            c = self.model.encode_inputs(inputs, **kwargs)

        stats_dict['time (encode inputs)'] = time.time() - t0

        mesh_all = {}
        # Note that for all current models, we use upsampling_steps==0.
        # If upsampling_steps > 0, we have to extract inner and outer
        # surfaces separately, this cancels out the benefit of MISE.
        # Besides, MISE is greedy and thus compromises surface accuracy.
        if self.upsampling_steps > 0:
            raise ValueError('We do not support MISE for double layer')
        else:
            mesh_all = self.generate_from_conditional(c, stats_dict=stats_dict, **kwargs)

        if return_stats:
            return mesh_all, stats_dict
        else:
            return mesh_all

    def generate_from_conditional(self, c, stats_dict={}, **kwargs):
        ''' Generates mesh from latent.

        Args:
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding

        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (nx,)*3
            ).to(self.device)

            out_dict = self.eval_points(pointsf, c, **kwargs)
            if self.tf_type is None:
                value_minimal_grid = out_dict['values_minimal'].detach().cpu().numpy().reshape([nx, nx, nx])
                value_cloth_grid = out_dict['values_cloth'].detach().cpu().numpy().reshape([nx, nx, nx])
                label_grid = out_dict['labels'].detach().cpu().numpy().reshape([nx, nx, nx])
                p_hat_grid = None
            else:
                value_minimal_grid = out_dict['values_minimal'].detach().cpu().numpy().reshape([nx, nx, nx])
                value_cloth_grid = out_dict['values_cloth'].detach().cpu().numpy().reshape([nx, nx, nx])
                label_grid = out_dict['labels'].detach().cpu().numpy().reshape([nx, nx, nx])
                p_hat_grid = out_dict['p_hats'].detach().cpu().numpy().reshape([nx, nx, nx, 3])

        else:
            raise ValueError('We do not support MISE for double layer')

        # Extract mesh
        stats_dict['time (eval points)'] = time.time() - t0

        mesh_all = {}
        # Generate body under cloth
        mesh_minimal = self.extract_mesh(value_minimal_grid, label_grid, p_hat_grid, c, stats_dict=stats_dict, **kwargs)
        for k, v in mesh_minimal.items():
            mesh_all['minimal_' + k] = v

        # Generate body with cloth
        mesh_cloth = self.extract_mesh(value_cloth_grid, label_grid, p_hat_grid, c, stats_dict=stats_dict, **kwargs)
        for k, v in mesh_cloth.items():
            mesh_all['cloth_' + k] = v

        return mesh_all

    def eval_points(self, p, c, **kwargs):
        ''' Evaluates the occupancy values for the points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''
        p_split = torch.split(p, self.points_batch_size)
        occ_minimal_hats = []
        occ_cloth_hats = []
        label_hats = []
        p_hats = []

        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                ci = self.model.get_point_features(pi, c=c, **kwargs)

                if self.tf_type is not None:
                    out_dict_tf = self.model.transform_points(pi, z=None, c=ci, **kwargs)
                    p_hat = out_dict_tf['p_hat']
                    if 'parts_softmax' in out_dict_tf.keys():
                        parts_softmax = out_dict_tf['parts_softmax']
                        kwargs.update({'parts_softmax': parts_softmax})
                else:
                    p_hat = pi

                out_dict = self.model.decode(p_hat, z=None, c=ci, **kwargs)
                if self.tf_type is not None:
                    out_dict.update(out_dict_tf)

                occ_logits = out_dict['logits']
                if 'out_cls' in out_dict.keys():
                    out_cls = out_dict['out_cls']
                else:
                    out_cls = None

                if 'parts_softmax' in out_dict.keys():
                    parts_softmax = out_dict['parts_softmax']
                elif out_cls is not None:
                    parts_softmax = torch.nn.functional.softmax(out_cls, dim=1)
                else:
                    parts_softmax = None

                # Compute translated points
                if self.tf_type is not None:
                    if parts_softmax is None:
                        part_logits = torch.max(occ_logits, dim=2)[0] if self.double_layer else occ_logits
                        p_hat = (p_hat.view(1, self.num_joints, 3, -1)
                                * torch.nn.functional.softmax(part_logits, dim=1).view(1, self.num_joints, 1, -1)).sum(1)
                    else:
                        p_hat = p_hat.view(1, self.num_joints, 3, -1)
                        p_hat = (p_hat * parts_softmax.view(1, self.num_joints, 1, -1)).sum(1)

                    p_hats.append(p_hat.squeeze(0).transpose(0, 1))

                # Compute part label logits
                if self.double_layer:
                    if out_cls is None:
                        label_logits = torch.max(occ_logits, dim=2)[0]
                        label_hat = label_logits.argmax(1)
                    else:
                        label_hat = out_cls.argmax(1)
                else:
                    raise NotImplementedError('We currently do not support iso-suface extraction for single-layer models')

                # Compute occupancy values
                if self.double_layer:
                    if len(occ_logits.shape) > 3:
                        softmax_logits = torch.max(occ_logits, dim=1)[0]
                    else:
                        softmax_logits = occ_logits
                    # else:
                    #     raise ValueError('Model type {} does not support double layer prediction'.format(self.model_type))

                    softmax_logits = torch.nn.functional.softmax(softmax_logits, dim=1)
                    cloth_occ_hat = 1. / (softmax_logits[:, 1, :] + softmax_logits[:, 2, :]) - 1
                    minimal_occ_hat = 1. / softmax_logits[:, 2, :] - 1
                    cloth_occ_hat = -1. * torch.log(torch.max(cloth_occ_hat, torch.zeros_like(cloth_occ_hat)))
                    minimal_occ_hat = -1. * torch.log(torch.max(minimal_occ_hat, torch.zeros_like(minimal_occ_hat)))
                    # occ_hat = torch.stack([minimal_occ_hat, cloth_occ_hat], dim=-1)

                    cloth_occ_hat = replace_infs(cloth_occ_hat)
                    minimal_occ_hat = replace_infs(minimal_occ_hat)
                else:
                    raise NotImplementedError('We currently do not support iso-suface extraction for single-layer models')

            occ_minimal_hats.append(minimal_occ_hat.squeeze(0).detach().cpu())
            occ_cloth_hats.append(cloth_occ_hat.squeeze(0).detach().cpu())
            label_hats.append(label_hat.squeeze(0).detach().cpu())

        occ_minimal_hat = torch.cat(occ_minimal_hats, dim=0)
        occ_cloth_hat = torch.cat(occ_cloth_hats, dim=0)
        label_hat = torch.cat(label_hats, dim=0)
        if self.tf_type is not None:
            p_hat = torch.cat(p_hats, dim=0)

            return {'values_minimal': occ_minimal_hat, 'values_cloth': occ_cloth_hat, 'labels': label_hat, 'p_hats': p_hat}
        else:
            return {'values_minimal': occ_minimal_hat, 'values_cloth': occ_cloth_hat, 'labels': label_hat}

    def extract_mesh(self, occ_hat, label_hat, p_hat, c, stats_dict=dict(), **kwargs):
        ''' Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            label_hat (tensor): value grid of predicted part labels
            p_hat (tensor): value grid of predicted locations in the A-pose
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)
        stats_dict['time (marching cubes)'] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # Undo padding
        vertices -= 1
        # Construct part labels and A-pose vertices by sampling
        # occupancy grid and translation grid
        r_verts = np.round(vertices).astype('int32')
        labels = label_hat[r_verts[:, 0], r_verts[:, 1], r_verts[:, 2]]
        colors = self.colors[labels]
        # Normalize to bounding box
        vertices /= np.array([n_x-1, n_y-1, n_z-1])
        if p_hat is not None:
            with torch.no_grad():
                v = torch.from_numpy(vertices).to(self.device).float()
                v = v * 2 - 1   # range: [-1, 1]
                v = v.unsqueeze(0).unsqueeze(1).unsqueeze(1)    # 1 x 1 x 1 x n_pts x 3
                p_hat = torch.from_numpy(p_hat).to(self.device).float()
                p_hat = p_hat.permute(3, 2, 1, 0).unsqueeze(0)   # 1 X C x D x H x W
                # v_rest is in [-1, 1]
                v_rest = torch.nn.functional.grid_sample(p_hat, v, align_corners=True)
                v_rest = v_rest.squeeze(0).squeeze(1).squeeze(1).transpose(0, 1) / 1.5 * kwargs['scale'] # + kwargs['loc']

            vertices_rest = v_rest.detach().cpu().numpy()
        else:
            vertices_rest = None

        # vertices = box_size * (vertices - 0.5)
        vertices = 4 / 3 * kwargs['scale'].item() * (vertices - 0.5) + kwargs['loc'].cpu().numpy()

        # Estimate normals if needed
        if self.with_normals and not vertices.shape[0] == 0:
            t0 = time.time()
            normals = self.estimate_normals(vertices, c)
            stats_dict['time (normals)'] = time.time() - t0

        else:
            normals = None

        # Create mesh
        mesh = {}
        mesh['part_labels'] = labels
        mesh['posed'] = trimesh.Trimesh(vertices, triangles,
                                        vertex_normals=normals,
                                        vertex_colors=colors,
                                        process=False)
        if vertices_rest is not None:
            mesh['unposed'] = trimesh.Trimesh(vertices_rest, triangles,
                                              vertex_normals=normals,
                                              vertex_colors=colors,
                                              process=False)

        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # TODO: normals are lost here
        if self.simplify_nfaces is not None:
            t0 = time.time()
            mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)
            stats_dict['time (simplify)'] = time.time() - t0

        # Refine mesh
        if self.refinement_step > 0:
            t0 = time.time()
            self.refine_mesh(mesh, occ_hat, c)
            stats_dict['time (refine)'] = time.time() - t0

        return mesh

    def estimate_normals(self, vertices, c):
        ''' Estimates the normals by computing the gradient of the objective.

        Args:
            vertices (numpy array): vertices of the mesh
            c (tensor): latent conditioned code c
        '''
        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)

        normals = []
        c = c.unsqueeze(0)
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(device)
            vi.requires_grad_()
            occ_hat = self.model(vi, None, c)
            out = occ_hat.sum()
            out.backward()
            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals

    def refine_mesh(self, mesh, occ_hat, c):
        ''' Refines the predicted mesh.

        Args:
            mesh (trimesh object): predicted mesh
            occ_hat (tensor): predicted occupancy grid
            c (tensor): latent conditioned code c
        '''

        self.model.eval()

        # Some shorthands
        n_x, n_y, n_z = occ_hat.shape
        assert(n_x == n_y == n_z)
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        threshold = self.threshold

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(self.device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces).to(self.device)

        # Start optimization
        optimizer = optim.RMSprop([v], lr=1e-4)

        for it_r in trange(self.refinement_step):
            optimizer.zero_grad()

            # Loss
            face_vertex = v[faces]
            eps = np.random.dirichlet((0.5, 0.5, 0.5), size=faces.shape[0])
            eps = torch.FloatTensor(eps).to(self.device)
            face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

            face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
            face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
            face_normal = torch.cross(face_v1, face_v2)
            face_normal = face_normal / \
                (face_normal.norm(dim=1, keepdim=True) + 1e-10)
            face_value = torch.sigmoid(
                self.decode(face_point.unsqueeze(0), c)
            )
            normal_target = -autograd.grad(
                [face_value.sum()], [face_point], create_graph=True)[0]

            normal_target = \
                normal_target / \
                (normal_target.norm(dim=1, keepdim=True) + 1e-10)
            loss_target = (face_value - threshold).pow(2).mean()
            loss_normal = \
                (face_normal - normal_target).pow(2).sum(dim=1).mean()

            loss = loss_target + 0.01 * loss_normal

            # Update
            loss.backward()
            optimizer.step()

        mesh.vertices = v.data.cpu().numpy()

        return mesh
