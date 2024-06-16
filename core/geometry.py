'''
    Classes for the detector geometry
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scipy.spatial import KDTree
from sklearn.decomposition import PCA
import binpacking

from tqdm import tqdm


class DetectorModule:

    def __init__(self, module: pd.Series):

        self.features = {key: float(value) for key, value in module.items()}

        self.volume_id = 0

        if 'volume_id' in self.features.keys():
            self.volume_id = self.features['volume_id']

    def vertices_uvw(self, mode: str = 'quiet'):
        '''
            Finds the faces (vertex coordinates of the faces) of the module in uvw coordinates
        '''

        center = [0, 0, 0]  # in u, v, w coordinates

        min_width_u = self.features['module_minhu']
        max_width_u = self.features['module_maxhu']
        width_v = self.features['module_hv']
        width_w = self.features['module_t']

        vertices = [
            [center[0] - min_width_u, center[1] - width_v, center[2] - width_w],
            [center[0] + min_width_u, center[1] - width_v, center[2] - width_w],
            [center[0] + max_width_u, center[1] + width_v, center[2] - width_w],
            [center[0] - max_width_u, center[1] + width_v, center[2] - width_w],
            [center[0] - min_width_u, center[1] - width_v, center[2] + width_w],
            [center[0] + min_width_u, center[1] - width_v, center[2] + width_w],
            [center[0] + max_width_u, center[1] + width_v, center[2] + width_w],
            [center[0] - max_width_u, center[1] + width_v, center[2] + width_w]
        ]

        self.features['vertices_uvw'] = vertices
        if mode == 'verbose':
            print('vertices_uvw: ', self.features['vertices_uvw'])

    def vertices_xyz(self, mode: str = 'quiet'):
        '''
            Finds the faces (vertex coordinates of the faces) of the module in xyz coordinates
        '''

        self.vertices_uvw()
        self.features['vertices_xyz'] = [[0, 0, 0]
                                         for vertex in self.features['vertices_uvw']]

        offset = [
            self.features['cx'],
            self.features['cy'],
            self.features['cz']
        ]

        rotation_matrix = [
            [self.features['rot_xu'], self.features['rot_xv'], self.features['rot_xw']],
            [self.features['rot_yu'], self.features['rot_yv'], self.features['rot_yw']],
            [self.features['rot_zu'], self.features['rot_zv'], self.features['rot_zw']]
        ]

        for vtx_xyz, vtx_uvw in zip(self.features['vertices_xyz'], self.features['vertices_uvw']):
            vtx_xyz[0] = vtx_uvw[0] * rotation_matrix[0][0] + vtx_uvw[1] * \
                rotation_matrix[0][1] + vtx_uvw[2] * \
                rotation_matrix[0][2] + offset[0]
            vtx_xyz[1] = vtx_uvw[0] * rotation_matrix[1][0] + vtx_uvw[1] * \
                rotation_matrix[1][1] + vtx_uvw[2] * \
                rotation_matrix[1][2] + offset[1]
            vtx_xyz[2] = vtx_uvw[0] * rotation_matrix[2][0] + vtx_uvw[1] * \
                rotation_matrix[2][1] + vtx_uvw[2] * \
                rotation_matrix[2][2] + offset[2]

        if mode == 'verbose':
            print('vertices_xyz: ', self.features['vertices_xyz'])

    def module_faces(self, vertices):
        '''
            Finds the faces (vertex coordinates of the faces) of the module in xyz coordinates

            vertices: list of vertices of the module
        '''

        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]]
        ]

        return faces

    def draw_uvw(self, ax: plt.Axes, mode: str = 'quiet', **kwargs):
        '''
            Draws the module in 3D
        '''
        self.vertices_uvw(mode)
        self.features['faces_uvw'] = self.module_faces(
            self.features['vertices_uvw'])
        ax.add_collection3d(Poly3DCollection(
            self.features['faces_uvw'], **kwargs))

    def draw_xyz(self, ax: plt.Axes, mode: str = 'quiet', **kwargs):
        '''
            Draws the module in 3D
        '''
        self.vertices_xyz(mode)
        self.features['faces_xyz'] = self.module_faces(
            self.features['vertices_xyz'])
        ax.add_collection3d(Poly3DCollection(
            self.features['faces_xyz'], **kwargs))

    def draw_rz(self, ax: plt.Axes, mode: str = 'quiet', **kwargs):
        '''
            Draws the module in 2D (rz plane)
        '''

        self.vertices_xyz()
        vertices_rz = []
        for vertex in self.features['vertices_xyz']:
            r = (vertex[0]**2 + vertex[1]**2)**0.5
            z = vertex[2]

            # add the vertex to the list if it is significantly different from the others
            new_vertex = True
            if len(vertices_rz) > 0:
                for [rp, zp] in vertices_rz:
                    if abs(rp - r) < 0.5 and abs(zp - z) < 0.5:
                        new_vertex = False
                        break

            if new_vertex:
                vertices_rz.append([r, z])

        self.features['vertices_rz'] = vertices_rz
        if mode == 'verbose':
            print('vertices_rz: ', self.features['vertices_rz'])

        # Draw the module as a rectangle in the rz plane (2d)
        for i in range(len(vertices_rz)):
            if i < len(vertices_rz) - 1:
                ax.plot([vertices_rz[i][1], vertices_rz[i+1][1]],
                        [vertices_rz[i][0], vertices_rz[i+1][0]], **kwargs)
            else:
                ax.plot([vertices_rz[i][1], vertices_rz[0][1]], [
                        vertices_rz[i][0], vertices_rz[0][0]], **kwargs)

    def draw_xy(self, ax: plt.Axes, mode: str = 'quiet', **kwargs):
        '''
            Draws the module in 2D (xy plane)
        '''

        self.vertices_xyz()
        vertices_xy = []
        for vertex in self.features['vertices_xyz']:
            x = vertex[0]
            y = vertex[1]

            # add the vertex to the list if it is significantly different from the others
            new_vertex = True
            if len(vertices_xy) > 0:
                for [xp, yp] in vertices_xy:
                    if abs(xp - x) < 0.5 and abs(yp - y) < 0.5:
                        new_vertex = False
                        break

            if new_vertex:
                vertices_xy.append([x, y])

        self.features['vertices_xy'] = vertices_xy
        if mode == 'verbose':
            print('vertices_xy: ', self.features['vertices_xy'])

        # Draw the module as a rectangle in the xy plane (2d)
        for i in range(len(vertices_xy)):
            if i < len(vertices_xy) - 1:
                ax.plot([vertices_xy[i][0], vertices_xy[i+1][0]],
                        [vertices_xy[i][1], vertices_xy[i+1][1]], **kwargs)
            else:
                ax.plot([vertices_xy[i][0], vertices_xy[0][0]], [
                        vertices_xy[i][1], vertices_xy[0][1]], **kwargs)


class DetectorGeometry:

    def __init__(self, inFile: str):

        if not inFile.endswith('.csv'):
            raise ValueError('The input file must be a csv file')

        self.detector = pd.read_csv(inFile, sep=' ')

        # string to select the inner barrel modules
        self.inner_barrel_sel = 'volume_id==8'
        # string to select the forward layer modules
        self.forward_layer_sel = 'layer_id==12 and (volume_id==14 or volume_id==18)'

        self.volume_colors = {7: 'b', 8: 'b', 9: 'b',
                              12: 'r', 13: 'r', 14: 'r',
                              16: 'g', 17: 'g', 18: 'g'
                              }

    def detector_display(self, ax: plt.Axes, mode: str = 'quiet', **kwargs):
        '''
            Draws the detector in 3D
        '''

        for imodule in self.detector.iterrows():
            module = DetectorModule(imodule[1])
            module.draw_xyz(
                ax, mode=mode, facecolors=self.volume_colors[module.volume_id], **kwargs)

        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')
        ax.set_xlim(-1000, 1000)
        ax.set_ylim(-1000, 1000)
        ax.set_zlim(-4000, 4000)

    def inner_barrel_display(self, ax: plt.Axes, mode: str = 'quiet', **kwargs):
        '''
            Draws the inner barrel modules in 3D
        '''

        for imodule in self.detector.query(self.inner_barrel_sel, inplace=False).iterrows():
            module = DetectorModule(imodule[1])
            module.draw_xyz(
                ax, mode=mode, facecolors=self.volume_colors[module.volume_id], **kwargs)

        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')
        ax.set_xlim(-200, 200)
        ax.set_ylim(-200, 200)
        ax.set_zlim(-600, 600)

    def forward_layer_display(self, ax: plt.Axes, mode: str = 'quiet', **kwargs):
        '''
            Draws the forward layer modules in 3D
        '''

        for imodule in self.detector.query(self.forward_layer_sel, inplace=False).iterrows():
            module = DetectorModule(imodule[1])
            module.draw_xy(
                ax, mode=mode, facecolors=self.volume_colors[module.volume_id], **kwargs)

        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_xlim(-1100, 1100)
        ax.set_ylim(-1100, 1100)

    def radial_display(self, ax: plt.Axes, mode: str = 'quiet', **kwargs):
        '''
            Draws the detector in the rz plane
        '''

        for imodule in self.detector.iterrows():
            module = DetectorModule(imodule[1])
            module.draw_rz(
                ax, mode=mode, facecolors=self.volume_colors[module.volume_id], **kwargs)

        ax.set_xlabel('z (mm)')
        ax.set_ylabel('r (mm)')
        ax.set_xlim(-4000, 4000)
        ax.set_ylim(0, 1100)


class DetectorHashTable:

    def __init__(self, detector_file: str):
        '''
            Class to create a hashtable for the detector modules
        '''

        self.detector_dataset = pd.read_csv(detector_file)
        self.hashtable = None
        self._detector_hashtable()

    def _detector_hashtable(self) -> None:
        '''
            Creates a hashtable for the detector modules
        '''

        self.hashtable = {}
        unique_id = 0
        for idx, row in self.detector_dataset.iterrows():
            row_dict = {key: float(value) for key, value in row.items()}
            key = (row_dict['volume_id'],
                   row_dict['layer_id'], row_dict['module_id'])
            if key not in self.hashtable:
                self.hashtable[key] = unique_id
                unique_id += 1

    def add_unique_module_id_column(self, input_dataset):
        '''
            Adds a unique module id column to the detector dataset
        '''

        input_dataset['unique_module_id'] = input_dataset.apply(lambda row: self.hashtable[(
            row['volume_id'], row['layer_id'], row['module_id'])], axis=1)
        return input_dataset


class VoxelGrid:

    def __init__(self, detector_file: str, grid_size_x: int, grid_size_y: int, grid_size_z: int, n_submodules: int):
        '''
            Class to create a voxel grid for the detector.
            Used to create the input data for the convolutional neural network.

            For the geometry used in the kaggle competition TrackML, a grid size of 25x25x40 
            captures each module in a single voxel.
            For a more detailed grid, modules can be divided into smaller voxels.
        '''

        self.grid_size = np.array((grid_size_x, grid_size_y, grid_size_z), dtype=int)
        self.voxel_grid = np.zeros(self.grid_size, dtype=np.float32)
        self.detector_hashtable = DetectorHashTable(detector_file)
        self.module_grid_positions = None   # position of the modules in the voxel grid

        self.detector_dataset = pd.read_csv(detector_file, sep=',')
        self._create_submodule_df(n_submodules)
        self._calculate_normalized_positions()
        #self._generate_adaptive_voxel_grid()
        #self._generate_z_order_voxel_grid()
        #self._generate_bin_packing_voxel_grid()
        #self._generate_pca_voxel_grid()
        self._use_normalized_positions()

        # create a KDTree for the submodule positions (efficient search for the submodule closest to a hit)
        self.submodule_kdtree = KDTree(
            self.submodule_dataset[['cx', 'cy', 'cz']].to_numpy())

        # self.detector_dataset = self.add_unique_module_id_column(self.detector_dataset)
        # self._compute_module_normalized_positions()

    # MODULE SEGMENTATION (DEPRECATED)

    def add_unique_module_id_column(self, input_dataset):
        '''
            Adds a unique module id column to the detector dataset
        '''

        input_dataset = self.detector_hashtable.add_unique_module_id_column(
            input_dataset)
        return input_dataset

    def _compute_module_normalized_positions(self) -> None:
        '''
            Computes the normalized positions of the modules. 
            These are used for the voxel grid.
        '''

        # check if the column 'unique_module_id' exists
        if 'unique_module_id' not in self.detector_dataset.columns:
            self.detector_dataset = self.add_unique_module_id_column(
                self.detector_dataset)

        transformed_positions = []
        for module in self.detector_dataset.iterrows():
            # Apply rotation and translation to the module center
            module = module[1]
            offset = np.array([module['cx'], module['cy'], module['cz']])
            rotation_matrix = np.array([
                [module['rot_xu'], module['rot_xv'], module['rot_xw']],
                [module['rot_yu'], module['rot_yv'], module['rot_yw']],
                [module['rot_zu'], module['rot_zv'], module['rot_zw']]
            ])
            # Module center in (u, v, w) coordinates
            module_center_uvw = np.array([0, 0, 0])
            module_center_xyz = rotation_matrix.dot(module_center_uvw) + offset
            transformed_positions.append(module_center_xyz)

            del offset, rotation_matrix, module_center_uvw, module_center_xyz

        # Convert to numpy array for easy manipulation
        transformed_positions = np.array(transformed_positions)

        # Determine the bounds and the scale factor
        min_bounds = np.min(transformed_positions, axis=0)
        max_bounds = np.max(transformed_positions, axis=0)
        scale = (self.grid_size - 1) / (max_bounds - min_bounds)

        # Normalize module positions to fit within the voxel grid
        self.module_grid_positions = {}
        for module, position in zip(self.detector_dataset.iterrows(), transformed_positions):
            module = module[1]
            normalized_x = int((position[0] - min_bounds[0]) * scale)
            normalized_y = int((position[1] - min_bounds[1]) * scale)
            normalized_z = int((position[2] - min_bounds[2]) * scale)
            self.module_grid_positions[module['unique_module_id']] = (
                normalized_x, normalized_y, normalized_z)

        del transformed_positions, min_bounds, max_bounds, scale

    # SUBMODULE SEGMENTATION - Creating the grid
    def _segment_module(self, module, n: int):
        '''
            Segments a module into n submodules. Since the modules are centered at (0, 0, 0) in uvw coordinates,
            the submodule position in xyz coordinates is just the submodule offset.
        '''

        submodules = []
        min_width_u = module['module_minhu']
        width_v = module['module_hv']

        submodule_width_u = 2 * min_width_u / n
        submodule_width_v = 2 * width_v / n

        u_positions = np.linspace(-min_width_u + submodule_width_u / 2, min_width_u - submodule_width_u / 2, n)
        v_positions = np.linspace(-width_v + submodule_width_v / 2, width_v - submodule_width_v / 2, n)

        rotation_matrix = np.array([
            [module['rot_xu'], module['rot_xv'], module['rot_xw']],
            [module['rot_yu'], module['rot_yv'], module['rot_yw']],
            [module['rot_zu'], module['rot_zv'], module['rot_zw']]
        ])

        for u in u_positions:
            for v in v_positions:

                submodule_offset_uvw = np.array([u, v, 0])
                submodule_additional_offset_xyz = rotation_matrix.dot(
                    submodule_offset_uvw) + np.array([module['cx'], module['cy'], module['cz']])

                submodule = module.copy()
                submodule['cx'] += submodule_additional_offset_xyz[0]
                submodule['cy'] += submodule_additional_offset_xyz[1]

                submodules.append(submodule)

        return pd.DataFrame(submodules)

    def _create_submodule_df(self, n: int) -> None:
        '''
            Creates a dataframe for detectr submodules and a 
        '''

        print('Creating submodule dataset...')

        self.submodule_dataset = pd.DataFrame()

        for _, module in self.detector_dataset.iterrows():
            submodules = self._segment_module(module, n)
            self.submodule_dataset = pd.concat(
                [self.submodule_dataset, submodules], ignore_index=True)

        # create a 'submodule_id' column containing the module index
        self.submodule_dataset['submodule_id'] = self.submodule_dataset.index

    def _calculate_bounding_box(self):
        '''
            Calculates the bounding box of the voxel grid
        '''

        print('Calculating bounding box...')

        min_x = self.submodule_dataset['cx'].min()
        min_y = self.submodule_dataset['cy'].min()
        min_z = self.submodule_dataset['cz'].min()

        max_x = self.submodule_dataset['cx'].max()
        max_y = self.submodule_dataset['cy'].max()
        max_z = self.submodule_dataset['cz'].max()

        return min_x, min_y, min_z, max_x, max_y, max_z

    def _calculate_normalized_positions(self):
        '''
            Calculates the normalized positions of the submodules
        '''

        print('Calculating normalized positions...')

        min_x, min_y, min_z, max_x, max_y, max_z = self._calculate_bounding_box()

        scale_x = (self.grid_size[0] - 1) / (max_x - min_x)
        scale_y = (self.grid_size[1] - 1) / (max_y - min_y)
        scale_z = (self.grid_size[2] - 1) / (max_z - min_z)

        self.submodule_dataset['normalized_x'] = (
            (self.submodule_dataset['cx'] - min_x) * scale_x).astype(int)
        self.submodule_dataset['normalized_y'] = (
            (self.submodule_dataset['cy'] - min_y) * scale_y).astype(int)
        self.submodule_dataset['normalized_z'] = (
            (self.submodule_dataset['cz'] - min_z) * scale_z).astype(int)

    # def _generate_adaptive_voxel_grid(self):
    #    '''
    #        Generates an adaptive voxel grid for the detector (avoid overlap and waste of empty space)
    #    '''
    #
    #    print('Generating adaptive voxel grid...')
    #
    #    position_map = defaultdict(int)
    #    heap = []
    #    max_cx, max_cy, max_cz = 0, 0, 0
    #
    #    for isubmodule, submodule in tqdm(self.submodule_dataset.iterrows()):
    #        pos = (submodule['normalized_x'], submodule['normalized_y'], submodule['normalized_z'])
    #        if pos not in position_map:
    #            heapq.heappush(heap, (0, pos))
    #
    #        while heap:
    #            pos = heapq.heappop(heap)[1]
    #            if pos not in position_map:
    #                position_map[pos] = 1
    #                self.submodule_dataset.at[isubmodule, 'normalized_x'], self.submodule_dataset.at[isubmodule, 'normalized_y'], self.submodule_dataset.at[isubmodule, 'normalized_z'] = pos
    #                max_cx = max(max_cx, pos[0])
    #                max_cy = max(max_cy, pos[1])
    #                max_cz = max(max_cz, pos[2])
    #                break
    #
    #            # Add surrounding positions to the heap
    #            for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
    #                new_pos = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
    #                if 0 <= new_pos[0] < self.grid_size[0] and 0 <= new_pos[1] < self.grid_size[1] and 0 <= new_pos[2] < self.grid_size[2]:
    #                    heapq.heappush(heap, (np.linalg.norm(new_pos), new_pos))
    #
    #    self.grid_size = np.array([max_cx + 1, max_cy + 1, max_cz + 1], dtype=int)

    def _use_normalized_positions(self):
        '''
            Uses the normalized positions of the submodules for the voxel grid
        '''

        self.submodule_dataset['grid_x'] = self.submodule_dataset['normalized_x']
        self.submodule_dataset['grid_y'] = self.submodule_dataset['normalized_y']
        self.submodule_dataset['grid_z'] = self.submodule_dataset['normalized_z']

    def _interleave_bits(self, x, y, z):
        '''
            Interleaves the bits of x, y and z to calculate the Z-order value
        '''

        x = int(x)
        y = int(y)
        z = int(z)

        result = 0
        for i in range(32):
            result |= ((x & (1 << i)) << 2*i) | ((y & (1 << i)) << (2*i + 1)) | ((z & (1 << i)) << (2*i + 2))

        return result

    def _generate_z_order_voxel_grid(self):
        '''
            Generates a voxel grid for the detector using a Z-order curve
        '''

        print('Generating Z-order voxel grid...')

        # Normalize the positions
        self._calculate_normalized_positions()

        # Calculate the Z-order values
        self.submodule_dataset['z_order'] = self.submodule_dataset.apply(lambda row: self._interleave_bits(row['normalized_x'], row['normalized_y'], row['normalized_z']), axis=1)

        # Sort the modules by their Z-order values
        self.submodule_dataset.sort_values('z_order', inplace=True)

        # Assign each module to a different grid point
        self.submodule_dataset['grid_x'] = self.submodule_dataset.index % self.grid_size[0]
        self.submodule_dataset['grid_y'] = (self.submodule_dataset.index / self.grid_size[1]) % self.grid_size[1]
        self.submodule_dataset['grid_z'] = self.submodule_dataset.index / (self.grid_size[2] * self.grid_size[2])

    def _generate_bin_packing_voxel_grid(self):
        '''
            Generates a voxel grid for the detector using a bin packing algorithm
        '''

        print('Generating bin packing voxel grid...')

        if 'normalized_x' not in self.submodule_dataset.columns:
            self._calculate_normalized_positions()

        # Combine the normalized positions into a single value
        self.submodule_dataset['combined'] = self.submodule_dataset[['normalized_x', 'normalized_y', 'normalized_z']].sum(axis=1)

        # Use a bin packing algorithm to assign each module to a different grid point
        bins = binpacking.to_constant_volume(self.submodule_dataset['combined'].to_dict(), 1)

        for i, bin in enumerate(bins):
            for module in bin:
                self.submodule_dataset.loc[module, 'grid_x'] = i % self.grid_size[0]
                self.submodule_dataset.loc[module, 'grid_y'] = (i / self.grid_size[1]) % self.grid_size[1]
                self.submodule_dataset.loc[module, 'grid_z'] = i / (self.grid_size[2] * self.grid_size[2])

    def _generate_pca_voxel_grid(self):

        print('Generating PCA voxel grid...')

        if 'normalized_x' not in self.submodule_dataset.columns:
            self._calculate_normalized_positions()

        pca = PCA(n_components=3)
        reduced_positions = pca.fit_transform(self.submodule_dataset[['cx', 'cy', 'cz']])

        min_reduced = np.min(reduced_positions, axis=0)
        max_reduced = np.max(reduced_positions, axis=0)

        self.grid_size = np.ceil(max_reduced - min_reduced + 1).astype(int)
        self.voxel_grid = np.zeros(self.grid_size, dtype=int)
        print('Lattice size: ', self.grid_size)

        # Normalize the reduced positions
        reduced_positions = np.round((reduced_positions - min_reduced) * (self.grid_size - 1) / (max_reduced - min_reduced)).astype(int)

        self.submodule_dataset['grid_x'] = reduced_positions[:, 0]
        self.submodule_dataset['grid_y'] = reduced_positions[:, 1]
        self.submodule_dataset['grid_z'] = reduced_positions[:, 2]



    # SUBMODULE SEGMENTATION - Interacting with the grid
    def get_voxel_position(self, x: float, y: float, z: float) -> tuple:
        '''
            Returns the voxel position of a given point based on the submodule the hit is closest to
        '''

        _, submodule_id = self.submodule_kdtree.query([x, y, z])
        submodule = self.submodule_dataset.iloc[submodule_id]
        return int(submodule['grid_x']), int(submodule['grid_y']), int(submodule['grid_z'])

    def add_hit_to_grid(self, x: float, y: float, z: float):
        '''
            Adds a hit to the voxel grid
        '''

        voxel_x, voxel_y, voxel_z = self.get_voxel_position(x, y, z)
        self.voxel_grid[voxel_x, voxel_y, voxel_z] += 1

    def show(self, ax: plt.Axes):
        '''
            Displays the voxel grid
        '''

        i, j, k = np.where(self.voxel_grid > 0)

        # Draw the voxel grid
        ax.scatter(i, j, k, c='r', s=self.voxel_grid[i, j, k])
        

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    def reset_grid(self):
        '''
            Resets the voxel grid
        '''

        self.voxel_grid = np.zeros(
            self.grid_size, dtype=np.float32)


class VoxelGridCylindrical:

    def __init__(self, detector_file: str, grid_size_r: int, grid_size_phi: int, grid_size_z: int, n_submodules: int = 1):
        '''
            Class to create a voxel grid for the detector in cylindrical coordinates.
            Used to create the input data for the convolutional neural network.
        '''

        self.grid_size = np.array((grid_size_r, grid_size_phi, grid_size_z), dtype=int)
        self.voxel_grid = np.zeros(
            (self.grid_size[0], self.grid_size[1], self.grid_size[2]), dtype=np.float32)
        self.detector_hashtable = DetectorHashTable(detector_file)
        
        self.submodule_dataset = None
        self.detector_dataset = pd.read_csv(detector_file, sep=',')
        self._create_submodule_df(n_submodules)
        
        self._evaluate_cylindrical_coordinates()
        self._calculate_normalized_positions()
        self._calculate_grid_positions()

        self.submodule_kdtree = KDTree(
            self.submodule_dataset[['cr', 'cphi', 'cz']].to_numpy())

    def _segment_module(self, module, n: int):
        '''
            Segments a module into n submodules. Since the modules are centered at (0, 0, 0) in uvw coordinates,
            the submodule position in xyz coordinates is just the submodule offset.
        '''

        submodules = []
        min_width_u = module['module_minhu']
        width_v = module['module_hv']

        submodule_width_u = 2 * min_width_u / n
        submodule_width_v = 2 * width_v / n

        u_positions = np.linspace(-min_width_u + submodule_width_u / 2, min_width_u - submodule_width_u / 2, n)
        v_positions = np.linspace(-width_v + submodule_width_v / 2, width_v - submodule_width_v / 2, n)

        rotation_matrix = np.array([
            [module['rot_xu'], module['rot_xv'], module['rot_xw']],
            [module['rot_yu'], module['rot_yv'], module['rot_yw']],
            [module['rot_zu'], module['rot_zv'], module['rot_zw']]
        ])

        for u in u_positions:
            for v in v_positions:

                submodule_offset_uvw = np.array([u, v, 0])
                submodule_additional_offset_xyz = rotation_matrix.dot(
                    submodule_offset_uvw) + np.array([module['cx'], module['cy'], module['cz']])

                submodule = module.copy()
                submodule['cx'] += submodule_additional_offset_xyz[0]
                submodule['cy'] += submodule_additional_offset_xyz[1]

                submodules.append(submodule)

        return pd.DataFrame(submodules)

    def _create_submodule_df(self, n: int) -> None:
        '''
            Creates a dataframe for detectr submodules and a 
        '''

        print('Creating submodule dataset...')

        self.submodule_dataset = pd.DataFrame()

        for _, module in self.detector_dataset.iterrows():
            submodules = self._segment_module(module, n)
            self.submodule_dataset = pd.concat(
                [self.submodule_dataset, submodules], ignore_index=True)

        # create a 'submodule_id' column containing the module index
        self.submodule_dataset['submodule_id'] = self.submodule_dataset.index

    def _evaluate_cylindrical_coordinates(self):
        '''
            Evaluates the cylindrical coordinates of the modules
        '''

        self.detector_dataset['cr'] = np.sqrt(
            self.detector_dataset['cx']**2 + self.detector_dataset['cy']**2)
        self.detector_dataset['cphi'] = np.arctan2(
            self.detector_dataset['cy'], self.detector_dataset['cx'])
        
        if self.submodule_dataset is not None:
            self.submodule_dataset['cr'] = np.sqrt(
                self.submodule_dataset['cx']**2 + self.submodule_dataset['cy']**2)
            self.submodule_dataset['cphi'] = np.arctan2(
                self.submodule_dataset['cy'], self.submodule_dataset['cx'])
        
    def _calculate_bounding_box(self):
        '''
            Calculates the bounding box of the voxel grid
        '''

        print('Calculating bounding box...')

        min_r = self.submodule_dataset['cr'].min()
        min_phi = self.submodule_dataset['cphi'].min()
        min_z = self.submodule_dataset['cz'].min()

        max_r = self.submodule_dataset['cr'].max()
        max_phi = self.submodule_dataset['cphi'].max()
        max_z = self.submodule_dataset['cz'].max()

        return min_r, min_phi, min_z, max_r, max_phi, max_z

    def _calculate_normalized_positions(self):
        '''
            Calculates the normalized positions of the submodules
        '''

        print('Calculating normalized positions...')

        min_r, min_phi, min_z, max_r, max_phi, max_z = self._calculate_bounding_box()

        scale_r = (self.grid_size[0] - 1) / (max_r - min_r)
        scale_phi = (self.grid_size[1] - 1) / (max_phi - min_phi)
        scale_z = (self.grid_size[2] - 1) / (max_z - min_z)

        self.submodule_dataset['normalized_r'] = (
            (self.submodule_dataset['cr'] - min_r) * scale_r).astype(int)
        self.submodule_dataset['normalized_phi'] = (
            (self.submodule_dataset['cphi'] - min_phi) * scale_phi).astype(int)
        self.submodule_dataset['normalized_z'] = (
            (self.submodule_dataset['cz'] - min_z) * scale_z).astype(int)

    def _calculate_grid_positions(self):
        '''
            Calculates the grid positions of the submodules
        '''

        self.submodule_dataset['grid_r'] = self.submodule_dataset['normalized_r']
        self.submodule_dataset['grid_phi'] = self.submodule_dataset['normalized_phi']
        self.submodule_dataset['grid_z'] = self.submodule_dataset['normalized_z']


    def get_voxel_position(self, r: float, phi: float, z: float) -> tuple:
        '''
            Returns the voxel position of a given point based on the submodule the hit is closest to
        '''

        _, submodule_id = self.submodule_kdtree.query([r, phi, z])
        submodule = self.submodule_dataset.iloc[submodule_id]
        return int(submodule['grid_r']), int(submodule['grid_phi']), int(submodule['grid_z'])    

    def add_dataset_to_grid(self, rs, phis, zs, weights=None):
        '''
            Adds a dataset to the voxel grid
        '''

        if weights is None:
            weights = np.ones(len(rs))

        for r, phi, z, weight in zip(rs, phis, zs, weights):
            voxel_r, voxel_phi, voxel_z = self.get_voxel_position(r, phi, z)
            self.voxel_grid[voxel_r, voxel_phi, voxel_z] += weight

    def show(self, ax: plt.Axes):
        '''
            Displays the voxel grid
        '''

        # Get the indices of the voxels where the condition is true
        i, j, k = np.where(self.voxel_grid > 0)

        # Draw the voxel grid
        ax.scatter(i, j, k, c='r', s=self.voxel_grid[i, j, k])

        ax.set_xlabel('r')
        ax.set_ylabel('phi')
        ax.set_zlabel('z')
        plt.show()