'''
    Classes for the detector geometry
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class DetectorModule:

    def __init__(self, module:pd.Series):
        
        self.features = {key: float(value) for key, value in module.items()}
        
        self.volume_id = 0
        
        if 'volume_id' in self.features.keys():
            self.volume_id = self.features['volume_id']

    def vertices_uvw(self, mode:str='quiet'):
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
        if mode == 'verbose':  print('vertices_uvw: ', self.features['vertices_uvw'])

    def vertices_xyz(self, mode:str='quiet'):
        '''
            Finds the faces (vertex coordinates of the faces) of the module in xyz coordinates
        '''

        self.vertices_uvw()
        self.features['vertices_xyz'] = [[0, 0, 0] for vertex in self.features['vertices_uvw']]

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

        for vtx_xyz, vtx_uvw in zip(self.features['vertices_xyz'], self.features['vertices_uvw']) :
            vtx_xyz[0] = vtx_uvw[0] * rotation_matrix[0][0] + vtx_uvw[1] * rotation_matrix[0][1] + vtx_uvw[2] * rotation_matrix[0][2] + offset[0]
            vtx_xyz[1] = vtx_uvw[0] * rotation_matrix[1][0] + vtx_uvw[1] * rotation_matrix[1][1] + vtx_uvw[2] * rotation_matrix[1][2] + offset[1]
            vtx_xyz[2] = vtx_uvw[0] * rotation_matrix[2][0] + vtx_uvw[1] * rotation_matrix[2][1] + vtx_uvw[2] * rotation_matrix[2][2] + offset[2]

        if mode == 'verbose':  print('vertices_xyz: ', self.features['vertices_xyz'])

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

    def draw_uvw(self, ax: plt.Axes, mode: str='quiet', **kwargs):
        '''
            Draws the module in 3D
        '''
        self.vertices_uvw(mode)
        self.features['faces_uvw'] = self.module_faces(self.features['vertices_uvw'])
        ax.add_collection3d(Poly3DCollection(self.features['faces_uvw'], **kwargs))

    def draw_xyz(self, ax: plt.Axes, mode: str='quiet', **kwargs):
        '''
            Draws the module in 3D
        '''
        self.vertices_xyz(mode)
        self.features['faces_xyz'] = self.module_faces(self.features['vertices_xyz'])
        ax.add_collection3d(Poly3DCollection(self.features['faces_xyz'], **kwargs))

    def draw_rz(self, ax: plt.Axes, mode: str='quiet', **kwargs):
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
        if mode == 'verbose':  print('vertices_rz: ', self.features['vertices_rz'])
        
        # Draw the module as a rectangle in the rz plane (2d)
        for i in range(len(vertices_rz)):
            if i < len(vertices_rz) - 1:
                ax.plot([vertices_rz[i][1], vertices_rz[i+1][1]], [vertices_rz[i][0], vertices_rz[i+1][0]], **kwargs)
            else:
                ax.plot([vertices_rz[i][1], vertices_rz[0][1]], [vertices_rz[i][0], vertices_rz[0][0]], **kwargs)
    
    def draw_xy(self, ax: plt.Axes, mode: str='quiet', **kwargs):
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
        if mode == 'verbose':  print('vertices_xy: ', self.features['vertices_xy'])
        
        # Draw the module as a rectangle in the xy plane (2d)
        for i in range(len(vertices_xy)):
            if i < len(vertices_xy) - 1:
                ax.plot([vertices_xy[i][0], vertices_xy[i+1][0]], [vertices_xy[i][1], vertices_xy[i+1][1]], **kwargs)
            else:
                ax.plot([vertices_xy[i][0], vertices_xy[0][0]], [vertices_xy[i][1], vertices_xy[0][1]], **kwargs)



class DetectorGeometry:

    def __init__(self, inFile:str):

        if not inFile.endswith('.csv'):
            raise ValueError('The input file must be a csv file')
        
        self.detector = pd.read_csv(inFile, sep=' ')
        
        self.inner_barrel_sel = 'volume_id==8'  # string to select the inner barrel modules
        self.forward_layer_sel = 'layer_id==12 and (volume_id==14 or volume_id==18)' # string to select the forward layer modules

        self.volume_colors = {7: 'b', 8: 'b', 9: 'b',
                              12: 'r', 13: 'r', 14: 'r',
                              16: 'g', 17: 'g', 18: 'g'
                              }
    

    def detector_display(self, ax: plt.Axes, mode: str='quiet', **kwargs):
        '''
            Draws the detector in 3D
        '''

        for imodule in self.detector.iterrows():
            module = DetectorModule(imodule[1])
            module.draw_xyz(ax, mode=mode, facecolors=self.volume_colors[module.volume_id], **kwargs)

        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')
        ax.set_xlim(-1000, 1000)
        ax.set_ylim(-1000, 1000)
        ax.set_zlim(-4000, 4000)
        
    def inner_barrel_display(self, ax: plt.Axes, mode: str='quiet', **kwargs):
        '''
            Draws the inner barrel modules in 3D
        '''

        for imodule in self.detector.query(self.inner_barrel_sel, inplace=False).iterrows():
            module = DetectorModule(imodule[1])
            module.draw_xyz(ax, mode=mode, facecolors=self.volume_colors[module.volume_id], **kwargs)

        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')
        ax.set_xlim(-200, 200)
        ax.set_ylim(-200, 200)
        ax.set_zlim(-600, 600)

    def forward_layer_display(self, ax: plt.Axes, mode: str='quiet', **kwargs):
        '''
            Draws the forward layer modules in 3D
        '''

        for imodule in self.detector.query(self.forward_layer_sel, inplace=False).iterrows():
            module = DetectorModule(imodule[1])
            module.draw_xy(ax, mode=mode, facecolors=self.volume_colors[module.volume_id], **kwargs)

        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_xlim(-1100, 1100)
        ax.set_ylim(-1100, 1100)

    def radial_display(self, ax: plt.Axes, mode: str='quiet', **kwargs):
        '''
            Draws the detector in the rz plane
        '''

        for imodule in self.detector.iterrows():
            module = DetectorModule(imodule[1])
            module.draw_rz(ax, mode=mode, facecolors=self.volume_colors[module.volume_id], **kwargs)

        ax.set_xlabel('z (mm)')
        ax.set_ylabel('r (mm)')
        ax.set_xlim(-4000, 4000)
        ax.set_ylim(0, 1100)