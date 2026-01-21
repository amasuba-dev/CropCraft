import numpy as np
import sys
from morphology.utils import *
from morphology.maize_constants import new_leaf, new_stem
import scipy.io as sio

#, leaf_Mask
#hs: an array of height of leaves
#dS: diameter of stem
#Nt: Number of corn leaves
#LA: an ararry of leaf areas
#Azi: an array of azimuth angles of corn leaves
#leafAspectRatio: the ratio of leaf length and leaf width
#leaf_angle: leaf bending angle
def maize_plant(hs, dS, Nt, LA, Azi, leafAspectRatio, leaf_angle, separate=False):

    height = hs[-1]

    Li = np.sqrt(LA/0.75/leafAspectRatio)
    Wi = Li*leafAspectRatio

    leaf_xyz, leaf_fac = [], []
    #Azi = 0
    #area = 0
    PlantHeight = 0
    for i in range(Nt):
        # if leaf_Mask[i]:
        #     continue
        xyz, fac = new_leaf(order = leaf_angle[i], length = Li[i], width = Wi[i]) 
        xyz[:, -1] += hs[i]
        if i == Nt-1:
            PlantHeight = np.max(xyz[:, -1])
        xyz = rotate2d(xyz, Azi[i], dims=[0, 1])

        #xyz = rotate2d(xyz, 90, dims=[1, 2])
        #Azi = (Azi + 180) % 360

        #vis_mesh(xyz, fac)
        leaf_xyz.append(xyz)
        leaf_fac.append(fac)
        #area += vis_mesh(xyz, fac)

    #print(area)
    if Nt > 7:
        stem_xyz, stem_fac = new_stem(hs[6], Nt, height, dS)
    else:
        stem_xyz, stem_fac = new_stem(0, Nt, height, dS)
    #stem_xyz = rotate2d(stem_xyz, 90, dims=[1, 2])

    if separate:
        return leaf_xyz, stem_xyz, leaf_fac, stem_fac, PlantHeight
    else:
        return agg(leaf_xyz + [stem_xyz], leaf_fac + [stem_fac])

# if __name__ == '__main__':
#     # Example parameters
#     Nt = 15
#     LA = 0.4
#     height = 1.5
#
#     mat1 = sio.loadmat('mat1.mat')
#     mat2 = sio.loadmat('mat2.mat')
#     mat1 = mat1['mat1']
#     mat2 = mat2['mat2']
#     Azi = []
#     for i in range(25):
#         if i % 2 == 0:
#             Azi.append(np.random.choice(mat2[0, :], p=mat2[1, :]))
#         else:
#             Azi.append(np.random.choice(mat1[0, :], p=mat1[1, :]))
#
#     xyz, fac = maize_plant(height, 0.04, Nt, LA, Azi)
#     mesh = vis_mesh(xyz, fac)
