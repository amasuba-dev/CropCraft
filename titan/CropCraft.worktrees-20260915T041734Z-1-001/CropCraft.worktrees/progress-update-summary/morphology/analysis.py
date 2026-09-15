import numpy as np
import sys
import os 
sys.path.append(os.getcwd())

from morphology.utils import *
from morphology.soybean_model import soybean_leaf, soybean_petiole
from morphology.reparameterize import soybean_info_parametric


def soybean_node_leaves(lat1L, lat1W, lat1ang, 
                        lat2L, lat2W, lat2ang, 
                        midL, midW, midang, 
                        pet1len, pet1ang, 
                        pet2len, pet2ang):
    """
    Returns XYZ points and faces (index triplets) for a node, which contains two petioles and a leaf trio.
    
    lat1L, lat1W, lat1ang: length, width, and z-angle of first lateral leaf (same for lat2 and mid)
    pet1len: length of parent petiole (petiole1)
    pet2len: length of petiole with middle leaf
    pet1ang: angle between petiole1 and parent stem
    pet2ang: angle between petiole2 and petiole1
    """
    mid_xyz, mid_fac = soybean_leaf(midL, midW, midang)
    mid_xyz[:, 0] += pet2len
    
    lat1_xyz, lat1_fac = soybean_leaf(lat1L, lat1W, lat1ang)
    lat2_xyz, lat2_fac = soybean_leaf(lat2L, lat2W, -lat2ang)  # rotate the other way from lat1
    
    pet2_xyz, pet2_fac = soybean_petiole(pet2len, 0.2)
    pet1_xyz, pet1_fac = soybean_petiole(pet1len, 0.3)
    
    all_xyz, all_fac = agg([mid_xyz, lat1_xyz, lat2_xyz, pet2_xyz], [mid_fac, lat1_fac, lat2_fac, pet2_fac])

    leaf_fac = np.concatenate([mid_fac, lat1_fac + len(mid_xyz), lat2_fac + len(mid_xyz) + len(lat1_xyz)], axis=0)

    all_xyz = rotate2d(all_xyz, pet2ang, dims=[0, 2])
    all_xyz[:, 0] += pet1len
    all_xyz, all_fac = agg([all_xyz, pet1_xyz], [all_fac, pet1_fac])
    all_xyz = rotate2d(all_xyz, pet1ang, dims=[0, 2])
    
    return all_xyz, all_fac, leaf_fac


def soybean_stem_leaves(stem_info):
    """
    Returns XYZ points and faces (index triplets) for a stem, which contains multiple nodes.
    """
    total_height = 0
    total_n = 0
    node_xyzs, node_facs = [], []
    all_leaf_fac = np.zeros((0, 3), dtype=int)
    for node_data in stem_info:
        node_xyz, node_fac, node_leaf_fac = soybean_node_leaves(**node_data['node_info'])
        node_xyz = rotate2d(node_xyz, node_data['node_rotation'], dims=[1, 2])
        node_xyz = rotate2d(node_xyz, node_data['node_ang'], dims=[0, 1])
        total_height += node_data['node_height']
        node_xyz[:, 0] += total_height
        node_xyzs.append(node_xyz)
        node_facs.append(node_fac)

        node_leaf_fac += total_n
        all_leaf_fac = np.append(all_leaf_fac, node_leaf_fac, axis=0)

        total_n += len(node_xyz)
        
    stem_xyz, stem_fac = soybean_petiole(total_height, 0.5)
    all_xyz, all_fac = agg(node_xyzs + [stem_xyz], node_facs + [stem_fac])
    return all_xyz, all_fac, all_leaf_fac

def soybean_plant_leaves(main_stem_info, branches_info):
    """
    Returns XYZ points and faces (index triplets) for a full plant, which contains a main stem and branches.
    """
    total_n = 0
    branch_xyzs, branch_facs = [], []
    all_leaf_fac = np.zeros((0, 3), dtype=int)
    for branch_data in branches_info:
        branch_xyz, branch_fac, branch_leaf_fac = soybean_stem_leaves(branch_data['branch_info'])
        branch_xyz = rotate2d(branch_xyz, branch_data['branch_ang'], dims=[0, 2])
        branch_xyz = rotate2d(branch_xyz, branch_data['branch_rotation'], dims=[1, 2])
        branch_xyz[:, 0] += branch_data['branch_height']
        branch_xyzs.append(branch_xyz)
        branch_facs.append(branch_fac)

        branch_leaf_fac += total_n
        all_leaf_fac = np.append(all_leaf_fac, branch_leaf_fac, axis=0)

        total_n += len(branch_xyz)
        
    main_stem_xyz, main_stem_fac, main_stem_leaf_fac = soybean_stem_leaves(main_stem_info)
    main_stem_leaf_fac += total_n

    all_xyz, all_fac = agg(branch_xyzs + [main_stem_xyz], branch_facs + [main_stem_fac])
    all_leaf_fac = np.concatenate([all_leaf_fac, main_stem_leaf_fac])
    return all_xyz, all_fac, all_leaf_fac


def soybean_plant_leaves_parametric(params, return_non_leaf=False):
    """Returns soybean plant with custom parameterization and the leaf faces isolated."""
    main_stem_info, branches_info = soybean_info_parametric(params)
    xyz, fac, leaf_fac = soybean_plant_leaves(main_stem_info, branches_info)
    if return_non_leaf:
        facs_tuples = [tuple(f) for f in fac]
        leaf_facs_tuples = [tuple(f) for f in leaf_fac]
        non_leaf_facs_tuples = [f for f in facs_tuples if f not in leaf_facs_tuples]
        non_leaf_fac = np.array(non_leaf_facs_tuples)
        return xyz, fac, leaf_fac, non_leaf_fac
    else:
        return xyz, fac, leaf_fac


def total_surface_area(xyz, fac):
    area = 0
    for i in range(0, len(fac)):
        a = xyz[fac[i][1]] - xyz[fac[i][0]]
        b = xyz[fac[i][2]] - xyz[fac[i][0]]
        area += np.linalg.norm(np.cross(a, b))/2

    return area
    

def azimuth_angles(xyz, fac, deg=True):
    angs = []
    for i in range(0, len(fac)):
        # get normal vector of triangle
        a = xyz[fac[i][1]] - xyz[fac[i][0]]
        b = xyz[fac[i][2]] - xyz[fac[i][0]]
        n = np.cross(a, b)

        # get angle between normal and (1, 0, 0)
        n = n / np.linalg.norm(n)
        # check if n has nan
        if np.isnan(n).any():
            continue
        prod = np.dot(n, np.array([1, 0, 0]))
        if prod < 0:
            prod = -prod
        angs.append(np.arccos(prod))
    if deg:
        return np.array(angs) * 180 / np.pi
    else:
        return np.array(angs)


def plant_leaf_area_parametric(params, in_m2=True):
    """Returns soybean plant area (fast leaf-counting method)."""
    
    # from M_branch_probability.txt
    BRANCH_NODE_COUNT_MEANS = [0.25, 1.3, 1.0, 0.55, 0.8, 0.5]
    NODE_FRAC = params['NODE_COUNT'] - int(params['NODE_COUNT'])
    NODE_COUNT = int(params['NODE_COUNT'])
    BRANCH_COUNT = min(max(0, NODE_COUNT - 7), 6)
    
    BASE_LEAF_L = np.array([51.82, 70.91, 77.24, 94.83, 87.5, 87.47, 93.9, 101.21, 108.53, 108.34, 98.2, 90.11, 84.63, 69.86, 58.42, 58.42]) / 10
    LEAF_W_TO_L = params['LEAF_W_TO_L'] if 'LEAF_W_TO_L' in params else 0.8

    if 'LEAF_L_SCALE' in params:
        BASE_LEAF_L = BASE_LEAF_L[len(BASE_LEAF_L) - NODE_COUNT - 1:]
        LEAF_L_SCALE = params['LEAF_L_SCALE']
        LEAF_W_SCALE = params['LEAF_L_SCALE'] * LEAF_W_TO_L
        BRANCH_LEAF_L = 8.  # base
    else:  # uniform leaf size
        BASE_LEAF_L[:] = params['LEAF_L']
        LEAF_L_SCALE = 1.
        LEAF_W_SCALE = LEAF_W_TO_L
        BRANCH_LEAF_L = params['LEAF_L']


    unit_leaf_area = total_surface_area(*soybean_leaf(1, 1, 0))
    main_area = 0
    for i in range(int(NODE_COUNT)):
        main_area += 3 * unit_leaf_area * BASE_LEAF_L[i]**2 * LEAF_L_SCALE * LEAF_W_SCALE 
    main_area += 3 * unit_leaf_area * BASE_LEAF_L[NODE_COUNT]**2 * LEAF_L_SCALE * LEAF_W_SCALE * NODE_FRAC**2
    branch_leaf_area = unit_leaf_area * BRANCH_LEAF_L**2 * LEAF_L_SCALE * LEAF_W_SCALE
    branches_area = sum([(BRANCH_NODE_COUNT_MEANS[j] * 3 * branch_leaf_area) for j in range(BRANCH_COUNT)])
    if BRANCH_COUNT < len(BRANCH_NODE_COUNT_MEANS):
        branches_area += BRANCH_NODE_COUNT_MEANS[BRANCH_COUNT] * NODE_FRAC**2 * 3 * branch_leaf_area
    area = main_area + branches_area
    
    if in_m2:
        area /= 10000
    return area


def maize_plant_leaf_area_parametric(params, in_m2=True):
    """Returns maize plant area (fast method)."""
    BASE_LA = np.array([0.00012, 0.00034, 0.000272, 0.0255, 0.00416, 0.00658, 0.01374, 0.02501, 0.03643, 
                        0.04901, 0.05913, 0.05637, 0.05128, 0.04519, 0.03713, 0.0285, 0.01981, 0.01278]) * 10000
    Nt = int(params['NODE_COUNT'])
    LA = BASE_LA[:Nt] * params['LEAF_L_SCALE'] ** 2
    area = np.sum(LA)
    if in_m2:
        area /= 10000
    return area


def leaf_area_index_parametric(params, row_dist, is_maize=False):
    """Returns LAI."""
    if is_maize:
        plant_area = maize_plant_leaf_area_parametric(params, in_m2=True)
    else:
        plant_area = plant_leaf_area_parametric(params, in_m2=True)
    return plant_area * params['PLANT_DENSITY'] / row_dist


def leaf_angles_parametric(params, samples=1):
    """Returns all leaf azimuth angles."""
    xyzs, facs, leaf_facs = soybean_plant_leaves_parametric(params)
    angs = azimuth_angles(xyzs, leaf_facs)
    if samples > 1:
        for _ in range(samples - 1):
            xyzs, facs, leaf_facs = soybean_plant_leaves_parametric(params)
            angs = np.concatenate([angs, azimuth_angles(xyzs, leaf_facs)])
    return angs


if __name__ == '__main__':
    # Example parameters
    params = {'LEAF_L_SCALE': 1., 'PET_ANG_SCALE': 1.4650475579010003, 'NODE_DIST_SCALE': 1.1465324617408656, 'PET_LEN_SCALE': 1.72937187938393, 
              'AZIMUTH_STD': 19.92396832909082, 'NODE_COUNT': 9.729604437133741, 'LEAF_L_STD': 0.0, 'PET1_ANG_STD': 0.0, 'NODE_DIST_STD': 0.0, 'PLANT_DENSITY': 20.0}

    np.random.seed(100)
    plant_areas = []
    for _ in range(50):
        # main_stem_info, branches_info = soybean_info_parametric(params)
        xyzs, facs, leaf_facs = soybean_plant_leaves_parametric(params)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(xyzs)
        mesh.triangles = o3d.utility.Vector3iVector(leaf_facs)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.25, 0.4, 0.25])

        print('Total surface area: %.4f m^2' % (total_surface_area(xyzs, leaf_facs) / 100**2))
        plant_areas.append(total_surface_area(xyzs, leaf_facs) / 100**2)
    print(np.mean(plant_areas))
    print(plant_leaf_area_parametric(params, in_m2=True))
    angs = azimuth_angles(xyzs, leaf_facs)
    
    import matplotlib.pyplot as plt
    plt.hist(angs, bins=20)
    plt.show()
    cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([mesh, cf], mesh_show_wireframe=True, mesh_show_back_face=True)