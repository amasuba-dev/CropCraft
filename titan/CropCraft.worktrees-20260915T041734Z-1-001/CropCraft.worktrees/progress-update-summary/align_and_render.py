import open3d as o3d
import skimage
import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
import os
import random
import pyransac3d as pyrsc
import json
import subprocess
from arguments import get_args_alignment


def ransac_plane(xyz, dist_thresh=0.05):
    rsc_plane = pyrsc.Plane()
    best_eq, best_inliers = rsc_plane.fit(xyz, thresh=dist_thresh, minPoints=5000, maxIteration=1000)
    return best_eq

def ransac_line(xyz, dist_thresh=0.1, return_inliers=False):
    rsc_line = pyrsc.Line()
    best_A, best_B, best_inliers = rsc_line.fit(xyz, thresh=dist_thresh, maxIteration=1000)
    if return_inliers:
        return best_A, best_B, best_inliers
    else:
        return best_A, best_B

def lsq_line(xy):
    m, b = np.polyfit(xy[:, 0], xy[:, 1], deg=1)
    A = np.array([1, m])
    B = np.array([0, b])
    return A, B

def project_point_to_line(pt, A, B):
    to_project = pt - B
    unit_A = A / np.linalg.norm(A)
    projected = np.outer(np.dot(to_project, unit_A), unit_A)
    if pt.ndim == 1:
        projected = projected[0]
    return projected + B

def project_point_to_plane(pt, eq):
    normal = eq[:3] / np.linalg.norm(eq[:3])
    dist = np.dot(normal, pt) + eq[3] 
    projected = pt - dist * normal
    return projected

def spheres_on_plane(plane_eq, n_spheres=100):
    spheres = []
    for _ in range(n_spheres):
        sphere_center = project_point_to_plane((np.random.rand(3) -0.5) * 2, plane_eq)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
        sphere = sphere.translate(sphere_center)
        sphere.paint_uniform_color([1, 0, 1])
        sphere.compute_vertex_normals()
        spheres.append(sphere)
    return spheres

def align_ground(xyz, ground_plane_eq):
    new_origin = project_point_to_plane(np.mean(xyz, axis=0), ground_plane_eq)
    
    # Align z-axis with normal of ground plane, positive INTO the ground
    basis_vec3 = ground_plane_eq[:3] / np.linalg.norm(ground_plane_eq[:3])
    if np.dot(basis_vec3, new_origin) < 0:
        basis_vec3 *= -1
    basis_vec1 = np.array([1., 0., 0.])
    basis_vec1 -= np.dot(basis_vec1, basis_vec3) * basis_vec3
    basis_vec1 /= np.linalg.norm(basis_vec1)
    basis_vec2 = np.cross(basis_vec3, basis_vec1)
    basis_vec2 /= np.linalg.norm(basis_vec2)
    new_basis = np.hstack([basis_vec1[:, None], basis_vec2[:, None], basis_vec3[:, None]])

    new_to_old = np.eye(4)
    new_to_old[:3, :3] = new_basis
    new_to_old[:3, 3] = new_origin
    old_to_new = np.linalg.inv(new_to_old)
    
    new_xyz = xyz - new_origin
    new_xyz = (np.linalg.inv(new_basis) @ new_xyz.T).T

    return new_xyz, old_to_new


def get_poses_and_row_dist_seqrsc(pcd, scale, 
                                  pose_height=None, 
                                  ds_voxel_size=0.01, 
                                  lab_thresh = (0, 0, 0),  # higher a means more fg points
                                  ground_inlier_thresh=0.1,
                                  row_inlier_thresh=0.24,
                                  refit_line=True,
                                  return_height=True,
                                  roi_x_center=-0.5,
                                  visualize=True,
                                  render_hw=(738, 994)):
    """
    Get standardized per-row camera poses in nerfstudio format and 
    the distance across rows, using sequential RANSAC to detect rows.
    """
    
    # Downsample point cloud (units are larger than meters by 'scale' factor)
    pcd = pcd.voxel_down_sample(voxel_size=ds_voxel_size * scale)
    pcd_xyz = np.asarray(pcd.points)
    pcd_rgb = np.asarray(pcd.colors)
    
    # Color threshold segmentation
    pcd_lab = (skimage.color.rgb2lab(pcd_rgb))
    is_black = pcd_lab[:, 0] < lab_thresh[0]
    is_green = pcd_lab[:, 1] < lab_thresh[1]
    is_blue = pcd_lab[:, 2] < lab_thresh[2]
    is_too_close = pcd_xyz[:, 2] > np.percentile(pcd_xyz[:, 2], 1, 0) + 2.
    
    # Align with ground plane
    # pcd_z = pcd_xyz[:, 2]
    # in_iqr = (pcd_z > np.percentile(pcd_z, 25)) & (pcd_z < np.percentile(pcd_z, 75)) 
    is_ground = ~is_green  & (~is_black) & (~is_blue) & (~is_too_close)
    ground_xyz = pcd_xyz[is_ground]
    ground_plane_eq = ransac_plane(ground_xyz, dist_thresh=ground_inlier_thresh * scale)
    pcd_xyz, old_to_new = align_ground(pcd_xyz, ground_plane_eq)

    capture_height = np.abs(ground_plane_eq[-1]) / scale
    print('estimated capture height:', capture_height)  

    # Select plant and ground points
    both_pcd = pcd
    both_pcd.points = o3d.utility.Vector3dVector(pcd_xyz)
    both_pcd = both_pcd.select_by_index(np.where((~is_black) & (~is_blue) & (~is_too_close))[0])

    # Visualize ground plane and aligned all points
    if visualize:
        # draw a wireframe square on xy plane
        square = o3d.geometry.TriangleMesh.create_box(width=4, height=4, depth=0.001)
        square = square.translate([-2, -2, 0])
        square = o3d.geometry.LineSet.create_from_triangle_mesh(square)
        square.paint_uniform_color([1, 0.5, 0])
        o3d.visualization.draw_geometries([square, both_pcd])

    # Get foreground (plant) points, in ground-aligned coordinates
    is_plant = is_green & (~is_black) & (~is_blue)
    plant_xyz = pcd_xyz[is_plant]
    plant_rgb = pcd_rgb[is_plant]
    plant_pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(plant_xyz))
    plant_pcd.colors = o3d.utility.Vector3dVector(plant_rgb)
    is_large_plant = np.sum(is_plant) > 0.75 * len(pcd_xyz) 
    print('plant fraction: %.3f | is large plant: %s' % (np.sum(is_plant) / len(pcd_xyz), is_large_plant))

    # Visualize plant points
    if visualize:
        o3d.visualization.draw_geometries([square, plant_pcd])

    # Extract a ROI (circular patch) and remove points too far and too close to ground
    x, y, z = np.moveaxis(plant_xyz, 1, 0)
    is_roi = (np.sqrt((x-roi_x_center)**2 + y**2) < 2 * scale)
    z_percentile_thr = 30 if is_large_plant else 50
    is_top = z < np.percentile(z, z_percentile_thr, 0)  # note that z is positive below (into) the ground, negative above 
    is_close = z > np.percentile(z, z_percentile_thr, 0) - 0.5  * scale 
    filtered_pcd = plant_pcd.select_by_index(np.where(is_top & is_roi & is_close)[0])
    filtered_xyz = np.asarray(filtered_pcd.points)
    filtered_rgb = np.asarray(filtered_pcd.colors)
    print('plant points before filtering: %d | after filtering: %d' % (len(plant_xyz), len(filtered_xyz)))
    if visualize:
        o3d.visualization.draw_geometries([square, filtered_pcd])

    # Segment into rows
    unassigned_pcd = filtered_pcd
    line_pcds = []
    line_coeffs = []
    print('Fitting rows...')
    while True:
        unassigned_xyz = np.asarray(unassigned_pcd.points)
        unassigned_xyz[:, 2] = 0
        if len(unassigned_xyz) < 3:
            break
        if not is_large_plant:
            row_inlier_thresh = 0.2
        A, B, inliers = ransac_line(unassigned_xyz, dist_thresh=row_inlier_thresh * scale, return_inliers=True)
        # display_inlier_outlier(unassigned_pcd, inliers)
        if len(inliers) < 1000 and len(inliers) < len(filtered_xyz) * 0.2:
            break
        
        line_pcds.append(unassigned_pcd.select_by_index(inliers))
        if refit_line:
            A, B = lsq_line(unassigned_xyz[inliers])
        line_coeffs.append((A, B))
        unassigned_pcd = unassigned_pcd.select_by_index(inliers, invert=True)

        print('Row %d: %d inliers, %d remaining' % (len(line_pcds), len(inliers), len(unassigned_xyz) - len(inliers)))

    print(f"{len(line_pcds)} rows detected.")
    assert len(line_pcds) > 1, 'not enough rows detected!'
    
    # Visualize segments
    if visualize:
        for i, line_pcd in enumerate(line_pcds):
            line_pcd.paint_uniform_color(plt.get_cmap("tab20")(i / len(line_pcds))[:3])
        o3d.visualization.draw_geometries(line_pcds)

    # Row "centers" (xy)
    seg_means = [np.mean(np.asarray(line_pcd.points), axis=0)[:2] for line_pcd in line_pcds]

    # Estimate cross-row distance from neighbor pairs
    neighbor_dists = []
    for i, seg_mean in enumerate(seg_means):
        dists = []
        for j, line_coeff in enumerate(line_coeffs):
            if j != i:
                A, B = line_coeff
                projected = project_point_to_line(seg_mean, A, B)
                dists.append(np.linalg.norm(seg_mean - projected))
                    
        neighbor_dists += sorted(dists)[:2]
    
    cross_dist = np.median(neighbor_dists)
    
    # Define camera to render depth map with
    if pose_height is None:
        pose_height = 1.25 if is_large_plant else 1.0
    
    ns_c2ws = []
    for i, seg_mean in enumerate(seg_means):
        dx, dy = line_coeffs[i][0][:2] if line_coeffs[i][0][0] > 0 else -line_coeffs[i][0][:2]
        theta = np.arctan(dy/dx)
        o3d_c2w = np.eye(4)
        o3d_c2w[:2, 3] = seg_mean[:2]
        o3d_c2w[2, 3] = -pose_height * scale
        o3d_c2w[0, 0] = np.cos(theta)
        o3d_c2w[1, 1] = np.cos(theta)
        o3d_c2w[0, 1] = -np.sin(theta)
        o3d_c2w[1, 0] = np.sin(theta)
        ns_c2w = np.linalg.inv(old_to_new) @ o3d_c2w
        ns_c2w[:3, 2] *= -1
        ns_c2ws.append(ns_c2w)

    if visualize:
        print('Previewing render camera poses (using point cloud)...')
        # Set up open3d renderer
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=render_hw[1], height=render_hw[0])
        vis.add_geometry(plant_pcd)
        ctr = vis.get_view_control()
        cam = ctr.convert_to_pinhole_camera_parameters()
        render_camera = o3d.camera.PinholeCameraParameters()
        render_camera.intrinsic = cam.intrinsic
        render_camera.extrinsic = np.linalg.inv(o3d_c2w)
        ctr.convert_from_pinhole_camera_parameters(render_camera)
        
        # Check depth render in plt
        vis.run()
        depth = vis.capture_depth_float_buffer(False)
        plt.imshow(depth)
        plt.colorbar()
        plt.show()
        cam = ctr.convert_to_pinhole_camera_parameters()
        vis.destroy_window()
    
    if return_height:
        return ns_c2ws, cross_dist, pose_height

    return ns_c2ws, cross_dist


def get_last_file_in_folder(folder):
    files = os.listdir(folder)
    return os.path.join(folder, sorted(files, reverse=True)[0])


if __name__ == '__main__':
    args = get_args_alignment()
    
    root_dir = args.root_dir
    scene_name = args.scene_name
    is_uav = args.is_uav
    visualize = not args.no_vis
    work_dir = os.path.join(root_dir, scene_name)
    ns_dir = get_last_file_in_folder(os.path.join(work_dir, 'nerfacto'))
    
    # Outputs will be created here
    render_dir = os.path.join(work_dir, 'renders')
    pcd_path = os.path.join(work_dir, 'point_cloud.ply')
    out_cp_json_path = os.path.join(work_dir, 'ns_camera_path.json')
    info_json_path = os.path.join(work_dir, 'info.json')
    
    # Get ns scale
    with open(os.path.join(ns_dir, 'dataparser_transforms.json')) as dt_json_f:
        dt_json = json.load(dt_json_f)
    scale = float(dt_json['scale'])       
    
    # Extract point cloud if not already done
    if not os.path.exists(pcd_path):# or True:
        print(pcd_path)
        if is_uav:
            result = subprocess.run([
                'ns-export', 'pointcloud',
                '--load-config', os.path.join(ns_dir, 'config.yml'),
                '--output-dir', work_dir,
                '--num-points', '5000000',
                '--remove-outliers', 'True',
                '--normal-method', 'open3d',
                '--use-bounding-box', 'True',
                '--bounding-box-min', str(-1), str(-1), str(-1),
                '--bounding-box-max', str(1), str(1), str(1),
            ])
        else:
            # Calling ns-export, 2m x 2m x 3m bounding box for iPad data
            result = subprocess.run([
                'ns-export', 'pointcloud',
                '--load-config', os.path.join(ns_dir, 'config.yml'),
                '--output-dir', work_dir,
                '--num-points', '100000',
                '--remove-outliers', 'True',
                '--normal-method', 'open3d',
                '--use-bounding-box', 'True',
                '--bounding-box-min', str(-scale), str(-scale), str(-3 * scale),
                '--bounding-box-max', str(scale), str(scale), str(0),
            ])
    
    # Fix seed
    random.seed(100)

    # Compute row-aligned poses and row distance
    pcd = o3d.io.read_point_cloud(pcd_path)
    lab_thresh = (32, 0, 8) if is_uav else (0, 2, 1)
    roi_x_center = -0.5 if is_uav else 0
    ground_inlier_thresh = 0.1 if is_uav else 0.05
    view_height = 5. if is_uav else None  # None for auto
    render_hw = (738, 994)
    print('scale (from nerfstudio):', scale)
    poses, row_dist, view_height = get_poses_and_row_dist_seqrsc(pcd, scale, view_height, 
                                                                 lab_thresh=lab_thresh, 
                                                                 ground_inlier_thresh=ground_inlier_thresh,
                                                                 roi_x_center=roi_x_center,
                                                                 visualize=visualize,
                                                                 render_hw=render_hw)
    row_dist /= scale
    print('render pose 0:\n')
    print(poses[0])

    # Creating camera path json for ns-render, start from existing default
    default_cp = './base_camera_path.json' 
    with open(default_cp) as cp_json_f:
        cp_json = json.load(cp_json_f)

    cp_json['keyframes'] = []
    new_camera_path = []
    vfov = cp_json['camera_path'][0]['fov']
    aspect = render_hw[1] / render_hw[0]  
    for i, pose in enumerate(poses):
        camera_dict = {
            'camera_to_world': pose.flatten().tolist(),
            'fov': vfov,
            'aspect': aspect,
        }
        new_camera_path.append(camera_dict)
    cp_json['camera_path'] = new_camera_path
    cp_json['render_height'] = render_hw[0]
    cp_json['render_width'] = render_hw[1]
    
    with open(out_cp_json_path, 'w') as json_f:
        json.dump(cp_json, json_f)
        
    # Calling ns-render
    result = subprocess.run([
        'ns-render', 'camera-path',
        '--load-config', os.path.join(ns_dir, 'config.yml'),
        '--camera-path-filename', out_cp_json_path, 
        '--output-path', render_dir,
        '--output-format', 'images',
        '--rendered-output-names', 'metric_depth', 'rgb',
    ])
    
    # Save useful info
    info = {'view_height': view_height, 'ns_scale': scale,  
            'row_dist': row_dist, 'vfov': vfov, 'aspect': aspect}
    with open(info_json_path, 'w') as json_f:
        json.dump(info, json_f)
    print(info)
    