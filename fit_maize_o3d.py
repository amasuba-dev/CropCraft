import os
os.environ["OMP_NUM_THREADS"] = "8"  # limiting might improve speed when running in parallel

import numpy as np
import cv2
import sys
import open3d as o3d
import time
import json
import wandb
import scipy.optimize
from skopt import gp_minimize

from morphology.maize_model import maize_plant, rotate2d, agg
from morphology.analysis import leaf_area_index_parametric
from morphology.reparameterize import maize_info_parametric
from arguments import get_args_optimization
from visualization_utils import visualize_maps, visualize_hists

sys.stdout = sys.stderr

PARAMETER_RANGES = {
    'LEAF_L_SCALE': [0.8, 1.2],
    'LEAF_ORDER_SHIFT': [-4.0, 4.0], 
    'NODE_DIST_SCALE': [0.8, 1.2], 
    'NODE_COUNT': [1, 18],
    
    'AZIMUTH_STD': [60, 60],
    'LEAF_L_STD': [0., 0.],
    'PET1_ANG_STD': [5., 5.],
    'NODE_DIST_STD': [0., 0.],
    
    'PLANT_DENSITY': [6.468, 6.468],
}

TO_FIT = ['LEAF_L_SCALE', 'LEAF_ORDER_SHIFT', 'NODE_DIST_SCALE', 'NODE_COUNT'] 


def maize_plant_parametric(params):
    """Returns maize plant with custom parameterization."""
    hs, dS, Nt, LA, Azi, leafAspectRatio, leaf_angle = maize_info_parametric(params)
    xyz, fac = maize_plant(hs, dS, Nt, LA, Azi, leafAspectRatio, leaf_angle)
    xyz = rotate2d(xyz, 90, dims=[0, 2]) 
    return xyz, fac


def maize_field_parametric(params, n_rows=5, row_len=6.6, row_dist=0.73025):
    """Returns maize field with custom parameterization."""
    xyzs, facs = [], []
    for j in range(n_rows):
        for center_x in np.arange(-row_len/2, row_len/2, 1/params['PLANT_DENSITY']):
            xyz, fac = maize_plant_parametric(params)
            xyz /= 100.  # cm to m
            xyz = rotate2d(xyz, 270, dims=[0, 2])  # Upright along z-axis
            xyz = rotate2d(xyz, np.random.choice([90, 270]), dims=[0, 1])  # Align
            xyz[:, 0] += center_x
            xyz[:, 1] += j * row_dist - (n_rows // 2) * row_dist

            xyzs.append(xyz)
            facs.append(fac)
    return agg(xyzs, facs)
    
    
def make_new_mesh(vis, params):
    """Update open3d mesh and visualizer with new maize field."""
    xyz, fac = maize_field_parametric(params)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(xyz)
    mesh.triangles = o3d.utility.Vector3iVector(fac)
    vis.clear_geometries()
    vis.add_geometry(mesh)
    return vis
    

def render_depth(vis, w2c_pose, intrinsic=None):
    """Render depth using open3d Visualizer."""
    ctr = vis.get_view_control()
    cam = ctr.convert_to_pinhole_camera_parameters()
    render_camera = o3d.camera.PinholeCameraParameters()
    if intrinsic is None:
        render_camera.intrinsic = cam.intrinsic
    else:
        render_camera.intrinsic = intrinsic
    render_camera.extrinsic = w2c_pose
    ctr.convert_from_pinhole_camera_parameters(render_camera, allow_arbitrary=True)

    vis.get_render_option().mesh_show_back_face=True
    depth = np.array(vis.capture_depth_float_buffer(True))
    depth[depth == 0] = 9999  # background
    return depth


def normals_from_depth(dmap):
    """
    Computes surface normals from a depth map using Sobel filter.
    :param dmap: A grayscale depth map image as a numpy array of size (H,W).
    :return: The corresponding surface normals map as numpy array of size (H,W,3).
    """
    zx = cv2.Sobel(dmap, cv2.CV_64F, 1, 0, ksize=5)
    zy = cv2.Sobel(dmap, cv2.CV_64F, 0, 1, ksize=5)
    
    # convert to unit vectors
    normals = np.dstack((-zx, -zy, np.ones_like(dmap)))
    length = np.linalg.norm(normals, axis=2)
    for c in range(3):
        normals[:, :, c] /= length

    # offset and rescale values to be in 0-1
    normals += 1
    normals /= 2
    return normals[:, :, :-1].astype(np.float32)


def hist_loss(h1, h2):
    """Calculate loss between two batches of histograms."""
    return np.mean(np.sum((h1 - h2)**2, axis=1))

    
def sample_params_uniform():
    """Sample a single parameter configuration from a uniform distribution."""
    params = {}
    for param in PARAMETER_RANGES:
        low, high = PARAMETER_RANGES[param]
        params[param] = np.random.uniform(low, high)
    return params
    
    
def hists_from_maps(maps, masks, bins):
    """Calculate histograms for multiple (batched) maps."""
    N = len(maps)
    h, w = maps[0].shape
    
    hists = np.zeros((N, len(bins) - 1))
    for j, m in enumerate(maps):
        total_pixels = np.sum(masks[j])  # or just h*w?
        hists[j] = np.histogram(m[masks[j]], bins=bins, density=False)[0] / max(1, total_pixels)
    return hists


def smooth_sobel(depth, blur_kernel=55, max_depth=1.5):
    """Compute Sobel map from smoothed (blurred) depth map."""
    depth = np.clip(depth, a_min=0, a_max=max_depth)
    blur = cv2.GaussianBlur(depth, (blur_kernel, blur_kernel), 0)
    sobelxy = cv2.Sobel(src=blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)
    return sobelxy
    
    
def all_hists_from_depths(depths, d_bins, nx_bins, ny_bins, y_bins, sb_bins, f_y, foreground_thr=1.5):
    """Calculate depth, y-coord, normal x, and normal y histograms for multiple depth and normal maps."""
    N = len(depths)
    h, w = depths[0].shape
    
    sobels = [smooth_sobel(d, max_depth=foreground_thr) for d in depths]
    normals = np.array([normals_from_depth(d) for d in depths])
    fg_masks = [(d < foreground_thr) for d in depths]

    xx, yy = np.meshgrid(np.arange(-w/2, w - w/2, 1), np.arange(-h/2, h - h/2, 1))
    ys = [(yy * d / f_y) for d in depths]

    d_hists = hists_from_maps(depths, fg_masks, d_bins)
    y_hists = hists_from_maps(np.abs(ys), fg_masks, y_bins)
    nx_hists = hists_from_maps(normals[:, :, :, 0], fg_masks, nx_bins)
    ny_hists = hists_from_maps(normals[:, :, :, 1], fg_masks, ny_bins)
    sb_hists = hists_from_maps(np.abs(sobels), np.ones((N, h, w), dtype=bool), sb_bins)
    return d_hists, nx_hists, ny_hists, y_hists, sb_hists, fg_masks


def params2str(params, subset=None):
    subset = subset if subset is not None else params
    return ', '.join([('%s: %.2f' % (param_name, params[param_name])) for param_name in subset])


def fit_maize(args, target_depths, w2c_poses, K,
              foreground_thr=1.5, method='bo', seed=1234):
    """Fit the model parameters to target (observed) depth maps, using Bayesian optimization."""
    
    def params_from_vector(param_vector):
        """Parameter dictionary from subset vector."""
        params = params_init.copy()
        for i in range(len(param_subset)):
            params[param_subset[i]] = param_vector[i] * ranges[i][1]
        return params
    
    def f(param_vector):
        """Function to optimize."""
        global iters
        params = params_from_vector(param_vector)

        if args.results_name == 'debug': 
            # Test with fixed params
            # params = {'LEAF_L_SCALE': 1.1997026167582134, 'LEAF_ORDER_SHIFT': -3.6533237320152963, 'NODE_DIST_SCALE': 0.8808780227175241, 'AZIMUTH_STD': 60.0, 'NODE_COUNT': 13.901712470694104, 'LEAF_L_STD': 0.0, 'PET1_ANG_STD': 5.0, 'NODE_DIST_STD': 0.0, 'PLANT_DENSITY': 6.468000000000001}
            params = {'LEAF_L_SCALE': 1.186971885731695, 'LEAF_ORDER_SHIFT': -4.0, 'NODE_DIST_SCALE': 0.8913570160876187, 'AZIMUTH_STD': 60.0, 'NODE_COUNT': 14.518445651102095, 'LEAF_L_STD': 0.0, 'PET1_ANG_STD': 5.0, 'NODE_DIST_STD': 0.0, 'PLANT_DENSITY': 6.468000000000001}
        
        if iters % args.print_freq == 0:
            print('Rep %d Iter %4d |' % (rep, iters), params2str(params, param_subset))
        
        # Create mesh and update visualizer
        make_new_mesh(vis, params)
        
        # Render depths 
        depths = [render_depth(vis, pose, intrinsic) for pose in w2c_poses]

        # Calculate necessary statistics for loss
        d_hists, nx_hists, ny_hists, y_hists, sb_hists, fg_masks = \
            all_hists_from_depths(depths, d_bins, nx_bins, ny_bins, y_bins, sb_bins, f_y, foreground_thr)
        mask_areas = np.array([np.mean(m) for m in fg_masks])
        
        # Visualize maps and histograms at current step:
        if iters % args.vis_freq == 0 and args.vis_freq >= 0:
            _, sobels, normals, ys, fg_masks = \
                all_hists_from_depths(depths, d_bins, nx_bins, ny_bins, y_bins, sb_bins, f_y, foreground_thr, return_maps=True)
            _, target_sobels, target_normals, target_ys, target_fg_masks = \
                all_hists_from_depths(target_depths, d_bins, nx_bins, ny_bins, y_bins, sb_bins, f_y, foreground_thr, return_maps=True)
            
            viz_dir = os.path.join(work_dir, 'viz', args.results_name, 'iter_%04d' % iters)
            os.makedirs(viz_dir, exist_ok=True)
            visualize_maps(depths[0], ys[0], sobels[0], 
                           target_depths[0], target_ys[0], target_sobels[0], fg_masks[0], target_fg_masks[0],
                           foreground_thr, viz_dir)
            visualize_hists(depths[0], d_hists[0], y_hists[0], sb_hists[0], 
                            target_depths[0], target_d_hists[0], target_y_hists[0], target_sb_hists[0],
                            d_bins, y_bins, sb_bins, foreground_thr, viz_dir)
        
        # Calculate loss terms
        mask_area_loss = np.mean((mask_areas - target_mask_areas)**2)
        d_loss = hist_loss(d_hists, target_d_hists)
        y_loss = hist_loss(y_hists, target_y_hists)
        sb_loss = hist_loss(sb_hists, target_sb_hists)
        
        loss = d_loss + \
               args.lambda_normal * (hist_loss(nx_hists, target_nx_hists) + hist_loss(ny_hists, target_ny_hists)) + \
               args.lambda_mask * mask_area_loss + \
               args.lambda_y * y_loss + \
               args.lambda_sobel * sb_loss
        
        if iters % args.print_freq == 0:
            print('Losses | d: %.4f(x%.1f), y: %.4f(x%.1f), sb: %.4f(x%.1f), mask: %.4f(x%.1f), total: %.4f' % 
                  (d_loss, 1, y_loss, args.lambda_y, sb_loss, args.lambda_sobel, mask_area_loss, args.lambda_mask, loss))
            print('-' * 40)
        iters += 1
        if args.use_wandb:
            wandb.log({'loss_%d' % rep: loss, 'logloss_%d' % rep: np.log10(loss), 
                    'LEAF_L_SCALE_%d' % rep: params['LEAF_L_SCALE'], 'NODE_COUNT_%d' % rep: params['NODE_COUNT']})
        
        if args.results_name == 'debug':
            exit(0)   

        return loss
    
    # Define histogram bins
    d_bins = np.arange(2.0, view_height - 0.0 + 1e-5, (view_height - 2.0) / args.n_bins_depth)
    nx_bins = np.arange(0, 1.01, 1/10)
    ny_bins = np.arange(0, 1.01, 1/10)
    y_bins = np.arange(0, view_height/2 + 1e-5, view_height/2 / args.n_bins_y)
    sb_bins = np.arange(0, 0.004 + 1e-5, 0.004 / args.n_bins_sb)

    # Get hists for target depths
    N = len(target_depths)
    h, w = target_depths[0].shape
    f_y = K[1, 1]

    target_d_hists, target_nx_hists, target_ny_hists, target_y_hists, target_sb_hists, target_fg_masks = \
        all_hists_from_depths(target_depths, d_bins, nx_bins, ny_bins, y_bins, sb_bins, f_y, foreground_thr)
    target_mask_areas = np.array([np.mean(m) for m in target_fg_masks])
    
    params_init = {k: v[0] for (k, v) in PARAMETER_RANGES.items()}
    
    param_subset = TO_FIT
    ranges = [PARAMETER_RANGES[param_name] for param_name in param_subset]
    scaled_ranges = [(r[0] / r[1], 1) for r in ranges]

    # Setup open3d rendering
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h, visible=False)  # set to true for on-screen open3d viz

    make_new_mesh(vis, params_init)

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    start = time.time()
        
    if method == 'tr':
        res = scipy.optimize.minimize(f, [(r[0] + r[1]) / 2 for r in scaled_ranges], 
                                      method='trust-constr', bounds=scaled_ranges, 
                                      options={'maxiter': args.bo_max_iter / 10})


        best_loss = res.fun
        best_params = params_from_vector(res.x)
        loss_history = [best_loss]  # does not return intermediate results
        params_history = [best_params]
        
    else:
        res = gp_minimize(f,                               # the function to minimize
                        scaled_ranges,                   # the bounds on each dimension of x
                        acq_func="EI",                   # the acquisition function
                        n_calls=args.bo_max_iter,           # the number of evaluations of f
                        n_initial_points=args.bo_init_pts,  # the number of random initialization points
                        initial_point_generator='sobol',    # the initial point generation strategy
                        noise='gaussian',                # the noise level (optional)
                        random_state=seed,               # the random seed
                        n_restarts_optimizer=10,
                        verbose=bool(args.verbose))          
            
        best_loss = res['fun']
        best_params = params_from_vector(res['x'])
        loss_history = res['func_vals']
        params_history = [params_from_vector(x_iter) for x_iter in res['x_iters']]
    print('total time:', time.time() - start, 's')
        
    if args.use_wandb:
        make_new_mesh(vis, best_params)
        result_depth_viz = -np.clip(render_depth(vis, w2c_poses[0], intrinsic), a_min=0, a_max=foreground_thr)[:, :, None]
        result_depth_viz = (result_depth_viz - result_depth_viz.min()) / (result_depth_viz.max() - result_depth_viz.min()) * 255
        result_depth_viz = wandb.Image(result_depth_viz.astype(np.uint8))
        target_depth_viz = -np.clip(target_depths[0], a_min=0, a_max=foreground_thr)[:, :, None]
        target_depth_viz = (target_depth_viz - target_depth_viz.min()) / (target_depth_viz.max() - target_depth_viz.min()) * 255
        target_depth_viz = wandb.Image(target_depth_viz.astype(np.uint8))
        target_mask_viz = wandb.Image(target_fg_masks[0].astype(np.float32)[:, :, None])
        wandb.log({
            "target_depth": target_depth_viz,
            "target_mask": target_mask_viz,
            "final_result_depth_%d" % rep: result_depth_viz,
        }, step=(rep + 1) * args.bo_max_iter)
        
    vis.destroy_window()
    
    return best_params, best_loss, loss_history, params_history


def preprocess_ns_depth(ns_depth, vfov, scale, mask, uncertain_depth=2, max_depth=5., min_depth=0.1, y_thr=0.5):
    """Preprocess nerfstudio depth: correct from distance to depth, and filter bg/invalid."""
    h, w = ns_depth.shape
    focal_length = (h // 2) / np.tan((vfov * np.pi / 180) / 2)
    xx, yy = np.meshgrid(np.arange(-w/2, w - w/2, 1), np.arange(-h/2, h - h/2, 1))
    image_plane_dist = np.sqrt(xx**2 + yy**2 + focal_length**2)
    depth = ns_depth.astype(np.float64) * focal_length / image_plane_dist
    depth /= scale
    ydev = np.abs(yy * depth / focal_length)
    
    depth[(mask == 0) & (depth > uncertain_depth)] = 9999
    depth[depth > max_depth] = 9999
    depth[depth < min_depth] = 9999
    depth[ydev > y_thr] = 9999
    if np.mean(mask) < 0.3:  # plants are small
        depth[ydev > y_thr / 2] = 9999
    
    return depth
    
    
def get_seg_mask(rgb, a_thresh=120, L_thresh=40):
    """Threshold-based segmentation mask from RGB."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return (lab[:, :, 1] < a_thresh) & (lab[:, :, 0] > L_thresh)
    
    
if __name__ == '__main__':
    
    # Get command-line args
    args = get_args_optimization()
    args.y_threshold = 3.
    args.uncertain_depth_threshold = 2.
    
    # Choose which renders to use
    render_idxs = [0]
    
    if args.use_wandb:
        # Start WandB logging
        wandb.init(
            project=args.project_name,
            name=args.scene_name + '_' + args.results_name,
            config=args
        )
                 
    work_dir = os.path.join(args.root_dir, args.scene_name)
    results_dir = os.path.join(work_dir, 'results', args.results_name)
    os.makedirs(results_dir, exist_ok=True)
        
    with open(os.path.join(work_dir, 'info.json'), 'rb') as info_json_f:
        info = json.load(info_json_f)
    scale = info['ns_scale']
    vfov = info['vfov']
    view_height = info['view_height']  # top-down height
    print(info)
              
    # Load and prepare target (observation) renders
    render_dir = os.path.join(work_dir, 'renders')
    target_rgbs = [cv2.imread(os.path.join(render_dir, '%05d.jpg' % i))[:, :, ::-1] for i in render_idxs]
    target_masks = [get_seg_mask(rgb) for rgb in target_rgbs]
    target_depths = [np.load(os.path.join(render_dir, '%05d.npy' % i))[:, :, 0] for i in render_idxs]
    target_depths = [preprocess_ns_depth(ns_depth, vfov, scale, mask, 
                                         uncertain_depth=view_height - args.uncertain_depth_threshold, y_thr=args.y_threshold)
                      for (ns_depth, mask) in zip(target_depths, target_masks)]
    
    w2c_poses = [np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, view_height],  
        [0, 0, 0, 1]
    ])]
    
    h, w, = target_depths[0].shape
    foc = (h // 2) / np.tan((vfov * np.pi / 180) / 2)
    K = np.array([
        [foc, 0, w//2],
        [0, foc, h//2],
        [0, 0, 1]
    ])
    
    # Fix seed
    np.random.seed(args.seed)
    
    all_final_best_loss, all_final_best_params = [], []
    
    for rep in range(args.n_reps):
        iters = 0  # make iteration counter global because it's used in opt function
        best_params, best_loss, best_loss_history, best_params_history = \
            fit_maize(args, target_depths, w2c_poses, K, 
                      foreground_thr=view_height, seed=rep)

        print('best params:', best_params)
        print('best_loss:', best_loss)
        print('lai:', leaf_area_index_parametric(best_params, info['row_dist'], is_maize=True))
        np.save(os.path.join(results_dir, 'best_loss_history_%02d.npy' % rep), best_loss_history)
        np.save(os.path.join(results_dir, 'best_params_history_%02d.npy' % rep), best_params_history)
        all_final_best_params.append(best_params)
        all_final_best_loss.append(best_loss)
        
    np.save(os.path.join(results_dir, 'all_final_best_params.npy'), all_final_best_params)
    np.save(os.path.join(results_dir, 'all_final_best_loss.npy'), all_final_best_loss)

    wandb.finish()
        
