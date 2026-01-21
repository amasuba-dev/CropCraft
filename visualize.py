import os
import json
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from fit_soybean_o3d import soybean_field_parametric
from fit_maize_o3d import maize_field_parametric
from arguments import get_args_visualization


if __name__ == '__main__':
    args = get_args_visualization()

    work_dir = os.path.join(args.root_dir, args.scene_name)
    results_dir = os.path.join(work_dir, 'results', args.results_name)
    viz_dir = os.path.join(work_dir, 'viz', args.results_name)

    with open(os.path.join(work_dir, 'info.json'), 'rb') as info_json_f:
        info = json.load(info_json_f)
    # row_dist = info['row_dist']
    row_dist = 0.73025 if args.is_maize else 0.762  # use known row distance (meters)

    all_final_bp = np.load(os.path.join(results_dir, 'all_final_best_params.npy'), allow_pickle=True)
    params = {k:np.mean([p[k] for p in all_final_bp], axis=0) for k in all_final_bp[0].keys()}

    if args.is_maize:
        xyzs, leaf_facs = maize_field_parametric(params, n_rows=args.n_rows, row_len=args.row_len, row_dist=row_dist)
    else:
        xyzs, leaf_facs = soybean_field_parametric(params, n_rows=args.n_rows, row_len=args.row_len, row_dist=row_dist)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(xyzs)
    mesh.triangles = o3d.utility.Vector3iVector(leaf_facs)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.3, 0.45, 0.25])

    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    os.makedirs(viz_dir, exist_ok=True)
    o3d.io.write_triangle_mesh(os.path.join(viz_dir, 'mesh_output.ply'), mesh)
