import subprocess
import os
from arguments import get_args_nerf


if __name__ == '__main__':
    args = get_args_nerf()

    if args.do_all:
        scene_names = sorted(os.listdir(args.data_dir))
    else:
        scene_names = [args.scene_name]

    for scene_name in scene_names:
        ns_in_dir = os.path.join(args.data_dir, scene_name)
        
        if args.is_uav:
            result = subprocess.run([
                'ns-train', 'nerfacto',
                '--data', ns_in_dir,
                '--output_dir', args.out_dir,
                '--vis', args.vis,
                '--project_name', args.project_name,
                '--experiment_name', args.scene_name,
                '--max_num_iterations', '20000',
                '--pipeline.model.near-plane', '0.05',
                '--pipeline.model.far-plane', '6.',
                '--pipeline.model.hidden-dim', '128', 
                '--pipeline.model.hidden-dim-color', '128', 
                'nerfstudio-data', '--orientation-method', 'vertical'
            ])
        else:
            result = subprocess.run([
                'ns-train', 'nerfacto',
                '--data', ns_in_dir,
                '--output_dir', args.out_dir,
                '--vis', args.vis,
                '--project_name', args.project_name,
                '--experiment_name', args.scene_name,
                '--max_num_iterations', '20000'
            ])

