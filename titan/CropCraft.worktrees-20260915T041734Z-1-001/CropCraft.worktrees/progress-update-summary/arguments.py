import argparse


def get_args_nerf():
    parser = argparse.ArgumentParser(
        description='NeRF Training')
    parser.add_argument('--data_dir', type=str, default="./data/soybean",
                        help='path containing parent folder of input data (default: ./data/soybean)')
    parser.add_argument('--do_all', action='store_true',
                        help='whether to process all scenes in data_dir (default: False)')
    parser.add_argument('--scene_name', type=str, default="20230616_S1",
                        help='scene name if not all, needs to be in data_dir (default: 20230616_S1)')
    parser.add_argument('--is_uav', action='store_true',
                        help='whether to use NeRF parameters for UAV maize data (default: False)')
    parser.add_argument('--vis', type=str, default="viewer",
                        help='which visualizer to use, see nerfstudio docs for options (default: viewer)')
    parser.add_argument('--project_name', type=str, default="cropcraft_nerf",
                        help='project name for visualizer (default: cropcraft_nerf)')
    parser.add_argument('--out_dir', type=str, default="./work_dirs/",
                        help='path containing work dirs (default: ./work_dirs/)')
    args = parser.parse_args()
    return args


def get_args_alignment():
    parser = argparse.ArgumentParser(
        description='Row Alignment')
    parser.add_argument('--root_dir', type=str, default="./work_dirs/",
                        help='path containing work dirs (default: ./work_dirs/)')
    parser.add_argument('--scene_name', type=str, default="20230616_S1",
                        help='scene name, needs to be in root_dir (default: 20230616_S1)')
    parser.add_argument('--is_uav', action='store_true',
                        help='whether to override parameters with defaults for UAV maize data (default: False)')
    parser.add_argument('--no_vis', action='store_true',
                        help="""whether to suppress visualization (default: False)""")
    args = parser.parse_args()
    return args


def get_args_optimization():
    parser = argparse.ArgumentParser(
        description='Inverse Parametric Morphology Optimization')

    # General arguments
    parser.add_argument('--seed', type=int, default=100,
                        help='random seed (default: 100)')
    parser.add_argument('--root_dir', type=str, default="./work_dirs/",
                        help='path containing work dirs (default: ./work_dirs/)')
    parser.add_argument('--scene_name', type=str, default="20230616_S1",
                        help='scene name, needs to be in root_dir (default: 20230616_S1)')
    parser.add_argument('--renders_name', type=str, default="renders",
                        help='renders name (default: renders)')
    parser.add_argument('--results_name', type=str, default="exp1",
                        help='results name (default: exp1)')
    parser.add_argument('--use_wandb', action='store_true',
                        help='whether to activate wandb logging')
    parser.add_argument('--project_name', type=str, default="cropcraft",
                        help='project name for wandb (default: cropcraft)')
    parser.add_argument('--vis_freq', type=int, default=-1,
                        help='frequency for dumping visualizations, use -1 to disable (default: -1)')
    parser.add_argument('--print_freq', type=int, default=20,
                        help='interval for printing loss values and chosen parameters')
    parser.add_argument('--verbose', action='store_true',
                        help='whether to be display info each optimization step')
    
    # Preprocesing hyperparameters
    parser.add_argument('-udt', '--uncertain_depth_threshold', type=float, default=0.25,
                        help='distance from ground threshold for uncertain pixels in meters, \
                              used together with color mask for filtering background')
    parser.add_argument('-yt', '--y_threshold', type=float, default=0.38,
                        help='y-deviation from center threshold in meters')
    
    # Loss hyperparameters
    parser.add_argument('-l_m', '--lambda_mask', type=float, default=1.0,
                        help='weight for mask loss')
    parser.add_argument('-l_y', '--lambda_y', type=float, default=2.0,
                        help='weight for y-coordinate loss')
    parser.add_argument('-l_n', '--lambda_normal', type=float, default=0.0,
                        help='weight for normal loss')
    parser.add_argument('-l_sb', '--lambda_sobel', type=float, default=4.0,
                        help='weight for sobel loss')
    parser.add_argument('-nb_d', '--n_bins_depth', type=int, default=20,
                        help='number of bins in depth histogram')
    parser.add_argument('-nb_y', '--n_bins_y', type=int, default=10,
                        help='number of bins in y-coordinate histogram')
    parser.add_argument('-nb_sb', '--n_bins_sb', type=int, default=10,
                        help='number of bins in sobel histogram')

    # Optimization hyperparameters
    parser.add_argument('-bo_mi', '--bo_max_iter', type=int, default=500,
                        help='number of total function evaluations')
    parser.add_argument('-bo_ip', '--bo_init_pts', type=int, default=200,
                        help='number of random initial points')
    parser.add_argument('--n_reps', type=int, default=1,
                        help='number of full optimization reps')

    args = parser.parse_args()
    return args


def get_args_visualization():
    parser = argparse.ArgumentParser(
        description='3D Visualization')
    parser.add_argument('--root_dir', type=str, default="./work_dirs/",
                        help='path containing work dirs (default: ./work_dirs/)')
    parser.add_argument('--scene_name', type=str, default="20230616_S1",
                        help='scene name, needs to be in root_dir (default: 20230616_S1)')
    parser.add_argument('--results_name', type=str, default="exp1",
                        help='results name (default: exp1)')
    parser.add_argument('--is_maize', action='store_true',
                        help='if maize rather than soybean (default: False)')
    parser.add_argument('--n_rows', type=int, default=3,
                        help='number of rows to generate (default: 3)')
    parser.add_argument('--row_len', type=float, default=2.0,
                        help='length of each row in meters (default: 2.0)')
    
    args = parser.parse_args()
    return args


def get_args_evaluation():
    parser = argparse.ArgumentParser(
        description='Evaluation Metrics Calculation')
    parser.add_argument('--gt_dir', type=str, default="./data/field_measurements/",
                        help='path containing GT field measurements (default: ./data/field_measurements/)')
    parser.add_argument('--root_dir', type=str, default="./work_dirs/",
                        help='path containing work dirs (default: ./work_dirs/)')
    parser.add_argument('--scene_name', type=str, default="20230616_S1",
                        help='scene name, can be "all_soy" or "all_maize" or single (default: 20230616_S1)')
    parser.add_argument('--renders_name', type=str, default="renders",
                        help='renders name (default: renders)')
    parser.add_argument('--results_name', type=str, default="exp1",
                        help='results name (default: exp1)')
    parser.add_argument('--is_maize', action='store_true',
                        help='if maize rather than soybean (default: False)')
    parser.add_argument('--verbose', action='store_true',
                        help='whether to be display extra info per scene (default: False)')
    
    args = parser.parse_args()
    return args
    