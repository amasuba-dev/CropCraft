import numpy as np
import os
import pandas as pd 
from morphology.analysis import leaf_area_index_parametric, leaf_angles_parametric
from arguments import get_args_evaluation


def load_lai(gt_lai_dir, scene_name):
    mmdd_date = scene_name[4:8]
    loc_name = scene_name[9:]
    mapping = {'0816': '0817', '0904': '0908', '0909': '0908'}
    if mmdd_date in mapping:
        mmdd_date = mapping[mmdd_date]
    lai_file = os.path.join(gt_lai_dir, '%s.csv' % mmdd_date)
    data = np.loadtxt(lai_file, skiprows=1, delimiter=',', dtype=str)
    assert loc_name in data[:, 0]
    lai = float(data[data[:, 0] == loc_name, 2])
    return lai

def load_lad(gt_lad_dir, scene_name):
    mmdd_date = scene_name[4:8]
    loc_name = scene_name[9:]
    df = pd.read_excel(os.path.join(gt_lad_dir, '%s_soybean.xlsx' % mmdd_date), 
                    header = None, sheet_name='Sheet%s' % loc_name[-1])
    df = df.to_numpy()
    gt_angs = df[1:, 1].astype(float)
    return gt_angs

def percent_error(gt, pred):
    return 100 * (pred - gt) / gt

def calc_ape(gts, preds):
    return np.mean([abs(percent_error(gt, pred)) for gt, pred in zip(gts, preds)])

def calc_rmse(gts, preds):
    return np.sqrt(np.mean([(gt - pred)**2 for gt, pred in zip(gts, preds)]))

def calc_ame(gts, preds):
    gt_means = [np.mean(gt) for gt in gts]
    pred_means = [np.mean(pred) for pred in preds]
    return np.sqrt(np.mean([(gt - pred)**2 for gt, pred in zip(gt_means, pred_means)]))

def calc_ase(gts, preds):
    gt_stds = [np.std(gt) for gt in gts]
    pred_stds = [np.std(pred) for pred in preds]
    return np.sqrt(np.mean([(gt - pred)**2 for gt, pred in zip(gt_stds, pred_stds)]))

def eval_lai_metrics(gt_lai_dir, root_dir, scene_names, results_name, 
                     is_maize=False, row_dist=0.762, verbose=True):
    gt_lais = []
    avg_pred_lais = []

    print('-' * 40 + '\nLAI evaluation for ' + results_name + '\n' + '-' * 40)
    for scene_name in scene_names:
        gt_lai = load_lai(gt_lai_dir, scene_name)
        gt_lais.append(gt_lai) 

        results_base_dir = os.path.join(root_dir, scene_name, 'results')
        results_dir = os.path.join(results_base_dir, results_name)

        all_final_bl = np.load(os.path.join(results_dir, 'all_final_best_loss.npy'))
        all_final_bp = np.load(os.path.join(results_dir, 'all_final_best_params.npy'), allow_pickle=True)

        avg_params = {k:np.mean([p[k] for p in all_final_bp], axis=0) for k in all_final_bp[0].keys()}
        avg_pred_lai = leaf_area_index_parametric(avg_params, row_dist, is_maize)
        avg_pred_lais.append(avg_pred_lai)

        if verbose:
            print('%s | L: %.4f, Pred LAI: %.2f, GT LAI: %.2f' % 
                  (scene_name, np.mean(all_final_bl), avg_pred_lai, gt_lai))

    laie = calc_rmse(gt_lais, avg_pred_lais)
    laipe = calc_ape(gt_lais, avg_pred_lais) / 100
    print('LAIE: %.4f, LAIPE: %.4f (%.4f%%)' % (laie, laipe, laipe * 100))

def eval_lad_metrics(gt_lad_dir, root_dir, scene_names, results_name, verbose=True):
    gt_lads = []
    avg_pred_lads = []

    print('-' * 40 + '\nLAD evaluation for ' + results_name + '\n' + '-' * 40)
    for scene_name in scene_names:
        gt_lad = load_lad(gt_lad_dir, scene_name)
        gt_lads.append(gt_lad) 

        results_base_dir = os.path.join(root_dir, scene_name, 'results')
        results_dir = os.path.join(results_base_dir, results_name)

        all_final_bp = np.load(os.path.join(results_dir, 'all_final_best_params.npy'), allow_pickle=True)

        avg_params = {k:np.mean([p[k] for p in all_final_bp], axis=0) for k in all_final_bp[0].keys()}
        avg_pred_lad = leaf_angles_parametric(avg_params, samples=100)
        avg_pred_lads.append(avg_pred_lad)

        if verbose:
            print('%s | Pred mean leaf angle: %.2f, GT mean leaf angle: %.2f' % 
                  (scene_name, np.mean(avg_pred_lad), np.mean(gt_lad)))
            
    ame = calc_ame(gt_lads, avg_pred_lads)
    ase = calc_ase(gt_lads, avg_pred_lads)
    print('AME: %.2f, ASE: %.2f' % (ame, ase))


if __name__ == '__main__':
    args = get_args_evaluation()

    if args.scene_name == 'all_soy':
        soybean_dates = ['0616', '0627', '0705',  '0711', '0720', '0801']
        soybean_locs = ['S1', 'S2', 'S3']
        soybean_scene_names = ['2023%s_%s' % (d, l) for d in soybean_dates for l in soybean_locs]
        existing_scene_names = os.listdir(args.root_dir)
        existing_soybean_scene_names = [s for s in soybean_scene_names if s in existing_scene_names]
        missing_soybean_scene_names = [s for s in soybean_scene_names if s not in existing_scene_names]
        if len(missing_soybean_scene_names) > 0:
            print('Warning: missing soybean scenes (%d):' % len(missing_soybean_scene_names), missing_soybean_scene_names)
        scene_names = existing_soybean_scene_names
    elif args.scene_name == 'all_maize':
        maize_dates = ['0816', '0828', '0904', '0909', '0914']
        maize_locs = ['FF']
        maize_scene_names = ['2023%s_%s' % (d, l) for d in maize_dates for l in maize_locs]
        existing_scene_names = os.listdir(args.root_dir)
        existing_maize_scene_names = [s for s in maize_scene_names if s in existing_scene_names]
        missing_maize_scene_names = [s for s in maize_scene_names if s not in existing_scene_names]
        if len(missing_maize_scene_names) > 0:
            print('Warning: missing maize scenes (%d):' % len(missing_maize_scene_names), missing_maize_scene_names)
        scene_names = existing_maize_scene_names
    else:
        scene_names = [args.scene_name]

    gt_lai_dir = os.path.join(args.gt_dir, 'LAI')
    gt_lad_dir = os.path.join(args.gt_dir, 'LAD')
    row_dist = 0.73025 if args.is_maize else 0.762  # use known row distance (meters)

    eval_lai_metrics(gt_lai_dir, args.root_dir, scene_names, args.results_name, 
                     is_maize=args.is_maize, row_dist=row_dist, verbose=args.verbose)
    eval_lad_metrics(gt_lad_dir, args.root_dir, scene_names, args.results_name, verbose=args.verbose)
