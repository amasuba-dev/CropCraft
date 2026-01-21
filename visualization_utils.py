import os
import numpy as np
import matplotlib.pyplot as plt


def visualize_maps(depth, y, sobel, target_depth, target_y, target_sobel, fg_mask, target_fg_mask,
                   foreground_thr, save_dir, show=False):

    def plot_map(data, cmap, clims, savename):
        plt.imshow(data, cmap=cmap)
        plt.axis('off')
        plt.clim(clims[0], clims[1])
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_dir, savename), dpi=200, bbox_inches='tight', pad_inches=0)
        if show:
            plt.show()
        plt.close()

    clim_low = np.percentile(np.append(depth, target_depth), 0.5) - 0.05
    clim_high = np.min([np.max([np.max(d) for d in [depth, target_depth]]), foreground_thr])
    plot_map(np.clip(depth, a_min=0, a_max=foreground_thr), cmap='inferno_r', clims=(clim_low, clim_high),
             savename='pred_depth.png')
    plot_map(np.clip(target_depth, a_min=0, a_max=foreground_thr), cmap='inferno_r', clims=(clim_low, clim_high),
             savename='obs_depth.png')

    y[fg_mask == 0] = 0
    target_y[target_fg_mask == 0] = 0
    clim_low = 0
    clim_high = np.max([np.max(np.abs(d)) for d in y + target_y])
    plot_map(np.clip(np.abs(y), a_min=clim_low, a_max=clim_high), cmap='viridis', clims=(clim_low, clim_high),
             savename='pred_y.png')
    plot_map(np.clip(np.abs(target_y), a_min=clim_low, a_max=clim_high), cmap='viridis', clims=(clim_low, clim_high),
             savename='obs_y.png')
            
    clim_low = 0
    clim_high = np.max([np.max(np.abs(d)) for d in sobel + target_sobel]) 
    plot_map(np.clip(np.abs(sobel), a_min=clim_low, a_max=clim_high), cmap='magma', clims=(clim_low, clim_high),
             savename='pred_sobel.png')
    plot_map(np.clip(np.abs(target_sobel), a_min=clim_low, a_max=clim_high), cmap='magma', clims=(clim_low, clim_high),
             savename='obs_sobel.png')
            
            
def visualize_hists(depth, d_hist, y_hist, sb_hist, 
                    target_depth, target_d_hist, target_y_hist, target_sb_hist,
                    d_bins, y_bins, sb_bins, foreground_thr,
                    save_dir, show=False):

    fig, axs = plt.subplots(2, 4, figsize=(16, 6),  gridspec_kw={'width_ratios': [2, 1, 1, 1]})
            
    clim_low = np.percentile(np.append(depth, target_depth), 0.5) - 0.05
    clim_high = np.min([np.max([np.max(d) for d in [depth, target_depth]]), foreground_thr])
    axs[0, 0].imshow(np.clip(target_depth, a_min=0, a_max=foreground_thr), cmap='inferno_r', vmin=clim_low, vmax=clim_high)
    axs[1, 0].imshow(np.clip(depth, a_min=0, a_max=foreground_thr), cmap='inferno_r', vmin=clim_low, vmax=clim_high)

    axs[0, 1].set_title('depth')
    axs[0, 1].bar(d_bins[:-1], target_d_hist, width=d_bins[1] - d_bins[0])
    # axs[0, 1].set_ylim(0, 0.3)
    axs[1, 1].bar(d_bins[:-1], d_hist, width=d_bins[1] - d_bins[0])
    # axs[1, 1].set_ylim(0, 0.3)
            
    axs[0, 2].set_title('y-coordinates')
    axs[0, 2].bar(y_bins[:-1], target_y_hist, width=y_bins[1] - y_bins[0])
    axs[1, 2].bar(y_bins[:-1], y_hist, width=y_bins[1] - y_bins[0])
            
    axs[0, 3].set_title('depth sobel')
    axs[0, 3].bar(sb_bins[:-1], target_sb_hist, width=sb_bins[1] - sb_bins[0])
    axs[1, 3].bar(sb_bins[:-1], sb_hist, width=sb_bins[1] - sb_bins[0])

    # for ii in range(2):
    #     for jj in range(1, 4):
    #         axs[ii, jj].tick_params(left=False, right=False, labelleft=False, 
    #                                 labelbottom=False, bottom=False) 
            
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hist_comparison.png'), dpi=200)
    if show:
        plt.show()
    plt.close()