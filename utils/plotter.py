"""
Used to draw R-D curves

data format:
bpp, ms-ssim(db), psnr(db)

"""
import os
import argparse
import numpy as np

from fjcommon import functools_ext as ft
from fjcommon.iterable_ext import flag_first as flag_first_iter
import matplotlib as mpl
mpl.rcParams['text.latex.unicode'] = True
mpl.use('Agg')  # No display
from matplotlib import pyplot as plt

import constants
from codes_distance import get_interpolated_values_bpg_jp2k, get_measures_readers, interpolate_ours, \
    DEFAULT_BPP_GRID, CODECS

plt.rcParams['font.sans-serif'] = ['SimHei'] # # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# mssim_path
BPG420_kodak_pth = "../results/image_compression_baselines/kodak/MS-SSIM_sRGB_RGB/bpg420.txt"
BPG444_kodak_pth = "../results/image_compression_baselines/kodak/MS-SSIM_sRGB_RGB/bpg444.txt"
JP2K_kodak_pth = "../results/image_compression_baselines/kodak/MS-SSIM_sRGB_RGB/j2k-kdu5.txt"
Jpeg_kodak_pth = "../results/image_compression_baselines/kodak/MS-SSIM_sRGB_RGB/jpeg420.txt"
WebP_kodak_pth = "../results/image_compression_baselines/kodak/MS-SSIM_sRGB_RGB/webp.txt"

balle_2018_opt_mse_pth = "../results/image_compression_baselines/kodak/MS-SSIM_sRGB_RGB/balle-2018-iclr-opt-mse.txt"


# baseline_GMM_pth = "../results/our_proposed/kodak/proposed_GMM_opt_mse.txt"
Proposed_pth = "../results/our_proposed/kodak/proposed_opt_mse_msssim.txt"

# ----------------------------
# psnr_path
# BPG420_kodak_pth = "../results/image_compression_baselines/kodak/PSNR_sRGB_RGB/bpg420.txt"
# BPG444_kodak_pth = "../results/image_compression_baselines/kodak/PSNR_sRGB_RGB/bpg444.txt"
# JP2K_kodak_pth = "../results/image_compression_baselines/kodak/PSNR_sRGB_RGB/j2k-kdu5.txt"
# Jpeg_kodak_pth = "../results/image_compression_baselines/kodak/PSNR_sRGB_RGB/jpeg420.txt"
# WebP_kodak_pth = "../results/image_compression_baselines/kodak/PSNR_sRGB_RGB/webp.txt"
#
# balle_2018_opt_mse_pth = "../results/image_compression_baselines/kodak/PSNR_sRGB_RGB/balle-2018-iclr-opt-mse.txt"
#
# Proposed_pth = "../results/our_proposed/kodak/proposed_opt_mse_psnr.txt"

# ----------------------------
# ablation_study
# baseline_pth = '../results/our_proposed/Ablation_study/baseline.txt'
# baseline_GMM_pth = '../results/our_proposed/Ablation_study/baseline+GMM.txt'
# baseline_GMM_CAM_pth = '../results/our_proposed/Ablation_study/baseline+GMM+attention.txt'

LABEL_BPG = 'BPG'
LABEL_JP2K = 'JPEG2000'
LABEL_JP = 'JPEG'
LABEL_WEBP = 'WebP'
LABEL_BPG_420 = "BPG(4:2:0)"
LABEL_BPG_444 = "BPG(4:4:4)"
LABEL_BALLE_MSE = 'Ballé[MSE][ICLR18]'
LABEL_BALLE_MSSSIM = 'Ballé[MS-SSIM][ICLR18]'
LABEL_proposed_GMM_opt_mse = 'Baseline with GMM'
LABEL_proposed_opt_mse = '本文方法'
LABEL_baseline = 'Baseline'
LABEL_baseline_GMM = 'Baseline+GMM'
LABEL_baseline_GMM_GCAM = 'Baseline+GMM+GCAM'

TITLES = {'u100': 'Urban100',
          'b100': 'B100',
          'rf100': 'ImageNetVal',
          'kodak': 'Kodak',
          'testset': 'TestSet'}

def read_resultTXT(file_pth):
	out = []
	with open(file_pth, 'r') as fr:
		for line in fr.readlines():
			if line[0].isalnum() is True:
				line = line.strip().split(',')
				out.append([float(ch) for ch in line])
			else:
				continue
	return out



def get_label_from_codec_short_name(codec_short_name):
    # 'bpg': LABEL_BPG,
    # 'jp2k': LABEL_JP2K,
    # 'jp': LABEL_JP,
    # 'webp': LABEL_WEBP,
    return {
            'bpg420':   LABEL_BPG_420,
            'bpg444':   LABEL_BPG_444,
            'j2k-kdu5': LABEL_JP2K,
            'jpeg420':  LABEL_JP,
            'webp':     LABEL_WEBP,
            'balle-2018-iclr-opt-mse':         LABEL_BALLE_MSE,
            'balle-2018-iclr-opt-msssim':      LABEL_BALLE_MSSSIM,
            'proposed_GMM_opt_mse':            LABEL_proposed_GMM_opt_mse,
            'proposed_opt_mse_msssim':         LABEL_proposed_opt_mse,
            'proposed_opt_mse_psnr':           LABEL_proposed_opt_mse,
            'baseline':                        LABEL_baseline,
            'baseline+GMM':                    LABEL_baseline_GMM,
            'baseline+GMM+attention':          LABEL_baseline_GMM_GCAM}[codec_short_name]

# style = {
#         LABEL_BPG_420: (cmap(0.7), '-', 1.5)
#         # LABEL_OURS: ('0', '-', 3),
#         # LABEL_RB: (cmap(0.9), '-', 1.5),
#         # LABEL_BPG: (cmap(0.7), '-', 1.5),
#         # LABEL_JP2K: (cmap(0.45), '-', 1.5),
#         # LABEL_JP: (cmap(0.2), '-', 1.5),
#         # LABEL_WEBP: (cmap(0.6), '-', 1.5),
#         # LABEL_JOHNSTON: (cmap(0.7), '--', 1.5),
#         # LABEL_BALLE: (cmap(0.45), '--', 1.5),
#         # LABEL_THEIS: (cmap(0.2), '--', 1.5),
#         # LABEL_Our: (cmap(0.2), '-', 1.5)
#     }


def plot_(measure_pth, style, evaluate='ms-ssim'):
    label = os.path.basename(measure_pth).split('.')[0] # get filename without extension name
    label = get_label_from_codec_short_name(label)
    col, line_style, line_width, marker = style[label]
    dashes = (5, 1) if line_style == '--' else [] # i do not know that dashes means
    
    metric = np.array(read_resultTXT(measure_pth))
    this_bpp, this_metric = metric[:, 0], metric[:, 1]
    # ms_ssim_db = -10 * log10(1 - ms_ssim)
    if evaluate == 'ms-ssim':
        this_metric = -10 * np.log10(1 - this_metric)
    plt.plot(this_bpp, this_metric, label=label, linewidth=line_width, color=col, dashes=dashes, marker=marker, ms=3)
    # plt.savefig('test_tmp.png')
#
# def plot_dot():
# 	pass

def interpolate_curve(x_range, y_range, output_path=None):
    if not output_path:
        output_path = 'tmp.png'

    plt.figure(figsize=(6, 6))
    # cmap = plt.cm.get_cmap('cool')
    cmap = plt.cm.get_cmap('tab20')

# -------------------------------------------
    # color, line_style, line_width, marker
    style = {
        LABEL_BPG_420: (cmap(1),  '-', 1.5, ''),
        LABEL_BPG_444: (cmap(2),  '-', 1.5, ''),
        LABEL_JP2K:    (cmap(3), '-', 1.5, ''),
        LABEL_JP:      (cmap(4),  '-', 1.5, ''),
        LABEL_WEBP:    (cmap(5),  '-', 1.5, ''),
        LABEL_BALLE_MSE: (cmap(6), '--', 1.5, ''),
        LABEL_BALLE_MSSSIM: (cmap(7), '--', 1.5, ''),
        LABEL_proposed_GMM_opt_mse: (cmap(8), '-', 1.5, 'o'),
        LABEL_proposed_opt_mse: (cmap(9), '-', 1.5, 'o'),
        LABEL_baseline: (cmap(10), '-', 1.5, 'o'),
        LABEL_baseline_GMM: (cmap(11), '-', 1.5, 'o'),
        LABEL_baseline_GMM_GCAM: (cmap(12), '-', 1.5, 'o')
    }
    
    pos = {
        LABEL_BPG_420: 1,
        LABEL_BPG_444: 2,
        LABEL_JP2K:    3,
        LABEL_JP:      4,
        LABEL_WEBP:    5,
        LABEL_BALLE_MSE: 6,
        LABEL_BALLE_MSSSIM: 7,
        LABEL_proposed_GMM_opt_mse: 8,
        LABEL_proposed_opt_mse: 9,
        LABEL_baseline: 10,
        LABEL_baseline_GMM: 11,
        LABEL_baseline_GMM_GCAM: 12
    }
    
    plt.figure(figsize=(6, 6))


    # BPG(4:2:0)
    plot_(BPG420_kodak_pth, style)
    # BPG(4:4:0)
    plot_(BPG444_kodak_pth, style)
    # JP2k
    plot_(JP2K_kodak_pth, style)
    # JPEG
    plot_(Jpeg_kodak_pth, style)
    # WebP
    plot_(WebP_kodak_pth, style)
    # Balle 2018 opt mse
    plot_(balle_2018_opt_mse_pth, style)
    # Balle 2018 opt msssim
    # plot_(balle_2018_msssim_opt_msssim_pth, style)
    
    # plot_(baseline_GMM_pth, style)
    plot_(Proposed_pth, style)
    
    # ablation study
    # plot_(baseline_pth, style, evaluate=None)
    # plot_(baseline_GMM_pth, style, evaluate=None)
    # plot_(baseline_GMM_CAM_pth, style, evaluate=None)
    
    # plt.title('{} on {}'.format('ms-ssim'.upper(), 'Kodak'))
    # plt.title('{} on {}'.format('psnr'.upper() + '(dB)', 'Kodak'))
    plt.xlabel('码率/bpp', labelpad=-1)
    # plt.ylabel('{} on {}'.format('ms-ssim'.upper() + '(dB)', 'Kodak'))
    plt.ylabel('{}/{}'.format('ms-ssim'.upper(),'dB'))
    plt.grid()
    
    
# ------------------------------------------------

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), reverse=True, key=lambda t: pos[t[0]]))
    ax.legend(handles, labels, loc=4, prop={'size': 12}, fancybox=True, framealpha=0.7)

    ax.yaxis.grid(b=True, which='both', color='0.8', linestyle='-')
    ax.xaxis.grid(b=True, which='major', color='0.8', linestyle='-')
    ax.set_axisbelow(True)

    ax.minorticks_on()
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))

    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.xticks(np.arange(x_range[0], x_range[1], 0.1)) # 0.1
    plt.yticks(np.arange(y_range[0], y_range[1], 2)) # 2
    print('Saving {}...'.format(output_path))
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log_dir_root', help='Path to dir containing log_dirs.')
    p.add_argument('--job_ids', help='Comma separated list of job_ids.')
    p.add_argument('images')
    p.add_argument('--x_range', default='0,1.2')
    p.add_argument('--y_range', default='0.85,1.0')
    p.add_argument('--latex', action='store_true')
    p.add_argument('--output_path', '-o', help='Path to store plot. Defaults to plot_DATASET.png.')
    p.add_argument('--style', nargs='+', default=['interp'], choices=['interp', 'mean'])
    p.add_argument('--paper_plot', action='store_true')
    p.add_argument('--ids', help='If given with --style mean, label mean points with these ids.',
                   nargs='+')
    flags = p.parse_args()

    range_to_floats = lambda r: tuple(map(float, r.split(',')))
    interpolated_curve(flags.log_dir_root, flags.job_ids, flags.images,
                       DEFAULT_BPP_GRID, 'quadratic',
                       plot_interp_of_ours='interp' in flags.style,
                       plot_mean_of_ours='mean' in flags.style,
                       plot_ids_of_ours=flags.ids,
                       metric='ms-ssim',
                       x_range=range_to_floats(flags.x_range), y_range=range_to_floats(flags.y_range),
                       use_latex=flags.latex,
                       output_path=flags.output_path,
                       paper_plot=flags.paper_plot)


if __name__ == '__main__':
    # psnr x : 0.1, 1.7
    # ms-ssim x :0.1, 1.8
    interpolate_curve(x_range=(0.1, 1.8), y_range=(10, 26)) # (10, 26) for msssim, (20, 43) for psnr
    # main()