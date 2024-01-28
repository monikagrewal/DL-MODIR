import os
import numpy as np
import matplotlib
# %matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage
import json
from typing import List, Optional
from sklearn.metrics import confusion_matrix
import cv2
from skimage.measure import find_contours
from skimage.morphology import disk
from scipy.ndimage import median_filter

import sys
sys.path.append("../")
from functions.functions_evaluation import compute_hv_in_higher_dimensions

plt.rcParams['text.usetex'] = True

plt_config = {"title": {"fontweight":"bold", "fontsize":14},
              "axis": {"fontweight":"normal", "fontsize":12},
              }

def mask_to_contour(mask:np.array, mode:str="outer", nclasses:int=5) -> dict:
	assert mask.ndim==2

	all_contours = {}
	for class_no in range(1, nclasses):
		mask_class = mask==class_no
		mask_class = skimage.filters.median(mask_class, disk(5))
		if mask_class.sum()>=10:
			# print("found auto contours for {}".format(class_no))
			contours = find_contours(mask_class.astype(np.float32), fully_connected='low', level=0.99)
			# note: contours is numner of contours * number of coordinates * (y, x)
			all_contours[class_no] = contours
	
	return all_contours



def plot_contours(ax, all_contours, colors, linestyle="-"):
    classes_to_pick = [2, 3]
    for iclass in classes_to_pick:
        contours = all_contours.get(iclass, None)
        if contours is not None:
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color=colors[iclass], linestyle=linestyle)
    return ax


data_root_path = "/export/scratch2/data/grewal/Data/Projects_JPG_data/MO_DIR/LUMC_cervical_test_annotated/preprocessed"

# filepath = "../runs/prelim_experiments/ls_khead_vm_learning_08082023_105726/fold0/run0/test/im0.npz"
# filepath = "../runs/prelim_experiments/weighted_hv_khead_vm_learning_gamma101_19082023_163349/fold0/run0/test/im0.npz"
filepath = "../runs/nsol25/hv_3obj_ref111_flow_init_sigma1_15112023_144512/fold0/run0/test/im0.npz"
data = np.load(filepath)

pts_filepath = os.path.join(data_root_path, "000_Fixed_points.json")
pts_names = json.load(open(pts_filepath, "r"))["names"]
print(pts_names)

# load all files
target_scan = data["target_image"]
source_scan = data["source_image"]
target_scan_seg = data["target_seg"]
source_scan_seg = data["source_seg"]
transformed_source_scans = data["transformed_source_images"]
transformed_source_segs = data["transformed_source_seg"]
dvfs_scan = data["dvfs"]
target_pts = data["target_pts"]
source_pts = data["source_pts"]
transformed_target_pts = data["transformed_target_pts"]
losses = np.squeeze(data["losses"])

nsols, nobj = losses.shape
nslices = dvfs_scan.shape[2]

# select a slice
slice_no = 60
target_image = target_scan[0, 0, slice_no, :, :]
source_image = source_scan[0, 0, slice_no, :, :]
warped_images = transformed_source_scans[:, 0, 0, slice_no, :, :]
dvfs = dvfs_scan[:, 0, slice_no, :, :, :]

target_seg = target_scan_seg[0, slice_no, :, :]
source_seg = source_scan_seg[0, slice_no, :, :]
warped_segs = transformed_source_segs[:, 0, slice_no, :, :]

# heuristic to select isols
sort_indices = np.argsort(losses[:, 0])
isol0 = np.argsort(losses[:, 1])[0]  #best deformationloss
isol1 = np.argsort(losses[:, 2])[0]  #best segloss
isol2 = np.argsort(losses[:, 0])[6]  #medium segloss
isol3 = np.argsort(losses[:, 0])[0]  #best nccloss
isols = [isol0, isol1, isol2, isol3]
plt_numbers = ["c", "d", "e", "f"]

# Set up transparency and colormap parameters
alpha = 0.5  # Transparency level for overlay
cmap = 'coolwarm'  # Colormap for highlighting differences

# Create a new figure
pf_cols = 3
fig, ax = plt.subplots(2, len(isols)+2+pf_cols, figsize=(15, 4), dpi=300, sharey=False, sharex=True)
plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0.02, right=0.98, bottom=0.02, top=0.95)

# merge subplots for pf
gs = ax[0,0].get_gridspec()
for axes in ax[:, :pf_cols-1].flatten():
    axes.remove()
axbig = fig.add_subplot(gs[:,:pf_cols-1], projection='3d')

## Plot Pareto front single view
# ---------- Pareto front --------------
view_angles = (70, -60, 0) #elevation, azimuth, and roll
loss_functions = [r'$\displaystyle\sum_{n=1}^\infty', "Deformation Loss", "SegSimilarity Loss"]
colors = ["red", "green", "orange", "magenta"]
contour_colors = ["black", "red", "cyan", "purple", "yellow"]
isol_colors = dict(zip(isols, colors))
all_colors = [isol_colors.get(idx, "blue") for idx in range(losses.shape[0])]

elev, azim, roll = view_angles
axbig.scatter(losses[:,0], losses[:,1], losses[:,2],
        color=all_colors, s=60, zorder=4)
axbig.set_xlabel(loss_functions[0], fontsize=12, fontweight="normal")
axbig.set_ylabel(loss_functions[1], fontsize=12, fontweight="normal")
axbig.set_zlabel(loss_functions[2], fontsize=12, fontweight="normal")
axbig.invert_yaxis()
axbig.grid(visible=True, color=(0.98, 0.98, 0.98), alpha=0.5, linestyle="-", zorder=0)
axbig.view_init(elev, azim, roll)
axbig.set_title("(a) Approximation Front", **plt_config["axis"])

# padding axes
ax[0, pf_cols-1].set_axis_off()
ax[1, pf_cols-1].set_axis_off()

# Display the target image
ax[0, pf_cols+0].imshow(target_image, cmap='gray')
target_contours = mask_to_contour(target_seg)
ax[0, pf_cols+0] = plot_contours(ax[0, pf_cols+0], target_contours, contour_colors, linestyle="-")
ax[0, pf_cols+0].set_title('(b) Target Image', **plt_config["axis"])
ax[0, pf_cols+0].set_axis_off()


# Display the source image
ax[0, -1].imshow(source_image, cmap='gray')
all_contours = mask_to_contour(source_seg)
ax[0, -1] = plot_contours(ax[0, -1], all_contours, contour_colors, linestyle="dotted")
ax[0, -1].set_title('(g) Source Image', **plt_config["axis"])
ax[0, -1].set_axis_off()


ax[1, pf_cols+0].set_axis_off()
# ax[2, 0].set_axis_off()
ax[1, -1].set_axis_off()
# ax[2, -1].set_axis_off()

# settings for dvf display
shape = target_image.shape
nsteps = 56
step_size = shape[0]//nsteps
x, y = np.meshgrid(np.linspace(0, shape[1], nsteps),
                    np.linspace(0, shape[0], nsteps))

for idx, isol in enumerate(isols):
    warped_image = warped_images[isol, :, :]
    # Display the warped image
    ax[0, pf_cols+1+idx].imshow(warped_image, cmap='gray')
    # dispaly seg contours
    all_contours = mask_to_contour(warped_segs[isol, :, :])
    ax[0, pf_cols+1+idx] = plot_contours(ax[0, pf_cols+1+idx], target_contours, contour_colors, linestyle="-")
    ax[0, pf_cols+1+idx] = plot_contours(ax[0, pf_cols+1+idx], all_contours, contour_colors, linestyle="dotted")

    plt_number = plt_numbers[idx]
    ax[0, pf_cols+1+idx].set_title(f"({plt_number}) Output {idx+1}", **plt_config["axis"])
    ax[0, pf_cols+1+idx].set_axis_off()

    # # Display the difference image
    # difference_image = target_image - warped_image
    # ax[1, 1+idx].imshow(difference_image, cmap=cmap)
    # ax[1, 1+idx].set_axis_off()

    # # Display the colorbar to indicate the colormap scale
    # cbar = plt.colorbar(location="bottom", shrink=0.6)
    # cbar.set_label('Difference Magnitude')

    # display DVF
    ax[1, pf_cols+1+idx].imshow(source_image, cmap='gray')
    dvf = (-1) * dvfs[isol, :, :, :]
    u = dvf[::step_size, ::step_size, 0]
    v = dvf[::step_size, ::step_size, 1]
    w = dvf[::step_size, ::step_size, 2]
    dvf_colors = matplotlib.colormaps['coolwarm'](w.reshape(-1))
    ax[1, pf_cols+1+idx].quiver(x, y, u, v, width=0.8, scale=1, units='xy', angles='xy', color=dvf_colors)
    ax[1, pf_cols+1+idx].set_axis_off()

# draw frames
rects = [plt.Rectangle((2006, 86), 452, 1004, facecolor="none", linewidth=3,
                        edgecolor=colors[0], zorder=2),
        plt.Rectangle((2488, 86), 452, 1004, facecolor="none", linewidth=3,
                        edgecolor=colors[1], zorder=2),
        plt.Rectangle((2970, 86), 452, 1004, facecolor="none", linewidth=3,
                        edgecolor=colors[2], zorder=2),
        plt.Rectangle((3452, 86), 452, 1004, facecolor="none", linewidth=3,
                        edgecolor=colors[3], zorder=2),
            ]
fig.patches.extend(rects)

# fig.savefig("../outputs/demonstrate_try.png")
# plt.show()