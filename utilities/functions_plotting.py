import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import json
from typing import Dict, List, Optional, Any
import logging

PLT_CONFIG = {"marker": {}
}
              

def plot_os_2D(mo_obj_val: np.array,
            filepath: str,
            loss_functions: List[str] = [],
            axis_prop: Optional[dict] = None,
            ) -> None:

    """
    assuming mo_obj_val to be n_obj * n_sol
    """
    cmap = plt.cm.get_cmap('tab20', 20)
    fig = plt.figure(dpi=100)

    ax = fig.add_subplot(1,1,1)
    n_obj, n_sol = mo_obj_val.shape

    for i_sol in range(n_sol):
        ax.scatter(mo_obj_val[0,i_sol], mo_obj_val[1,i_sol], 
                    color=cmap(i_sol),
                    label=f"Pred{i_sol}")
            
    ax.legend(loc="upper center", ncols=n_sol, bbox_to_anchor=(-0.5, 0.15, 2, 1))

    if axis_prop is not None:
        max_vals = axis_prop.get("max", [1, 1, 1])
        ax.set_xlim(0, max_vals[0])
        ax.set_ylim(0, max_vals[1])
        axis_limit = [(0, max_vals[0]), (0, max_vals[1])]
    else:
        axis_limits = [ax.get_xlim(), ax.get_ylim()]

    ax.set_xlabel(loss_functions[0])
    ax.set_ylabel(loss_functions[1])

    plt.subplots_adjust(wspace=0.2, left=0.1, right=0.95)
    plt.savefig(filepath)
    plt.close(fig)


def plot_os_3D(mo_obj_val: np.array,
            filepath: str,
            loss_functions: List[str] = [],
            axis_prop: Optional[dict] = None,
            ) -> None:

    """
    assuming mo_obj_val to be n_obj * n_sol
    """
    cmap = plt.cm.get_cmap('tab20', 20)
    fig = plt.figure(figsize=(15,5), dpi=100)
    view_list = [(30, -60, 0),
                (30, -30, 0),
                (60, -60, 0)] #elevation, azimuth, and roll

    for i, view_angles in enumerate(view_list):
        elev, azim, roll = view_angles
        
        ax = fig.add_subplot(1,3,i+1, projection='3d')
        n_obj, n_sol = mo_obj_val.shape
        if i==1:
            for i_sol in range(n_sol):
                ax.scatter(mo_obj_val[0,i_sol], mo_obj_val[1,i_sol], mo_obj_val[2,i_sol], 
                            color=cmap(i_sol),
                            label=f"Pred{i_sol}")
                    
            ax.legend(loc="upper center", ncols=n_sol, bbox_to_anchor=(-0.5, 0.15, 2, 1))
        else:
            for i_sol in range(n_sol):
                ax.scatter(mo_obj_val[0,i_sol], mo_obj_val[1,i_sol], mo_obj_val[2,i_sol], 
                        color=cmap(i_sol))
        
        if axis_prop is not None:
            max_vals = axis_prop.get("max", [1, 1, 1])
            ax.set_xlim(0, max_vals[0])
            ax.set_ylim(0, max_vals[1])
            ax.set_zlim(0, max_vals[2])
            axis_limit = [(0, max_vals[0]), (0, max_vals[1]), (0, max_vals[2])]
        else:
            axis_limits = [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]
    
        # # inner corners
        # ax.plot(axis_limits[0], [0, 0], [0, 0], c='k')
        # ax.plot([0, 0], axis_limits[1], [0, 0], c='k')
        # ax.plot([0, 0], [0, 0], axis_limits[2],  c='k')
        ax.invert_yaxis()
        ax.set_xlabel(loss_functions[0])
        ax.set_ylabel(loss_functions[1])
        ax.set_zlabel(loss_functions[2])
        ax.view_init(elev, azim, roll)
    
    plt.subplots_adjust(wspace=0.2, left=0.05, right=0.95)
    plt.savefig(filepath)
    plt.close(fig)


def save_os_visualization(mo_obj_val: np.array, 
            out_dir: str,
            iter: int,
            loss_functions: List[str] = [],
            ref_point: List[float] = [1, 1, 1]
            ) -> None:

    """
    assuming mo_obj_val to be n_obj * n_sol
    """
    nobj = mo_obj_val.shape[1]
    if nobj==2:
        plt_fn = plot_os_2D
    elif nobj==3:
        plt_fn = plot_os_3D
    else:
        logging.error(f"visualization not implemented for nobj={nobj}.")

    ndim = mo_obj_val.ndim
    if ndim == 3: # assuming first axis to be samples
        # axis_maxs = 1.01*np.max(mo_obj_val, axis=(0,2))
        axis_prop = {"max": ref_point}
        for i_sample in range(mo_obj_val.shape[0]):
            filepath = os.path.join(out_dir, f"pareto_front{iter}_im{i_sample}.png")
            plt_fn(mo_obj_val[i_sample], filepath, loss_functions=loss_functions)
    elif ndim==2: # assuming n_obj * n_sol
        if len(loss_functions)==0:
            loss_functions = [f"Loss{i}" for i in range(mo_obj_val.shape[0])]
        else:
            assert len(loss_functions)==mo_obj_val.shape[0]

        # axis_prop = {"max": ref_point}
        filepath = os.path.join(out_dir, f"pareto_front{iter}_mean.png")
        plt_fn(mo_obj_val, filepath, loss_functions=loss_functions)
    else:
        logging.warning(f"mo_obj_val has {ndim} dimensions."\
            "don't know what that means.")
    

