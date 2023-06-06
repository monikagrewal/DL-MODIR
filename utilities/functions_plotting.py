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

def plot_os_3D(mo_obj_val: np.array,
            filepath: str,
            loss_functions: List[str] = [],
            axis_prop: Optional[dict] = None,
            ) -> None:

    """
    assuming mo_obj_val to be n_obj * n_sol
    """
    cmap = plt.cm.get_cmap('tab20', 20)
    fig = plt.figure(figsize=(10,10), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    if ax is None:
        fig = plt.figure(figsize=(10,10), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

    n_obj, n_sol = mo_obj_val.shape
    for i_sol in range(n_sol):
        ax.scatter(mo_obj_val[0,i_sol], mo_obj_val[1,i_sol], mo_obj_val[2,i_sol], 
                   color=cmap(i_sol),
                   label=f"Pred{i_sol}",
                    **PLT_CONFIG["marker"])
            
    ax.legend(loc="upper center", ncols=n_sol)
    if axis_prop is not None:
        max_vals = axis_prop.get("max", [1, 1, 1])
        ax.set_xlim(0, max_vals[0])
        ax.set_ylim(0, max_vals[1])
        ax.set_zlim(0, max_vals[2])
    ax.set_xlabel(loss_functions[0])
    ax.set_ylabel(loss_functions[1])
    ax.set_zlabel(loss_functions[2])
    ax.invert_yaxis()
    ax.set_title("Pareto front approximation")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close(fig)


def save_os_visualization(mo_obj_val: np.array, 
            cache: Any,
            loss_functions: List[str] = [],
            ) -> None:

    """
    assuming mo_obj_val to be n_obj * n_sol
    """
    ndim = mo_obj_val.ndim
    if ndim == 3: # assuming first axis to be samples
        axis_maxs = 1.01*np.max(mo_obj_val, axis=(0,2))
        axis_prop = {"max": axis_maxs}
        for i_sample in range(mo_obj_val.shape[0]):
            filepath = os.path.join(cache.out_dir_val, f"pareto_front_im{i_sample}.png")
            plot_os_3D(mo_obj_val[i_sample], filepath, loss_functions=loss_functions, axis_prop=axis_prop)
    elif ndim==2: # assuming n_obj * n_sol
        if len(loss_functions)==0:
            loss_functions = [f"Loss{i}" for i in range(mo_obj_val.shape[0])]
        else:
            assert len(loss_functions)==mo_obj_val.shape[0]

        axis_prop = {"max": [1, 1, 1]}
        filepath = os.path.join(cache.out_dir_val, f"pareto_front_iter{cache.iter}.png")
        plot_os_3D(mo_obj_val, filepath, loss_functions=loss_functions, axis_prop=axis_prop)
    else:
        logging.warning(f"mo_obj_val has {ndim} dimensions."\
            "don't know what that means.")
    

