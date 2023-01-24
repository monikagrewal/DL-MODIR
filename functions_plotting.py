import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import json



def set_plotting_details(mo_obj_val,mo_obj_val_per_sample):
    ''' set axis limits to around the observed loss values'''
    ## for plots of means
    axis_min_list = list()
    axis_max_list = list()
    n_mo_obj = mo_obj_val.shape[0]
    for i_label in range(0,n_mo_obj):
        cur_obj_max_val = np.max(mo_obj_val[i_label,:])
        axis_min_list.append(-0.01 * cur_obj_max_val)
        axis_max_list.append(1.01 * cur_obj_max_val)
    ## for plots per sample
    axis_min_list_per_sample = list()
    axis_max_list_per_sample = list()
    for i_label in range(0,n_mo_obj):
        cur_obj_max_val = np.max(mo_obj_val_per_sample[:,i_label,:])
        axis_min_list_per_sample.append(-0.01 * cur_obj_max_val)
        axis_max_list_per_sample.append(1.01 * cur_obj_max_val)
    return(axis_min_list,axis_max_list,axis_min_list_per_sample,axis_max_list_per_sample)

def plot_selected_iterations(statistics_writer,cfg,output_folder_name,extensive_plotting):
    """
    plot the NNs' positions in objective space for the specific iterations 
    (first, last, iter with best training HV, iter with best validation HV, many intermediate iterations if extensive_plotting == True)
    """
    # # sort losses for better visualization
    # indices = np.argsort(mo_obj_val_per_sample[:,0,:], axis=-1)
    # tmp = []
    # for i, idx in enumerate(indices):
    #     tmp.append(mo_obj_val_per_sample[i, :, idx].T)  #I know this is wierd, but it's correct
    # mo_obj_val_per_sample = np.array(tmp)
    # mo_obj_val_mean = np.mean(mo_obj_val_per_sample, axis=0)
    
    assert ((statistics_writer.n_mo_obj == 2) or (statistics_writer.n_mo_obj == 3))
    mo_obj_val = statistics_writer.mo_obj_val_list
    mo_obj_val_per_sample_list = statistics_writer.mo_obj_val_per_sample_list

    max_iter = len(mo_obj_val)
    n_mo_obj = statistics_writer.n_mo_obj
    ref_point = statistics_writer.ref_point
    # create list of labels
    label_list = list()
    for i_label in range(0,n_mo_obj):
        label_list.append(cfg["obj_func"][i_label])
    pareto_front = None

    # first iter
    cur_iter = 0
    title = 'Training front, first iteration ' + str(cur_iter)
    image_file_name = 'training_front_first_iter_' + str(cur_iter)
    image_file_name_per_sample = image_file_name+'_per_sample'
    # set axis limits to around the observed loss values
    axis_min_list,axis_max_list,axis_min_list_per_sample,axis_max_list_per_sample = set_plotting_details(mo_obj_val[cur_iter],mo_obj_val_per_sample_list[cur_iter])
    if n_mo_obj == 2:
        plot_os_single_2D(mo_obj_val[cur_iter],pareto_front,axis_min_list,axis_max_list,label_list,title,output_folder_name,image_file_name)
        plot_os_single_per_sample_2D(mo_obj_val[cur_iter],mo_obj_val_per_sample_list[cur_iter],pareto_front,axis_min_list_per_sample,axis_max_list_per_sample,label_list,title,output_folder_name,image_file_name_per_sample)
    elif n_mo_obj == 3:
        plot_os_single_3D(mo_obj_val[cur_iter],pareto_front,axis_min_list,axis_max_list,label_list,title,output_folder_name,image_file_name)

    # last iter
    cur_iter = max_iter-1 # length of list - 1
    title = 'Training front, last iteration ' + str(cur_iter)  
    image_file_name = 'training_front_last_iter_' + str(cur_iter)
    image_file_name_per_sample = image_file_name+'_per_sample'
    # set axis limits to around the observed loss values
    axis_min_list,axis_max_list,axis_min_list_per_sample,axis_max_list_per_sample = set_plotting_details(mo_obj_val[cur_iter],mo_obj_val_per_sample_list[cur_iter])
    if n_mo_obj == 2:
        plot_os_single_2D(mo_obj_val[cur_iter],pareto_front,axis_min_list,axis_max_list,label_list,title,output_folder_name,image_file_name)
        plot_os_single_per_sample_2D(mo_obj_val[cur_iter],mo_obj_val_per_sample_list[cur_iter],pareto_front,axis_min_list_per_sample,axis_max_list_per_sample,label_list,title,output_folder_name,image_file_name_per_sample)
    elif n_mo_obj == 3:
        plot_os_single_3D(mo_obj_val[cur_iter],pareto_front,axis_min_list,axis_max_list,label_list,title,output_folder_name,image_file_name)

    # validation front at last iter
    if not (statistics_writer.best_validation_iter == -np.inf):
        cur_iter = max_iter-1
        title = 'Validation front, last iteration ' + str(cur_iter)
        image_file_name = 'validation_front_last_iter_' + str(cur_iter)
        image_file_name_per_sample = image_file_name+'_per_sample'
        validation_mo_obj_val = statistics_writer.validation_mo_obj_val_list[-1]
        validation_mo_obj_val_per_sample = statistics_writer.validation_mo_obj_val_per_sample_list[-1]
        # set axis limits to around the observed loss values
        axis_min_list,axis_max_list,axis_min_list_per_sample,axis_max_list_per_sample = set_plotting_details(validation_mo_obj_val, validation_mo_obj_val_per_sample)
        if n_mo_obj == 2:
            plot_os_single_2D(validation_mo_obj_val,pareto_front,axis_min_list,axis_max_list,label_list,title,output_folder_name,image_file_name)
            plot_os_single_per_sample_2D(validation_mo_obj_val,validation_mo_obj_val_per_sample,pareto_front,axis_min_list_per_sample,axis_max_list_per_sample,label_list,title,output_folder_name,image_file_name_per_sample)
        elif n_mo_obj == 3:
            plot_os_single_3D(validation_mo_obj_val,pareto_front,axis_min_list,axis_max_list,label_list,title,output_folder_name,image_file_name)


    # best training iter
    cur_iter = statistics_writer.best_iter
    title = 'Training front with best training HV, iteration ' + str(cur_iter)
    image_file_name = 'training_front_best_training_hv_iter_' + str(cur_iter)
    image_file_name_per_sample = image_file_name+'_per_sample'
    # set axis limits to around the observed loss values
    axis_min_list,axis_max_list,axis_min_list_per_sample,axis_max_list_per_sample = set_plotting_details(mo_obj_val[cur_iter],mo_obj_val_per_sample_list[cur_iter])
    if n_mo_obj == 2:
        plot_os_single_2D(mo_obj_val[cur_iter],pareto_front,axis_min_list,axis_max_list,label_list,title,output_folder_name,image_file_name)
        plot_os_single_per_sample_2D(mo_obj_val[cur_iter],mo_obj_val_per_sample_list[cur_iter],pareto_front,axis_min_list_per_sample,axis_max_list_per_sample,label_list,title,output_folder_name,image_file_name_per_sample)
    elif n_mo_obj == 3:
        plot_os_single_3D(mo_obj_val[cur_iter],pareto_front,axis_min_list,axis_max_list,label_list,title,output_folder_name,image_file_name)

    # if it the iter is not greater than -inf, then there probably has never been an evaluation of validation performance, so skip it
    if not (statistics_writer.best_validation_iter == -np.inf):
        # training front at best validation iter
        cur_iter = statistics_writer.best_validation_iter
        title = 'Training front at iteration with best validation HV, iteration ' + str(cur_iter)
        image_file_name = 'training_front_at_best_validation_hv_iter_' + str(cur_iter)
        image_file_name_per_sample = image_file_name+'_per_sample'
        # set axis limits to around the observed loss values
        axis_min_list,axis_max_list,axis_min_list_per_sample,axis_max_list_per_sample = set_plotting_details(mo_obj_val[cur_iter],mo_obj_val_per_sample_list[cur_iter])
        if n_mo_obj == 2:
            plot_os_single_2D(mo_obj_val[cur_iter],pareto_front,axis_min_list,axis_max_list,label_list,title,output_folder_name,image_file_name)
            plot_os_single_per_sample_2D(mo_obj_val[cur_iter],mo_obj_val_per_sample_list[cur_iter],pareto_front,axis_min_list_per_sample,axis_max_list_per_sample,label_list,title,output_folder_name,image_file_name_per_sample)
        elif n_mo_obj == 3:
            plot_os_single_3D(mo_obj_val[cur_iter],pareto_front,axis_min_list,axis_max_list,label_list,title,output_folder_name,image_file_name)

        # validation front at best validation iter
        cur_iter = statistics_writer.best_validation_iter
        title = 'Validation front at iteration with best validation HV, iteration ' + str(cur_iter)
        image_file_name = 'validation_front_at_best_validation_hv_iter_' + str(cur_iter)
        image_file_name_per_sample = image_file_name+'_per_sample'
        best_validation_mo_obj_val = statistics_writer.best_validation_mo_obj_val
        best_validation_mo_obj_val_per_sample = statistics_writer.best_validation_mo_obj_val_per_sample
        # set axis limits to around the observed loss values
        axis_min_list,axis_max_list,axis_min_list_per_sample,axis_max_list_per_sample = set_plotting_details(best_validation_mo_obj_val,best_validation_mo_obj_val_per_sample)
        if n_mo_obj == 2:
            plot_os_single_2D(best_validation_mo_obj_val,pareto_front,axis_min_list,axis_max_list,label_list,title,output_folder_name,image_file_name)
            plot_os_single_per_sample_2D(best_validation_mo_obj_val,best_validation_mo_obj_val_per_sample,pareto_front,axis_min_list_per_sample,axis_max_list_per_sample,label_list,title,output_folder_name,image_file_name_per_sample)
        elif n_mo_obj == 3:
            plot_os_single_3D(best_validation_mo_obj_val,pareto_front,axis_min_list,axis_max_list,label_list,title,output_folder_name,image_file_name)

    else:
        print('best_validation_iter is -inf, probably because validation performance was never checked. Plotting of the set of solutions with best validation performance is skipped.')

    if extensive_plotting:
        ## sampled iterations
        # create array of iteration numbers for plotting
        # include the first 10 iterations	
        first_iterations = np.arange(0,10)
        # use log spacing to early drop in training loss (many samples early, fewer samples later)
        log_spacing = np.round(np.geomspace(1,(max_iter-1),5)).astype(int)
        # use linear spacing to sample evenly because it is unknown when validation error goes up
        linear_spacing = np.round(np.linspace(1,(max_iter-1),100)).astype(int)
        # combine all
        selected_iterations = np.unique(np.concatenate((first_iterations,log_spacing,linear_spacing),axis = 0))
        # truncate at max_iter-1
        selected_iterations = selected_iterations[(selected_iterations <= (max_iter-1))]
        # create subfolder
        iteration_plotting_folder = os.path.join(output_folder_name,'sampled_iterations')
        if not os.path.exists(iteration_plotting_folder):
            os.makedirs(iteration_plotting_folder)
        # set axis limits to include the entire reference region        
        axis_min_list = list()
        axis_max_list = list()
        for i_label in range(0,n_mo_obj):
            axis_min_list.append(-0.01 * ref_point[i_label])
            axis_max_list.append(1.01 * ref_point[i_label])

        # plot iterations
        for i_iter in selected_iterations:
            title = 'Training front, iteration ' + str(i_iter)
            image_file_name = 'iter_' + str(i_iter)
            plot_os_single_2D(mo_obj_val[i_iter],pareto_front,axis_min_list,axis_max_list,label_list,title,iteration_plotting_folder,image_file_name)
        


def plot_os_single_per_sample_2D(mo_obj_val,mo_obj_val_per_sample,pareto_front,axis_min_list,axis_max_list,label_list,title,output_folder_name,image_file_name):
    """
    plot the NNs' positions in objective space
    """
    # color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    n_samples = mo_obj_val_per_sample.shape[0]
    color_list = plt.cm.get_cmap('viridis', min(n_samples, 1000))
    fig,cur_ax = plt.subplots(figsize = (10,10), dpi = 100)
    line_handle_list = list()
    legend_label_list = list()
    if pareto_front is not None:
        # plot Pareto set/front as red dots
        cur_ax.scatter(pareto_front[:,0],pareto_front[:,1], facecolor = 'r',linewidth = 2.0)
    n_mo_sol = mo_obj_val.shape[1]
    # loop over each solution so that each solution gets a different color (assuming that this matches all other plots)
    # for i_mo_sol in range(0,n_mo_sol):
    #     line_handle = cur_ax.scatter(mo_obj_val[0,i_mo_sol],mo_obj_val[1,i_mo_sol], linewidth=2.0, edgecolor='black', color=color_list(i_mo_sol))
    #     legend_label = 'Net ' + str(i_mo_sol)
    #     legend_label_list.append(legend_label)
    #     line_handle_list.append(line_handle)
    #     colors = [color_list(i) for i in range(n_samples)] 
    #     # cur_ax.scatter(mo_obj_val_per_sample[:,0,i_mo_sol], mo_obj_val_per_sample[:,1,i_mo_sol], linewidth=2.0, color=color_list(i_mo_sol))
    #     cur_ax.scatter(mo_obj_val_per_sample[:,0,i_mo_sol], mo_obj_val_per_sample[:,1,i_mo_sol], linewidth=2.0, color=colors)
    for i_sample in range(0, min(n_samples, 1000)):
        xdata = mo_obj_val_per_sample[i_sample, 0, :]
        ydata = mo_obj_val_per_sample[i_sample, 1, :]
        sort_indices = np.argsort(xdata)
        xdata = xdata[sort_indices]
        ydata = ydata[sort_indices]
        color = color_list(i_sample)
        cur_ax.plot(xdata, ydata, linewidth=2.0, alpha=0.9, color=color_list(i_sample))
        
    plt.xlim(axis_min_list[0],axis_max_list[0])
    plt.ylim(axis_min_list[1],axis_max_list[1])
    plt.xlabel(label_list[0])
    plt.ylabel(label_list[1])
    plt.title(title)
    plt.legend(line_handle_list,legend_label_list,loc = 'upper left',bbox_to_anchor = (1,1))
    plt.savefig(os.path.join(output_folder_name,image_file_name +'.png'), bbox_inches = "tight")
    plt.close(fig)

def plot_os_single_per_sample(mo_obj_val,pareto_front,axis_lims,x_label,y_label,title,output_folder_name,image_file_name,mo_obj_val_per_sample):
    """
    plot the NNs' positions in objective space
    """
    # color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_list = plt.cm.get_cmap('tab20',20)
    n_samples = mo_obj_val_per_sample.shape[0]
    fig,cur_ax = plt.subplots(figsize = (10,10), dpi = 100)
    line_handle_list = list()
    legend_label_list = list()
    if pareto_front is not None:
        # plot Pareto set/front as red dots
        cur_ax.scatter(pareto_front[:,0],pareto_front[:,1], facecolor = 'r',linewidth = 2.0)
    n_mo_sol = mo_obj_val.shape[1]
    # loop over each solution so that each solution gets a different color (assuming that this matches all other plots)
    for i_mo_sol in range(0,n_mo_sol):
        line_handle = cur_ax.scatter(mo_obj_val[0,i_mo_sol],mo_obj_val[1,i_mo_sol],linewidth = 2.0,edgecolor = 'black',color = color_list(i_mo_sol))
        legend_label = 'Net ' + str(i_mo_sol)
        legend_label_list.append(legend_label)
        line_handle_list.append(line_handle)       
    for i_sample in range(0,n_samples):
        for i_mo_sol in range(0,n_mo_sol):
            cur_ax.scatter(mo_obj_val_per_sample[i_sample,0,i_mo_sol],mo_obj_val_per_sample[i_sample,1,i_mo_sol],linewidth = 2.0,color = color_list(i_mo_sol))
            
    plt.xlim(axis_lims[0],axis_lims[1])
    plt.ylim(axis_lims[2],axis_lims[3])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(line_handle_list,legend_label_list,loc = 'upper left',bbox_to_anchor = (1,1))
    plt.savefig(os.path.join(output_folder_name,image_file_name +'.png'), bbox_inches = "tight")
    plt.close(fig)

def plot_os_single_2D(mo_obj_val,pareto_front,axis_min_list,axis_max_list,label_list,title,output_folder_name,image_file_name):
    """
    plot the NNs' positions in objective space
    """
    color_list = plt.cm.get_cmap('tab20',20)
    fig,cur_ax = plt.subplots(figsize = (10,10), dpi = 100)
    line_handle_list = list()
    legend_label_list = list()
    if pareto_front is not None:
        # plot Pareto set/front as red dots
        cur_ax.scatter(pareto_front[:,0],pareto_front[:,1], facecolor = 'r',linewidth = 2.0)
    n_mo_sol = mo_obj_val.shape[1]
    # loop over each solution so that each solution gets a different color (assuming that this matches all other plots)
    for i_mo_sol in range(0,n_mo_sol):
        line_handle = cur_ax.scatter(mo_obj_val[0,i_mo_sol],mo_obj_val[1,i_mo_sol],linewidth = 2.0,color = color_list(i_mo_sol))
        legend_label = 'Net ' + str(i_mo_sol)
        legend_label_list.append(legend_label)
        line_handle_list.append(line_handle)
            
    plt.xlim(axis_min_list[0],axis_max_list[0])
    plt.ylim(axis_min_list[1],axis_max_list[1])
    plt.xlabel(label_list[0])
    plt.ylabel(label_list[1])
    plt.title(title)
    plt.legend(line_handle_list,legend_label_list,loc = 'upper left',bbox_to_anchor = (1,1))
    plt.savefig(os.path.join(output_folder_name,image_file_name +'.png'), bbox_inches = "tight")
    plt.close(fig)

def plot_os_single_3D(mo_obj_val,pareto_front,axis_min_list,axis_max_list,label_list,title,output_folder_name,image_file_name):
    """
    plot the NNs' positions in objective space
    """
    color_list = plt.cm.get_cmap('tab20',20)
    fig = plt.figure(figsize = (10,10), dpi = 100)
    cur_ax = fig.add_subplot(111, projection='3d')
    line_handle_list = list()
    legend_label_list = list()
    if pareto_front is not None:
        # plot Pareto set/front as red dots
        cur_ax.scatter(pareto_front[:,0],pareto_front[:,1],pareto_front[:,2], facecolor = 'r',linewidth = 2.0)
    n_mo_sol = mo_obj_val.shape[1]
    # loop over each solution so that each solution gets a different color (assuming that this matches all other plots)
    for i_mo_sol in range(0,n_mo_sol):
        line_handle = cur_ax.scatter(mo_obj_val[0,i_mo_sol],mo_obj_val[1,i_mo_sol],mo_obj_val[2,i_mo_sol],linewidth = 2.0,color = color_list(i_mo_sol))
        
        # add projection in loss planes
        for i_loss in range(0,3): # select planes here, range(2,3) for bottom plane
            start_point = mo_obj_val[:,i_mo_sol].copy()
            # the 2nd loss plane is at axis max, the other 2 loss planes are at axis min
            if i_loss == 1:
                start_point[i_loss] = axis_max_list[i_loss].copy()
            else:
                start_point[i_loss] = axis_min_list[i_loss].copy()
            end_point =  mo_obj_val[:,i_mo_sol].copy()
            cur_ax.scatter(start_point[0],start_point[1],start_point[2],linewidth = 1.0,c = 'w',edgecolors = color_list(i_mo_sol)) # points in loss planes
            cur_ax.plot([start_point[0],end_point[0]],[start_point[1],end_point[1]],[start_point[2],end_point[2]],linewidth = 1.0,linestyle = '--',c = 'gray') # connecting dashed lines

        legend_label = 'Net ' + str(i_mo_sol)
        legend_label_list.append(legend_label)
        line_handle_list.append(line_handle)
            
    plt.xlim(axis_min_list[0],axis_max_list[0])
    plt.ylim(axis_min_list[1],axis_max_list[1])
    cur_ax.set_zlim(axis_min_list[2],axis_max_list[2])
    plt.xlabel(label_list[0])
    plt.ylabel(label_list[1])
    cur_ax.set_zlabel(label_list[2])
    plt.title(title)
    plt.legend(line_handle_list,legend_label_list,loc = 'upper left',bbox_to_anchor = (1,1))
    plt.savefig(os.path.join(output_folder_name,image_file_name +'.png'), bbox_inches = "tight")
    plt.close(fig)

def plot_loss_weights(statistics_writer,cfg,output_folder_name):
    """
    plot the weights assigned to the NNs losses over all iterations
    """   
    weight_list = statistics_writer.dynamic_weights_list
    n_iter = len(weight_list)
    n_mo_sol = statistics_writer.n_mo_sol
    n_mo_obj = statistics_writer.n_mo_obj
    list_of_weight_ratio_lists = list()
    weight_ratios = np.zeros((n_iter,n_mo_sol))
    norm_weights = np.zeros((n_iter,n_mo_sol,n_mo_obj))
    weights = np.zeros((n_iter,n_mo_sol,n_mo_obj))
    weight_eps = 10**-100
    for i_iter in range(0,n_iter):
        for i_mo_sol in range(0,n_mo_sol):
            cur_weights = weight_list[i_iter]
            # compute ratio of w0 and w1
            # weight_ratios[i_iter,i_mo_sol] = cur_weights[0,i_mo_sol]/np.maximum(cur_weights[1,i_mo_sol],weight_eps)
            for i_mo_obj in range(0,n_mo_obj):
                # divide weights by their absolute sum
                # norm_weights[i_iter,i_mo_sol,i_mo_obj] = cur_weights[i_mo_obj,i_mo_sol]/np.sum(np.abs(cur_weights[:,i_mo_sol]))
                # just take the weights without adjustment
                weights[i_iter,i_mo_sol,i_mo_obj] = cur_weights[i_mo_obj,i_mo_sol]


    x_label = 'Iterations'
    y_label = r'$w_i$'
    title = 'Weights over iterations'
    image_file_name = 'weights_over_iterations'
    fig,cur_ax = plt.subplots(n_mo_sol,figsize = (10,10), dpi = 100)
    for i_mo_sol in range(0,n_mo_sol):
        for i_mo_obj in range(0,n_mo_obj):
            cur_ax[i_mo_sol].plot(weights[:,i_mo_sol,i_mo_obj])

        cur_ax[i_mo_sol].set_ylabel(y_label)
        cur_ax[i_mo_sol].set_ylim(top = np.max(weights)+0.1, bottom = np.min(weights)-0.1)
    cur_ax[0].set_title(title)
    cur_ax[-1].set_xlabel(x_label)
    cur_ax[0].legend(cfg["obj_func"])
    plt.savefig(os.path.join(output_folder_name,image_file_name +'.png'), bbox_inches = "tight")
    plt.close(fig)


def plot_convergence(statistics_writer_instance,cfg,output_folder_name):
    n_mo_obj = statistics_writer_instance.n_mo_obj
    n_mo_sol = statistics_writer_instance.n_mo_sol

    image_file_name = 'training_convergence_plots'
    # stats for first plot
    mean_hv_over_samples_list = statistics_writer_instance.mean_hv_over_samples_list
    best_mean_hv_over_samples = np.max(mean_hv_over_samples_list)
    best_mean_hv_over_samples_marker = np.array(mean_hv_over_samples_list)
    best_mean_hv_over_samples_marker[best_mean_hv_over_samples_marker != best_mean_hv_over_samples] = np.nan
    # stats for second plot
    hv_over_mean_losses_list = statistics_writer_instance.hv_list
    best_hv_over_mean_losses = np.max(hv_over_mean_losses_list)
    best_hv_over_mean_losses_marker = np.array(hv_over_mean_losses_list)
    best_hv_over_mean_losses_marker[best_hv_over_mean_losses_marker != best_hv_over_mean_losses] = np.nan
    # axis limit factors
    lower_limit_diff = 10**-5
    lower_limit_factor = (1-lower_limit_diff)
    upper_limit_factor = (1+ 0.1 * lower_limit_diff)
    # plotting
    fig,cur_ax = plt.subplots(2,figsize = (10,10), dpi = 100)
    cur_ax[0].plot(mean_hv_over_samples_list,zorder = 0,c = 'green')
    cur_ax[0].scatter(range(0,len(best_mean_hv_over_samples_marker)),best_mean_hv_over_samples_marker,c = 'green',edgecolor = 'black')
    cur_ax[0].set_ylabel(r'Mean $\mathrm{HV}$ over samples')
    cur_ax[0].set_yscale('log')
    cur_ax[0].set_ylim(lower_limit_factor*best_mean_hv_over_samples,upper_limit_factor * best_mean_hv_over_samples)
    cur_ax[1].plot(hv_over_mean_losses_list,zorder = 0,c = 'blue')
    cur_ax[1].scatter(range(0,len(best_hv_over_mean_losses_marker)),best_hv_over_mean_losses_marker,c = 'blue',edgecolor = 'black')
    cur_ax[1].set_ylabel(r'$\mathrm{HV}$ of mean losses')
    cur_ax[1].set_yscale('log')
    cur_ax[1].set_xlabel('Iterations')
    cur_ax[1].set_ylim(lower_limit_factor*best_hv_over_mean_losses,upper_limit_factor * best_hv_over_mean_losses)
    # add title
    cur_ax[0].set_title('Convergence')
    # save figure
    plt.savefig(os.path.join(output_folder_name,image_file_name +'.png'), bbox_inches = "tight")
    plt.close(fig)



def plot_training_process(statistics_writer_instance,cfg,output_folder_name):
    # n_iters_per_epoch = np.floor(cfg["data_params"]["n_samples"]/cfg["batch_size"])
    # n_learning_iterations = cfg["n_epochs"] * np.floor(cfg["data_params"]["n_samples"]/cfg["batch_size"]) # should this be np.ceil?
    # n_epoch = cfg["n_epochs"]
    color_list = plt.cm.get_cmap('tab20',20)
    n_mo_obj = statistics_writer_instance.n_mo_obj
    n_mo_sol = statistics_writer_instance.n_mo_sol
    mo_obj_val_list = statistics_writer_instance.mo_obj_val_list
    validation_mo_obj_val_list = statistics_writer_instance.validation_mo_obj_val_list

    validation_iters = statistics_writer_instance.validation_iter_list
    x_label = 'Iterations'
    y_label = r'$\mathrm{HV}$'
    title = 'Training and validation metrics over iterations'
    image_file_name = 'training_and_validation_metrics'
    fig,cur_ax = plt.subplots(1 + n_mo_obj,figsize = (10,10), dpi = 100)
    # hypervolume & mean HV over samples
    cur_ax[0].plot(statistics_writer_instance.hv_list,zorder = 0,c = 'blue')
    cur_ax[0].plot(statistics_writer_instance.mean_hv_over_samples_list,zorder = 0,c = 'green')
    cur_ax[0].scatter(validation_iters,statistics_writer_instance.validation_hv_list,zorder = 1,c = 'red',edgecolor = 'black')
    cur_ax[0].scatter(validation_iters,statistics_writer_instance.validation_mean_hv_over_samples_list,zorder = 1,c = 'orange',edgecolor = 'black')
    cur_ax[0].set_ylabel(r'$\mathrm{HV}$')
    cur_ax[0].legend(['Training HV of mean losses','Training Mean HV over samples','Validation HV of mean losses','Validation Mean HV over samples'])
    # cur_ax[0].legend(['Training HV','Training Mean HV','Validation HV'])
    cur_ax[0].set_yscale('log')
    # one plot per loss
    for i_mo_obj in range(0,n_mo_obj):
        # reset color cycle ... does it work? is it necessary?
        # cur_ax[i_mo_obj].set_prop_cycle(None)
        
        # initialize legend lists
        legend_line_list = list()
        legend_label_list = list()
        # create dummy entries in legend to describe line/marker types
        legend_dummy_line = matplotlib.lines.Line2D([0],[0],color = 'gray')
        legend_line_list.append(legend_dummy_line)
        legend_label_list.append('Training loss')
        # legend_dummy_circle = matplotlib.patches.Circle([0,0],radius = 0, color = 'gray')
        legend_dummy_circle = cur_ax[1 + i_mo_obj].scatter(None,None, color = 'gray')
        legend_line_list.append(legend_dummy_circle)
        legend_label_list.append('Validation loss')
        # Problem specific settings
        if (cfg['problem_name'] == 'vincent_van_jarjar') and (cfg['obj_func'][i_mo_obj] == 'StyleLoss'):
            cur_ax[1 + i_mo_obj].set_yscale('log')
        for i_mo_sol in range(0,n_mo_sol):
            # training loss
            cur_loss = [mo_obj_val_list[i_iter][i_mo_obj,i_mo_sol] for i_iter in range(0,len(mo_obj_val_list))]
            line_handle, = cur_ax[1 + i_mo_obj].plot(cur_loss,zorder = 0,color = color_list(i_mo_sol))
            legend_line_list.append(line_handle)
            # validation loss
            cur_validation_loss = [validation_mo_obj_val_list[i_entry][i_mo_obj,i_mo_sol] for i_entry in range(0,len(validation_mo_obj_val_list))]
            cur_ax[1 + i_mo_obj].scatter(validation_iters,cur_validation_loss,zorder = 1,color = color_list(i_mo_sol),edgecolor = 'black')
            legend_label_list.append('Net ' + str(i_mo_sol))
        # add y label on each subplot
        cur_ax[1 + i_mo_obj].set_ylabel(cfg["obj_func"][i_mo_obj])
        # add legend
        cur_ax[1 + i_mo_obj].legend(legend_line_list,legend_label_list,bbox_to_anchor = (1,1)) # place legend outside on the right

    # add x label on last subplot
    cur_ax[-1].set_xlabel(x_label)
    # add title
    cur_ax[0].set_title(title)
    # save figure
    plt.savefig(os.path.join(output_folder_name,image_file_name +'.png'), bbox_inches = "tight")
    plt.close(fig)


def plot_mtl_as_moo(info_lists, output_folder="./output_files/plots", names=["MultiMNIST", "MultiFashionMNIST", "MultiFashion+MNIST"]):
    if len(info_lists)>3:
        raise ValueError("It's not gonna work for info_lists with len > 3")

    title_props = dict(fontweight='bold', fontsize=11)
    label_props = dict(weight="demi", fontsize=10)
    tick_props = dict(rotation=0, fontsize=8, fontweight="normal")
    legend_props = dict(size=12, weight='bold')
    line_props = dict(linewidth=3)

    os.makedirs(output_folder, exist_ok=True)
    config_names = {"left": "task left", "right": "task right",
                     "pytorch_weighted": "Linear Scalarization",
                     "pytorch_higamo": "HV-NN",
                     "pareto_mtl": "Pareto-MTL",
                     "epo": "epo"}
    scatter_props = {"pytorch_weighted": {"c":"red", "marker":"o"},
                     "pytorch_higamo": {"c":"blue", "marker":"s"},
                     "pareto_mtl": {"c":"green", "marker":">"},
                     "epo": {"c":"magenta", "marker":"<"}
                     }

    plt.figure(figsize=(17, 5), dpi=300)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.23, top=0.9, wspace=0.25)
    for i, info_list in enumerate(info_lists):
        plt.subplot(1, 3, i+1)
        for idx, info in enumerate(info_list):
            config_path = info["path"]
            cfg = info["config"]
            run_paths = info["run_paths"]
            task = cfg["data_params"]["task"]
            mo_optimizer = cfg["mo_optimizer"]

            filename = os.path.join(config_path, "test_output.json")
            test_output = json.load(open(filename, "r"))
            all_run_metrics = []
            for run_path in run_paths:
                metrics = test_output[run_path]
                all_run_metrics.append(metrics)

            all_run_metrics = np.array(all_run_metrics)
            print(all_run_metrics)
            # ---- average over runs ----
            all_run_metrics = np.average(all_run_metrics, axis=0)
            if task == "left":
                xdata = all_run_metrics[0] * np.ones(11)
                ydata = np.arange(0, 110, 10)
                plt.plot(xdata, ydata, linestyle="-", label=f"{config_names[task]}", color="grey", **line_props)
            elif task == "right":
                xdata = np.arange(0, 110, 10)
                ydata = all_run_metrics[0] * np.ones(11)
                plt.plot(xdata, ydata, linestyle="-", label=f"{config_names[task]}", color="grey", **line_props)   
            elif task == "both":
                xdata = np.array(all_run_metrics)[:, 0]
                ydata = np.array(all_run_metrics)[:, 1]
                plt.scatter(xdata, ydata, label=f"{config_names[mo_optimizer]}", **scatter_props[mo_optimizer])

        plt.grid(linewidth=1, linestyle="--", color='0.7', alpha=0.7)
        plt.xticks(np.arange(0, 100.1, 10), np.arange(0, 100.1, 10), **tick_props)
        plt.yticks(np.arange(0, 100.1, 10), np.arange(0, 100.1, 10), **tick_props)
        plt.xlim(0, 100.1)
        plt.ylim(0, 100.1)
        plt.xlabel("task left", **label_props)
        plt.ylabel("task right", **label_props)
        plt.title(names[i], **title_props)
        if i==1:
            plt.legend(edgecolor="none", loc="lower left", prop=legend_props, ncol=4, bbox_to_anchor=(-0.3, -0.3, 1.5, 0.2))
    
    plt.savefig(os.path.join(output_folder, 'MTL_as_MOO.png'), bbox_inches="tight")


if __name__ == '__main__':
    import _pickle as pickle
    rootpath = "/export/scratch3/grewal/hv_nn_training/output_files/mtl_new_approach_vs_old_20-11-2020_183132/1/run_42"
    checkpoint_path = os.path.join(rootpath, "checkpoint.pckl")
    checkpoint = pickle.load(open(checkpoint_path, "rb"))
    statistics_writer = checkpoint["stats"]

    cfg_path = os.path.join(rootpath, "..", "cfg.json")
    cfg = json.load(open(cfg_path, 'r'))

    output_folder_name = os.path.join(rootpath, "new_plotting")
    os.makedirs(output_folder_name, exist_ok=True)
    plot_selected_iterations(statistics_writer, cfg, output_folder_name, False)

