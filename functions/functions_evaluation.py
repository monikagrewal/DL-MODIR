import numpy as np
import pdb

from functions.functions_hv_python3 import HyperVolume

def compute_ud_gradient_2d(mo_obj_val,ref_point,ud_eps):
    n_obj = mo_obj_val.shape[0]
    n_mo_sol = mo_obj_val.shape[1]
    assert n_obj == 2
    ud,ud_contr,obj_space_ud_components = compute_ud_2d(mo_obj_val,ref_point,ud_eps)
    ud_gradient = 2 * obj_space_ud_components
    assert not np.any(np.isnan(ud_gradient))
    return(ud_gradient)


def compute_ud_2d(mo_obj_val,ref_point,ud_eps):
    n_obj = mo_obj_val.shape[0]
    n_mo_sol = mo_obj_val.shape[1]
    # NOTE: this can be made more efficient by vectorizing, not computing dominance of ALL 'inverse' corner points to a dominated point
    assert n_obj == 2
    obj_space_ud_components = np.zeros((2,n_mo_sol))
    # initialization: the closest point of an mo-solution is set to itself (the closest point is only recorded for possible future use)
    ud_closest_point = mo_obj_val.copy()

    # determine and count mo_sols that dominate reference point
    ref_dom_indices,n_ref_dom_mo_sol,mo_sol_dominates_ref_point = determine_ref_dom_mo_sol(mo_obj_val,ref_point)
    # select non-dominated mo_sols
    non_dom_indices,mo_sol_is_dominated = determine_non_dom_mo_sol(mo_obj_val)
    n_non_dom = np.sum(mo_sol_is_dominated == False) # CHECK

    # special case: if all solutions are non-dominated and all dominate the reference point, UD = 0
    if (n_non_dom == n_mo_sol) and (n_ref_dom_mo_sol == n_mo_sol):
        ud_contr = np.zeros(n_mo_sol)
        ud = np.zeros(1)
        return(ud,ud_contr,obj_space_ud_components)

    ## determine whether the UD is computed w.r.t. the reference box or the domination boundary of the non-dominated mo-solutions
    # if no point dominates the reference point, compute each mo_sol's distance to the reference box
    ref_point_is_corner_point = False
    if n_ref_dom_mo_sol == 0:
        inv_corner_obj_val = ref_point
        n_inv_corner = 1
        ref_point_is_corner_point = True
        # add 2nd dimension to array because later we will iterate over all inv. corners via the 2nd dimension
        inv_corner_obj_val = inv_corner_obj_val[:,None] 
    else:
        # select mo-solutions that dominate the reference point AND are non-dominated, sort in increasing order of objective one
        # _,_,ref_dom_non_dom_indices,n_ref_dom_non_dom_mo_sol,inv_sort_indices,sorted_ref_dom_non_dom_obj_val = self.determine_ref_dom_non_dom_mo_sol()
        # select mo-solutions that dominate the reference point AND are non-dominated, sort in increasing order of objective one
        _,_,ref_dom_non_dom_indices,n_ref_dom_non_dom_mo_sol,inv_sort_indices,sorted_ref_dom_non_dom_obj_val = determine_ref_dom_non_dom_mo_sol(mo_obj_val,ref_point,non_dom_indices,ref_dom_indices)
        # if there is just one mo-solution on the domination boundary, it itself is the only cornerpoint
        if n_ref_dom_non_dom_mo_sol == 1:
            inv_corner_obj_val = sorted_ref_dom_non_dom_obj_val
            n_inv_corner = 1
        # otherwise construct the 'inverse' corner points from multiple mo-solutions on the domination boundary
        elif n_ref_dom_non_dom_mo_sol > 1:
            n_inv_corner = n_ref_dom_non_dom_mo_sol-1
            inv_corner_obj_val = np.zeros((n_obj,n_inv_corner))
            
            inv_corner_obj_val[0,:] = sorted_ref_dom_non_dom_obj_val[0,1:]
            inv_corner_obj_val[1,:] = sorted_ref_dom_non_dom_obj_val[1,:-1]
        else:
            raise ValueError('Unknown case. n_ref_dom_mo_sol != 0  but n_ref_dom_non_dom_mo_sol < 1.')

    ## compute UD
    # initialize ud contribution
    ud_contr = np.zeros(n_mo_sol)
    for i_mo_sol in range(0,n_mo_sol):
        # if solution is dominated or outside the reference box (i.e. not dominating the ref point), then compute its ud contribution, otherwise leave it at 0
        # if self.mo_sol_is_dominated[i_mo_sol]:
        if mo_sol_is_dominated[i_mo_sol] or (not mo_sol_dominates_ref_point[i_mo_sol]):    
            dist_to_dom_boundary_list = list()
            horz_dist_list = list()
            vert_dist_list = list()
            # compute the distances to each of the boxes 'underneath' the domination boundary
            for i_inv_cor in range(0,n_inv_corner):
                dist_to_box, horz_dist, vert_dist = compute_eucl_distance_to_box(inv_corner_obj_val[:,i_inv_cor],mo_obj_val[:,i_mo_sol],ud_eps)
                dist_to_dom_boundary_list.append(dist_to_box)
                horz_dist_list.append(horz_dist)
                vert_dist_list.append(vert_dist)
            # select the distance to the closest box and make their the domianted mo solution's UD contribution
            min_ind = np.argmin(dist_to_dom_boundary_list)
            ud_contr[i_mo_sol] = dist_to_dom_boundary_list[min_ind]
            # record the horizontal and vertical components for UD-gradient computation         
            obj_space_ud_components[0,i_mo_sol] = horz_dist_list[min_ind]
            obj_space_ud_components[1,i_mo_sol] = vert_dist_list[min_ind]
            # record the closest point (maybe useful in the future)
            ud_closest_point = inv_corner_obj_val[:,min_ind]


    # the UD is the mean of the distances to the power of self.n_obj (for 2d: self.n_obj == 2)
    ud = np.mean(ud_contr**n_obj)
    assert not np.isnan(ud)
    assert not np.any(np.isnan(ud_contr))
    assert ud >= 0
    assert np.all(ud_contr >=0)
    return(ud,ud_contr,obj_space_ud_components)


def compute_eucl_distance_to_box(corner_point,free_point,ud_eps):
    horz_dist = free_point[0] - (corner_point[0] - ud_eps)
    vert_dist = free_point[1] - (corner_point[1] - ud_eps)

    # if the vertical distance is negative, the free point is below the corner point
    if (horz_dist > 0) and (vert_dist < 0):
        eucl_dist_to_box = horz_dist
        vert_dist = 0
    # if the horizontal distance is negative, the free point is to the left of the corner point
    elif (horz_dist < 0) and (vert_dist > 0):
        eucl_dist_to_box = vert_dist
        horz_dist = 0
    # if both distances are positive, the free point is to the top right of the corner point
    elif (horz_dist > 0) and (vert_dist > 0):
        # pythagoras
        eucl_dist_to_box = np.sqrt((horz_dist**2.0 + vert_dist**2.0))
    else:
        pdb.set_trace()
        raise ValueError('Unexpected case. Dominated point seems to dominate a corner point on the domination boundary or the reference point?')
    
    return(eucl_dist_to_box,horz_dist,vert_dist)

def compute_hv_gradient_2d_with_duplicate_handling(mo_obj_val,ref_point):
    # find unique mo_obj_val
    unique_mo_obj_val, mapping_indices = np.unique(mo_obj_val, axis = 1, return_inverse = True)
    # compute hv_grad for unique mo-solutions
    unique_hv_grad = compute_hv_gradient_2d(unique_mo_obj_val,ref_point)
    # assign the same gradients to duplicate mo_obj_val
    hv_grad = unique_hv_grad[:,mapping_indices]
    return(hv_grad)

def compute_hv_gradient_2d(mo_obj_val,ref_point):
    # the hv gradient of given mo-solution is
    # for objective 0, the vertical length of the rectangle from the neighboring mo solution on the left to the given mo-solution.
    # for objective 1, the horizontal lenght of the rectangle from the neighboring mo solution on the right to the given mo-solution.
    # first and last mo-solutions need to consider the reference box as 'neighboring mo-solutions' (draw it, and then you see it)
    # if there is only one mo-solution, which dominates the reference point and is non-dominated by any other mo-solution, to consider the reference box as 'neighboring mo-solutions' (draw it, and then you see it)

    n_obj = mo_obj_val.shape[0]
    n_solutions = mo_obj_val.shape[1]
    assert n_obj == 2
    # determine and count mo_sols that dominate reference point
    ref_dom_indices,n_ref_dom_mo_sol,_ = determine_ref_dom_mo_sol(mo_obj_val,ref_point)
    
    # if no point dominates the reference point, return 0 for all gradients
    hv_gradient = np.zeros_like(mo_obj_val)
    if not n_ref_dom_mo_sol == 0:
        # select non-dominated mo_sols
        non_dom_indices,mo_sol_is_dominated = determine_non_dom_mo_sol(mo_obj_val)
        # select mo-solutions that dominate the reference point AND are non-dominated, sort in increasing order of objective one
        _,_,ref_dom_non_dom_indices,n_ref_dom_non_dom_mo_sol,inv_sort_indices,sorted_ref_dom_non_dom_obj_val = determine_ref_dom_non_dom_mo_sol(mo_obj_val,ref_point,non_dom_indices,ref_dom_indices)
        # select mo-solutions that dominate the reference point AND are non-dominated, sort in increasing order of objective one
        # _,_,ref_dom_non_dom_indices,n_ref_dom_non_dom_mo_sol,inv_sort_indices,sorted_ref_dom_non_dom_obj_val = self.determine_ref_dom_non_dom_mo_sol()            

        hv_gradient_sorted_ref_dom_non_dom = np.zeros((n_obj,n_ref_dom_non_dom_mo_sol))
        # if there is only one mo-solution that dominates the ref point and is non-dominated by other mo-solutions, the hv gradient is defined by the rectangle between the mo-solution and the reference point
        if n_ref_dom_non_dom_mo_sol == 1:
            hv_gradient_sorted_ref_dom_non_dom[0,0] = - ( ref_point[1] - sorted_ref_dom_non_dom_obj_val[1,0] )
            hv_gradient_sorted_ref_dom_non_dom[1,0] = - ( ref_point[0] - sorted_ref_dom_non_dom_obj_val[0,0] )
        elif n_ref_dom_non_dom_mo_sol > 1:
            # first mo-solution
            hv_gradient_sorted_ref_dom_non_dom[0,0] = - ( ref_point[1] - sorted_ref_dom_non_dom_obj_val[1,0] )
            hv_gradient_sorted_ref_dom_non_dom[1,0] = - ( sorted_ref_dom_non_dom_obj_val[0,(0+1)] - sorted_ref_dom_non_dom_obj_val[0,0] )
            # intermediate mo-solutions
            for i_mo_obj_val in range(1,n_ref_dom_non_dom_mo_sol-1): # 1 -1 because the first and last mo-solutions need to be treated separately
                hv_gradient_sorted_ref_dom_non_dom[0,i_mo_obj_val] = - ( sorted_ref_dom_non_dom_obj_val[1,(i_mo_obj_val-1)] - sorted_ref_dom_non_dom_obj_val[1,i_mo_obj_val] )
                hv_gradient_sorted_ref_dom_non_dom[1,i_mo_obj_val] = - ( sorted_ref_dom_non_dom_obj_val[0,(i_mo_obj_val+1)] - sorted_ref_dom_non_dom_obj_val[0,i_mo_obj_val] )

            # last last mo-solution
            hv_gradient_sorted_ref_dom_non_dom[0,-1] = - ( sorted_ref_dom_non_dom_obj_val[1,-2] - sorted_ref_dom_non_dom_obj_val[1,-1] )
            hv_gradient_sorted_ref_dom_non_dom[1,-1] = - ( ref_point[0] - sorted_ref_dom_non_dom_obj_val[0,-1] )
        else:
            raise ValueError('Unknown case. There should always be 1 mo-solution in this if-statement.')
        
        hv_gradient[:,ref_dom_non_dom_indices] = hv_gradient_sorted_ref_dom_non_dom[:,inv_sort_indices]
    assert np.all(hv_gradient <= 0) # we are minimizing the mo-objectives. Therefore, an increase in the objectives should always yield to a decrease in HV.
    assert not np.any(np.isnan(hv_gradient))
    return(hv_gradient)


def compute_hv_2d(mo_obj_val,ref_point):
    # sum the area of rectangles starting on the horizontal line, which is defined by the reference point and the y-axis, and ending in a mo-solution and its neighboring mo-solution on the right (last rectangle is fully defined by reference point and the last mo-solution)
    # check that input has correct dimension
    assert len(mo_obj_val.shape) == 2
    assert len(ref_point) == 2
    n_obj = mo_obj_val.shape[0]
    n_mo_sol = mo_obj_val.shape[1]
    assert n_obj == 2

    # determine and count mo_sols that dominate reference point
    ref_dom_indices,n_ref_dom_mo_sol,_ = determine_ref_dom_mo_sol(mo_obj_val,ref_point)

    # if no point dominates the reference point, return 0
    if n_ref_dom_mo_sol == 0:
        hv = np.zeros(1)
    else:
        # select non-dominated mo_sols
        non_dom_indices,mo_sol_is_dominated = determine_non_dom_mo_sol(mo_obj_val)
        # select mo-solutions that dominate the reference point AND are non-dominated, sort in increasing order of objective one
        _,_,_,n_ref_dom_non_dom_mo_sol,_,sorted_ref_dom_non_dom_obj_val = determine_ref_dom_non_dom_mo_sol(mo_obj_val,ref_point,non_dom_indices,ref_dom_indices)
        hv = np.zeros(1)
        for i_mo_obj_val in range(0,n_ref_dom_non_dom_mo_sol-1): # -1 because the last rectangle needs to be treated separately
            hv += ( ref_point[1] - sorted_ref_dom_non_dom_obj_val[1,i_mo_obj_val] )  * ( sorted_ref_dom_non_dom_obj_val[0,(i_mo_obj_val+1)] - sorted_ref_dom_non_dom_obj_val[0,i_mo_obj_val] )
        # last rectangle
        hv += ( ref_point[1] - sorted_ref_dom_non_dom_obj_val[1,-1] ) * ( ref_point[0] - sorted_ref_dom_non_dom_obj_val[0,-1] )
        # note: in the case that there is only one solution, the for loop is not used and the last rectangle is also the only rectangle
    assert not np.isnan(hv)
    assert hv >= 0
    return(hv)



def determine_ref_dom_non_dom_mo_sol(mo_obj_val,ref_point,non_dom_indices,ref_dom_indices):

    # select mo-solutions that dominate the reference point AND are non-dominated, sort in increasing order of objective one
    ref_dom_non_dom_indices = np.intersect1d(ref_dom_indices,non_dom_indices)
    ref_dom_non_dom_obj_val = mo_obj_val[:,ref_dom_non_dom_indices]
    ref_dom_non_dom_mo_sol = mo_obj_val[:,ref_dom_non_dom_indices]
    n_ref_dom_non_dom_mo_sol = ref_dom_non_dom_obj_val.shape[1]

    # sort points in increasing order of objective one
    sort_indices = np.argsort(ref_dom_non_dom_obj_val[0,:])
    # sort_indices = sort_indices[0] # somehow this is necessary
    sorted_ref_dom_non_dom_obj_val = ref_dom_non_dom_obj_val[:,sort_indices]
    # use argsort to find indices that revert the previous sorting. Note to self: sketch an example
    inv_sort_indices = np.argsort(sort_indices)

    # assert that the indexing logic is correct (the inversion of the sorting)
    assert np.all( mo_obj_val[:,ref_dom_non_dom_indices] == sorted_ref_dom_non_dom_obj_val[:,inv_sort_indices])
    return(ref_dom_non_dom_mo_sol,ref_dom_non_dom_obj_val,ref_dom_non_dom_indices,n_ref_dom_non_dom_mo_sol,inv_sort_indices,sorted_ref_dom_non_dom_obj_val)


def determine_non_dom_mo_sol(mo_obj_val):
    # get set of non-dominated solutions, returns indices of non-dominated and booleans of dominated mo_sol
    n_mo_sol = mo_obj_val.shape[1]
    domination_rank = fastNonDominatedSort(mo_obj_val)
    non_dom_indices = np.where(domination_rank == 0)
    non_dom_indices = non_dom_indices[0] # np.where returns a tuple, so we need to get the array inside the tuple
    # non_dom_mo_sol = mo_sol[:,non_dom_indices]
    # non_dom_mo_obj_val = mo_obj_val[:,non_dom_indices]
    mo_sol_is_non_dominated = np.zeros(n_mo_sol,dtype = bool)
    mo_sol_is_non_dominated[non_dom_indices] = True
    mo_sol_is_dominated = np.bitwise_not(mo_sol_is_non_dominated)
    return(non_dom_indices,mo_sol_is_dominated)


def determine_ref_dom_mo_sol(mo_obj_val,ref_point):
    # select only mo-solutions that dominate the reference point
    ref_point_temp = ref_point[:,None] # add axis so that comparison works
    ref_dom_booleans = np.all(mo_obj_val < ref_point_temp  , axis = 0)
    ref_dom_indices = np.where(ref_dom_booleans == True)

    mo_sol_dominates_ref_point = ref_dom_booleans
    ref_dom_indices = ref_dom_indices[0] # somehow this is necessary
    # ref_dom_mo_sol = mo_sol[:,ref_dom_indices]
    ref_dom_mo_obj_val = mo_obj_val[:,ref_dom_indices]
    n_ref_dom_mo_sol = ref_dom_mo_obj_val.shape[1]

    assert n_ref_dom_mo_sol >= 0
    # assert not np.any(np.isnan(ref_dom_mo_sol))
    assert not np.any(np.isnan(ref_dom_mo_obj_val))
    return(ref_dom_indices,n_ref_dom_mo_sol,mo_sol_dominates_ref_point)


def fastNonDominatedSort(objVal):
    # taken from my reinforcement learning repository
    # As in Deb et al. (2002) NSGA-II
    N_OBJECTIVES = objVal.shape[0] 
    N_SOLUTIONS = objVal.shape[1]

    rankIndArray = - 999 * np.ones(N_SOLUTIONS, dtype = int) # -999 indicates unassigned rank
    solIndices = np.arange(0,N_SOLUTIONS) # array of 0 1 2 ... N_SOLUTIONS
    ## compute the entire domination matrix
    # dominationMatrix: (i,j) is True if solution i dominates solution j
    dominationMatrix = np.zeros((N_SOLUTIONS,N_SOLUTIONS), dtype = bool)
    for p in solIndices:
        objValA = objVal[:,p][:,None] # add [:,None] to preserve dimensions
        # objValArray =  np.delete(objVal, obj = p axis = 1) # dont delete solution p because it messes up indices
        dominates = checkDomination(objValA,objVal)
        dominationMatrix[p,:] = dominates

    # count the number of times a solution is dominated
    dominationCounter = np.sum(dominationMatrix, axis = 0)

    ## find rank 0 solutions to initialize loop
    isRankZero = (dominationCounter == 0) # column and row binary indices of solutions that are rank 0
    # pdb.set_trace()
    rankZeroRowInd = solIndices[isRankZero] 
    # mark rank 0's solutions by -99 so that they are not considered as members of next rank
    dominationCounter[rankZeroRowInd] = -99
    # initialize rank counter at 0
    rankCounter = 0
    # assign solutions in rank 0 rankIndArray = 0
    rankIndArray[isRankZero] = rankCounter

    isInCurRank = isRankZero
    # while the current rank is not empty
    while not (np.sum(isInCurRank) == 0):
        curRankRowInd = solIndices[isInCurRank] # column and row numbers of solutions that are in current rank 
        # for each solution in current rank
        for p in curRankRowInd:
            # decrease domination counter of each solution dominated by solution p which is in the current rank
            dominationCounter[dominationMatrix[p,:]] -= 1 #dominationMatrix[p,:] contains indices of the solutions dominated by p
        # all solutions that now have dominationCounter == 0, are in the next rank		
        isInNextRank = (dominationCounter == 0)
        rankIndArray[isInNextRank] = rankCounter + 1	
        # mark next rank's solutions by -99 so that they are not considered as members of future ranks
        dominationCounter[isInNextRank] = -99
        # increase front counter
        rankCounter += 1
        # check which solutions are in current rank (next rank became current rank)
        isInCurRank = (rankIndArray == rankCounter)
        if not np.all(isInNextRank == isInCurRank): # DEBUGGING, if it works fine, replace above assignment
            pdb.set_trace()
    return(rankIndArray)

def checkDomination(objValA,objValArray):
    # taken from my reinforcement learning repository
    dominates = ( np.any(objValA < objValArray, axis = 0) & np.all(objValA <= objValArray , axis = 0) )
    return(dominates)

def compute_hv_in_higher_dimensions(mo_obj_val, ref_point):
    n_mo_obj = mo_obj_val.shape[0]
    n_mo_sol = mo_obj_val.shape[1]
    assert len(ref_point) == n_mo_obj
    # initialize hv computation instance
    hv_computation_instance = HyperVolume(tuple(ref_point))
    # turn numpy array to list of tuples
    list_of_mo_obj_val = list()
    for i_mo_sol in range(n_mo_sol):
        list_of_mo_obj_val.append(tuple(mo_obj_val[:,i_mo_sol]))

    hv = float(hv_computation_instance.compute(list_of_mo_obj_val))
    return(hv)