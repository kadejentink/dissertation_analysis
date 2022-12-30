"""
DESCRIPTION
contains several helper functions for RSA 
these include:
-option_check - checks provided parameters against allowed values
-expanded_names - generates string variables 
-get_file_info - gets header info from file(s) and adds data to input variable for searchlight
-calc_rsm_model_matrix - defines what is done to each searchlight. in this case, create a neural RDM, then correlate it to a hypothetical model matrix
-calc_rsm_neural_matrices - defines what is done to each searchlight. for this one, it computes two neural RDMs and correlates them
"""

# TODO
# could add searchlight size comparison to "calc_rsm_neural_matrices" like how it's implemented in "calc_rsm_model_matrices"

import nibabel as nib
import numpy as np
from nilearn.image import get_data, concat_imgs
from scipy import stats
import math

import cfg

# initialize dot bins
dot_bin_nums = [*range(7,17),*range(18,28)]

# possible comparison, phase and run numbers - these are all possible options I can select
RSA_options = ('neural_matrices','model_matrix')
model_options = ('category','motor','spectrum')
phase_options = ('train','test')
run_options = (1,2,3,4,'avg_234','avg_1234')
mask_options = ('whole_brain','single_voxel','all_ones')


def option_check(RSA_selection,model_selection,first_phase,first_run,second_phase,second_run,mask_selection):
    # RSA_selection - which RSA comparison would you like to make?
    if not all(x in RSA_options for x in [RSA_selection]):
        raise Exception(f"You entered {RSA_selection} as RSA option. Pick either 'neural_matrices' or 'model_matrix'.")

    # first phase/run selection check - you have to at least fill this out to do an RSA at all
    if not all(x in phase_options for x in [first_phase]):
        raise Exception(f"You entered {first_phase} as phase 1. Pick either 'train' or 'test'")
    if not all(x in run_options for x in [first_run]):
        raise Exception(f"You entered {first_run} as run 1. Pick either a number 1-4 or 'avg_234'/'avg_1234'")

    # model_selection - if you choose 'model_matrix', which one would you like to use?
    if RSA_selection is 'model_matrix':
        if not all(x in model_options for x in [model_selection]):
            raise Exception(f"You entered {model_selection} as model. Pick either 'category' or 'spectrum'.")

    if RSA_selection is 'neural_matrices':
        if second_phase not in phase_options:
            raise Exception(f"You entered {second_phase} as phase 2. Pick either 'train' or 'test'")
        if second_run not in run_options:
            raise Exception(f"You entered {second_run} as run 2. Pick either 1-4 or 'avg_234'/'avg_1234'")

    if not all(x in mask_options for x in [mask_selection]):
        raise Exception(f"You entered {mask_selection} for mask. Pick 'whole_brain', 'single_voxel', or 'all_ones'")

    print("\n")
    print("Options look good!")
    print("")


def expanded_names(first_run,second_run):
    # pre-formatted strings for run
    if (first_run is not 'avg_234' and first_run is not 'avg_1234'):
        first_run_name = f'run-{first_run:>02}'
    else:
        first_run_name = first_run

    if second_run is not None:
        if second_run is not 'avg':
            second_run_name = f'run-{second_run:>02}'
        else:
            second_run_name = second_run
    elif second_run is None:
        second_run_name = second_run

    return first_run_name,second_run_name


def get_file_info(RSA_selection,phase_images,phase_image_files,mask_np):
    data = []
    # get info from one file
    if RSA_selection is 'model_matrix':
        # only need one key because it's the same for both files
        key = list(phase_image_files.keys())[0]
        # just load the first file because affine/dim is same for all
        one_bin = nib.load(phase_image_files[key][0])
        affine_mat = one_bin.affine
        dimsize = one_bin.header.get_zooms()

        # create a list of conditions by using the generated condition strings attached to each batch of files
        condition_list = {"0":key}
        # concatenate all single trial maps from the same trial type - just goes in order files are listed in folder
        all_bins = concat_imgs(phase_images[key])
        # extract data into a numpy array to be later fed into BrainIAK searchlight function
        all_bins_data = get_data(all_bins)
        # append extracted data into variable for BrainIAK
        data.append(all_bins_data)

    # or get info from one file from each key - partially to collect info, partially to compare for equivalency
    if RSA_selection is 'neural_matrices':
        # only need one set of keys because it's the same for both files
        key1,key2 = phase_image_files.keys()
        # load a file from each phase to check they're similar in shape
        one_bin_1 = nib.load(phase_image_files[key1][0])
        one_bin_2 = nib.load(phase_image_files[key2][0])
        affine_mat_1, affine_mat_2 = one_bin_1.affine, one_bin_2.affine
        dimsize_1, dimsize_2 = one_bin_1.header.get_zooms(), one_bin_2.header.get_zooms()
        # condition_list is just to help remember which spot in 'data' is which condition
        # a dict would be better, but I think brainiak needs a list? could a dict do that?
        condition_list = {"0":key1,"1":key2}
        all_bins_1 = concat_imgs(phase_images[key1])
        all_bins_2 = concat_imgs(phase_images[key2])
        data_1 = get_data(all_bins_1)
        data_2 = get_data(all_bins_2)
        data.append(data_1)
        data.append(data_2)
        print(f'Are affine matrices equal: {np.array_equal(affine_mat_1,affine_mat_2)}')
        print(f'Are voxel sizes equal: {np.array_equal(dimsize_1,dimsize_2)}')
        print(f'Are data the same shape: {np.array_equal(data[0].shape,data[1].shape)}')
        # assign one of the two identical matrices to a generic variable shared by other paths of the script
        affine_mat = affine_mat_1
        dimsize = dimsize_1

    # only checks one data file because a) it works for both RSA_selection options, and b) for 'neural_matrices', both should be the same size
    print(f'Is your mask the same size as your data: {np.array_equal(data[0].shape[:3],mask_np.shape)}')

    return data,condition_list,affine_mat,dimsize


def calc_rsm_model_matrix(data, sl_mask, myrad, bcvar, ):
    # extract 1st subject's data and labels
    data4D = data[0]
    labels = bcvar
    
    # applies to Ball shape only - specifies maximum searchlight size based on predefined radius
    # used to determine whether or not there's sufficient voxels available to run a searchlight
    if myrad == 1:
        total_voxels = 7
    elif myrad == 2:
        total_voxels = 33
    elif myrad == 3:
        total_voxels = 123
    elif myrad == 4:
        total_voxels = 257
    elif myrad == 5:
        total_voxels = 515
    elif myrad == 6:
        total_voxels = 925
    elif myrad == 7:
        total_voxels = 1419
    elif myrad == 8:
        total_voxels = 2109
    elif myrad == 10:
        total_voxels = 4169

    # set the minimum number of required voxels for a searchlight. roughly half (rounded up) total voxels in a searchlight cluster
    voxel_threshold = math.ceil(total_voxels/2)
    # return 0 if searchlight is too small
    if np.sum(sl_mask) < voxel_threshold:
        return 0

    # reshape data into "betas x voxels" using the current searchlight cluster voxels (i.e., sl_mask == 1)
    bolddata_sl = data4D[sl_mask==1].T
    
    # identify the size of the current searchlight cluster
    current_cluster_size = bolddata_sl.shape[1]
    
    # using a global variable to track the largest cluster size, compare the size of the current cluster against the previous largest
    # this process ends up storing the maximum cluster size across all searchlight comparisons
    if cfg.searchlight_cluster_size >= current_cluster_size:
        cfg.searchlight_cluster_size = cfg.searchlight_cluster_size
    elif current_cluster_size > cfg.searchlight_cluster_size:
        cfg.searchlight_cluster_size = current_cluster_size

    # voxels are columns, rows are values for each trial
    # if the number of columns (voxels) NOT containing all zeros is less than threshold, return 0
    # (the number of voxels containing data for all trials should be greater than the voxel threshold)
    # (voxels without data would be a column of all 0 - would sum to 0)
    if np.count_nonzero(np.sum(bolddata_sl,axis=0)) < voxel_threshold:
        return 0

    # Pearson correlation, excluding voxels outside the brain i.e., "bolddata_sl != 0" excludes columns (voxels) of all 0's
    # even though we're excluding searchlights with a lot of zero columns, there may still be some in passing searchlight clusters
    rsm = np.corrcoef(bolddata_sl[:,np.any(bolddata_sl!=0,axis=0)])
    # 1-r is a common transform in RSA - turns data from similarity matrix to DISsimilarity matrix
    RDM = 1 - rsm

    # np.round because otherwise scientific notation makes some '0' values not equal to 0
    round_RDM = np.round(RDM,decimals=8)

    # compare the neural RDM to a model, using only the upper off-diagonal triangle values (excluding zero diagonal)
    triu_RSA, _ = stats.spearmanr(round_RDM[np.triu_indices(round_RDM.shape[0],k=1)],
                                  labels[np.triu_indices(labels.shape[0],k=1)])

    # Fisher z-transform the r-value
    fisher_RSA = np.arctanh(triu_RSA)

    # return the Fisher z-transformed RSA value for the center of each searchlight cluster
    return fisher_RSA


def calc_rsm_neural_matrices(data, sl_mask, myrad, bcvar, ):
    # extract both sets of data
    data_train = data[0]
    data_test = data[1]

    # apply to Ball shape only
    if myrad == 1:
        total_voxels = 7
    elif myrad == 2:
        total_voxels = 33
    elif myrad == 3:
        total_voxels = 123
    elif myrad == 4:
        total_voxels = 250
    elif myrad == 5:
        total_voxels = 515
    elif myrad == 6:
        total_voxels = 925
    elif myrad == 7:
        total_voxels = 1419
    elif myrad == 8:
        total_voxels = 2109
    elif myrad == 10:
        total_voxels = 4169

    voxel_threshold = math.ceil(total_voxels/2)
    if np.sum(sl_mask) < voxel_threshold:
        return 0

    # selecting only voxels from the training data where the searchlight mask == 1
    bolddata_train = data_train[sl_mask==1].T
    bolddata_test = data_test[sl_mask==1].T

    # this checks to make sure that, after masking the BOLD data, there aren't too many 0 columns
    if np.count_nonzero(np.sum(bolddata_train,axis=0)) < voxel_threshold:
        return 0
    if np.count_nonzero(np.sum(bolddata_test,axis=0)) < voxel_threshold:
        return 0

    # do the Pearson correlation but exclude columns containing zero (if there are any - there may not be)
    rsm_train = np.corrcoef(bolddata_train[:,np.any(bolddata_train!=0,axis=0)])
    RDM_train = 1 - rsm_train

    rsm_test = np.corrcoef(bolddata_test[:,np.any(bolddata_test!=0,axis=0)])
    RDM_test = 1 - rsm_test

    # do minor rounding
    round_RDM_train = np.round(RDM_train,decimals=8)
    round_RDM_test = np.round(RDM_test,decimals=8)

    # takes two vectors and gives one RSA value
    # the matrices turn into vectors when I slice them using [round_RDM_train!=0]
    triu_RSA, _ = stats.spearmanr(round_RDM_train[np.triu_indices(round_RDM_train.shape[0],k=1)],
                                  round_RDM_test[np.triu_indices(round_RDM_test.shape[0],k=1)])

    # Fisher z-transform the r-value
    fisher_RSA = np.arctanh(triu_RSA)

    return fisher_RSA
