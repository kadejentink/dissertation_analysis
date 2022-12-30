"""
DESCRIPTION
provides framework for RSA analysis 
general order of operations:
-generate a few useful string variables
-grab the single trial maps needed for the analysis
-compute whole brain mask
-initialize and begin searchlight
-save resulting RSA map
incorporates elements from helper scripts
"""

# import calls
import sys
import os
import time

import math
import numpy as np
from scipy import stats
import nibabel as nib
from nilearn.image import get_data, concat_imgs
from nilearn.masking import intersect_masks, compute_background_mask
from brainiak.searchlight.searchlight import Searchlight, Ball
from brainiak import io
from pathlib import Path
from mpi4py import MPI

import importlib

# custom helper functions
from RSA_helpers import option_check, expanded_names, get_file_info
from RSA_helpers import calc_rsm_model_matrix, calc_rsm_neural_matrices
# global variables
import cfg

# option initialization for how numbers are displayed in jupyter notebooks
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=4, suppress=True)


def full_RSA(RSA_selection,model_selection,first_phase,first_run,second_phase,second_run,mask_selection,
             data_path,base_out_path,suffix,sl_rad,max_blk_edge,pool_size,sub_name,first_run_name,
             second_run_name,all_dots_height):

    # some conditional initializations
    # get some strings formatted and assign "None" to unused variables if it wasn't done in initialization
    
    # don't need a value for "second run" if we're not analyzing one
    if RSA_selection is 'model_matrix':
        second_phase = second_run = None
    # similarly, don't need a model matrix if we're just comparing two neural RDMs
    elif RSA_selection is 'neural_matrices':
        model_selection = None

    # naming comparison for output
    if RSA_selection is 'model_matrix':
        run_comparison_name = f"{first_phase}-{first_run_name}_model-{model_selection}"
        out_path = os.path.join(base_out_path,run_comparison_name)
    elif RSA_selection is 'neural_matrices':
        run_comparison_name = f"{first_phase}-{first_run_name}_{second_phase}-{second_run_name}"
        out_path = os.path.join(base_out_path,run_comparison_name)

    # create path for output
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # set output name/path
    output_file_name = f'{sub_name}_{run_comparison_name}_mask-{mask_selection}_rad-{sl_rad}.nii.gz'
    output_name = os.path.join(out_path, output_file_name)

    # print out basic filename information to keep track of who is being processed
    print("Processing...")
    print(f"Output file name: \n{output_file_name}")


    # two dict objects (in this order): a list (generator) of nibabel NIfTI objects, and a list of file paths as str
    
    # initialize empty dictionaries
    phase_images = {}
    phase_image_files = {}
    # grabs the two different types of dict containers
    if RSA_selection is 'model_matrix':
        phase_images[f"{first_phase}_{first_run_name}"] = io.load_images_from_dir(os.path.join(
            data_path,sub_name,first_phase,first_run_name), suffix)
        phase_image_files[f"{first_phase}_{first_run_name}"] = sorted(Path(os.path.join(
            data_path,sub_name,first_phase,first_run_name)).glob("*" + suffix))
        # load in model matrices if it exists
        if model_selection is not None:
            model = np.loadtxt(f"{data_path}/model_{model_selection}.csv",delimiter=',')

    if RSA_selection is 'neural_matrices':
        for phase,run in zip([first_phase,second_phase],[first_run_name,second_run_name]):
            phase_images[f"{phase}_{run}"] = io.load_images_from_dir(os.path.join(
                data_path,sub_name,phase,run), suffix)
            phase_image_files[f"{phase}_{run}"] = sorted(Path(os.path.join(
                data_path,sub_name,phase,run)).glob("*" + suffix))

    # compute masks - load a random file (within a run, they're all the same shape/size), compute a mask
    # create a variable for whatever derivatives folder you have selected
    if all_dots_height is 'all':
        file_name = f"{sub_name}_{first_phase}_{first_run_name}_category_0_001.nii"
    elif all_dots_height is 'dots':
        # each run has the same dot bins, which is why we can hard-code this value ("dots-07")
        if first_run_name is 'avg_234':
            file_name = f"{sub_name}_{first_phase}_runs-{first_run_name[-3:]}_dots-07.nii"
        elif first_run_name is 'avg_1234':
            file_name = f"{sub_name}_{first_phase}_runs-{first_run_name[-4:]}_dots-07.nii"
        else:
            file_name = f"{sub_name}_{first_phase}_{first_run_name}_dots-07.nii"
    elif all_dots_height is 'height':
        # same idea as above, each run has the same height bins, so we can hard-code this
        if first_run_name is 'avg_234':
            file_name = f"{sub_name}_{first_phase}_runs-{first_run_name[-3:]}_height-183.nii"
        elif first_run_name is 'avg_1234':
            file_name = f"{sub_name}_{first_phase}_runs-{first_run_name[-4:]}_height-183.nii"
        else:
            file_name = f"{sub_name}_{first_phase}_{first_run_name}_height-183.nii"
    
    # compute masks - load a random file (within a run, they're all the same shape/size), compute a mask
    # if 'neural_matrices' is selected, compute an intersection of both functional masks
    if RSA_selection is 'model_matrix':
        # use Nibabel to load data as NIfTI object
        mask = nib.load(os.path.join(data_path,sub_name,first_phase,first_run_name,file_name))
        # computer background mask (looks for 0-values in data and excludes them from the mask)
        mask = compute_background_mask(mask)
        # import data into numpy array
        mask_np = get_data(mask)
    elif RSA_selection is 'neural_matrices':
        mask1 = nib.load(os.path.join(data_path,sub_name,first_phase,first_run_name,file_name))
        mask2 = nib.load(os.path.join(data_path,sub_name,second_phase,second_run_name,file_name))

        mask1 = compute_background_mask(mask1)
        mask2 = compute_background_mask(mask2)

        mask_list = [mask1,mask2]
        mask = intersect_masks(mask_list,threshold=1.0)
        mask_np = get_data(mask)

    # get misc info from one file for saving later
    # if RSA_selection = 'neural_matrices', get one file from each phase/run combo and also checks that they're equal (affine, voxel size)
    # it should be the same for all files in both phases already, but let's double-check

    # loads one file to get header info, then concatenates all the images for each run and imports their data into a numpy array
    data,condition_list,affine_mat,dimsize = get_file_info(RSA_selection,phase_images,phase_image_files,mask_np)

    # BrainIAK initializations for searchlight analysis
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    # first is whole mask, second is significant voxel (this can be changed), third is entire box of 1's for testing searchlight shape
    # mask is a computed background mask
    if mask_selection is 'whole_brain':
        small_mask = mask_np
    # 1-voxel at a specified highly significant spot for RDM v cat model comparison
    elif mask_selection is 'single_voxel':
        small_mask = np.zeros(mask_np.shape)
        small_mask[42,28,49] = 1
    # entire box of ones for visualizing searchlight shapes or other reasons
    elif mask_selection is 'all_ones':
        mask_test = mask_np.copy()
        mask_test.fill(1)
        small_mask = mask_test

    # bcvar could be model, or if you're comparing two neural matrices, "None" (it's unused in that comparison)
    if RSA_selection is 'model_matrix':
        bcvar = model
    elif RSA_selection is 'neural_matrices':
        bcvar = None


    # create searchlight object and make information available for processing. select debugging or not
    
    # creates searchlight object
    sl = Searchlight(sl_rad=sl_rad,max_blk_edge=max_blk_edge,shape=Ball)

    # text update - data shape should match mask shape (in first 3 dimensions)
    print("Setup searchlight inputs")
    print("Input data shape: " + str(data[0].shape))
    print("Input mask shape: " + str(small_mask.shape) + "\n")

    # Distribute the information to the searchlights (preparing it to run)
    # 'data' needs to be a list i.e., data[0].shape = x,y,z,trials/epochs
    # the list can contain data for multiple subjects, but in this case, it has data for both phases
    sl.distribute(data, small_mask)
    # Data that is needed for all searchlights is sent to all cores via the sl.broadcast function.
    # In this example, we are sending the labels for classification to all searchlights.
    sl.broadcast(bcvar)

    # Start the searchlight timer
    begin_time = time.time()

    # text update
    print("")
    print("Begin Searchlight")

    # run the searchlight, utilizing the proper helper function
    if RSA_selection is 'model_matrix':
        sl_result = sl.run_searchlight(calc_rsm_model_matrix, pool_size=pool_size)
    elif RSA_selection is 'neural_matrices':
        sl_result = sl.run_searchlight(calc_rsm_neural_matrices, pool_size=pool_size)

    print("End Searchlight\n")

    # stop timer
    end_time = time.time()

    # number of individual searchlight cluster
    print("Number of searchlights run: " + str(len(sl_result[small_mask==1])))
    # duration of searchlight analysis
    print('Total searchlight duration (including start up time):\n %.4f' % (end_time - begin_time))
    # this is an additional check to make sure the initial calculations regarding the maximum size of the searchlight cluster are accurate
    # can remove later or just leave for sanity
    print("Maximum size of searchlight ""Ball"" is " + str(cfg.searchlight_cluster_size))


    # convert 'None' and int array to 'NaN' and double then replace 'NaN' with 0
    result_vol = sl_result.astype('double')
    result_vol[np.isnan(result_vol)] = 0
    # convert results to NiBabel NIfTI image with original affine matrix
    sl_nii = nib.Nifti1Image(result_vol, affine_mat)
    # set voxel size in header
    # you can access the zooms data through sl_nii.header even though you set it through 'hdr'
    hdr = sl_nii.header
    hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2]))


    print("Saving...")

    # save data
    nib.save(sl_nii, output_name)

    print("Saved!")
    print("---")
