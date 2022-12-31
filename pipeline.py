"""
DESCRIPTION
takes t-maps (in subject space) for single trials and creates neural representational dissimilarity matrices (RDMs)
these can be compared against hypothetical models for how representations might be "structured" or other RDMs
the data sampled for the neural RDMs comes from a whole brain searchlight
the goal is to locate regions in the brain which match either the hypothetical model or the other RDMs
general order of operations:
-initialize parameters
-check that parameters match allowable values and confirm their selection
-generate certain string values that are needed later
-initialize, then run whole brain searchlight representational similarity analysis (RSA)
"""

# basic imports, including code to help run the RSA and check your starting parameters
import sys
import os

from RSA_helpers import option_check, expanded_names
from full_RSA import full_RSA
import cfg

"""
Description of various parameters that need to be initialized
For most of these, there is a discrete set of options allowed

Parameters
----------
sub_num: int [list]
     a vector of subject number(s). put them in brackets e.g., "[2]" or "[1,4,5]"
project_name: str
    the name used to save the project folder
RSA_selection: str
    comparison to make for RSA - compare neural RDM to a hypothetical model, or to another neural RDM
        : 'model_matrix', 'neural_matrices'
model_selection: str
    which model to use for model comparison - currently only works for "category" and "spectrum"
        : 'category', 'motor', 'spectrum', None
first_phase, second_phase: str
    names of phase from the task for comparison
        : 'train', 'test', None
first_run, second_run: int/str
    specific run number (1-4) or 'avg' of runs
        : 1, 2, 3, 4, 'avg_1234', 'avg_234', None
mask_selection: str
    which of several mask options to select. whole brain is generated on the fly using nilearn, the other two are for debugging only
        : 'whole_brain', 'single_voxel', 'all_ones'
all_dots_heght: str
    subjects either had rotation_dots as defining features, or color_height. this setting determines which groups to process
    everybody all at once (all), rotation_dots group (dots), or color_height (height)
        : 'all', 'dots', 'height'
data_path: str
    base location of files
base_out_path: str
    base location of where you want output
suffix: str
    file extension for data-to-be-loaded
sl_rad: int
    the number of voxels not counting the center (e.g., rad=2 means 5 voxels wide at center
max_blk_edge: int
    see comment
pool_size: int
    see comment
"""

# subject group numbers - reference so you don't have to open the subject spreadsheet
# rot_dots = [2,3,8,9,10,14,15,17,18,22,23,24]
# col_hght = [4,5,6,7,11,12,13,16,19,20,21,25]

# this is where you establish the basic starting parameters following the guidelines above
sub_num = [2]
project_name = 'delete'
RSA_selection = 'model_matrix'
model_selection = 'category'
first_phase = 'train'
first_run = 'avg_234'
second_phase = None
second_run = None
mask_selection = 'whole_brain'
all_dots_height = 'dots'

if all_dots_height is 'all':
    derivatives_folder = 'derivatives'
elif all_dots_height is 'dots':
    derivatives_folder = 'derivatives_spectrum_dots'
elif all_dots_height is 'height':
    derivatives_folder = 'derivatives_spectrum_height'

# path where bin files are located
data_path = os.path.join('/','media','shareDrive2','data','overshadowing','dissertation','dataset',
                         'derivatives','derivatives','1st_level','02',derivatives_folder)

# set up output folder
base_out_path = os.path.join(os.path.expanduser('~'),'brainiak_results','searchlight_results',project_name,RSA_selection)

# this would need to change if you used something other than SPM i.e., '.nii.gz'
suffix = '.nii'

# radius
sl_rad = 4

# "When the searchlight function carves the data up into chunks,
# it doesn't distribute only a single searchlight's worth of data.
# it creates a block of data, with the edge length specified by this variable"
max_blk_edge = 5

# maximum number of cores running on a block (the blocks defined by max_blk_edge?)
pool_size = 1

# check your selections for all the different options
option_check(RSA_selection,model_selection,first_phase,first_run,second_phase,second_run,mask_selection)
# get some strings formatted
first_run_name, second_run_name = expanded_names(first_run,second_run)

# display parameters of analysis for confirmation
print("Please confirm processing settings:")
print(f"You are running:")
for subs in sub_num:
    print(f"sub-{subs:>02}")
print(f"Type of RSA: {RSA_selection}")
print(f"Mask selection: {mask_selection}")
if RSA_selection is 'model_matrix':
    print(f"You are comparing '{first_run_name}' data from the '{first_phase}' phase to a '{model_selection}' model")
elif RSA_selection is 'neural_matrices':
    print(f"You are comparing '{first_run_name}' data from the '{first_phase}' phase to '{second_run_name}' data "
          f"from the '{second_phase}' phase")
print(f"General output path:\n {base_out_path}")
if RSA_selection is 'neural_matrices' and first_run == second_run and first_phase == second_phase:
    sanity_check = input("Are you performing a sanity check? Your phase and run RSA_selections are identical.")

pause = input("Press 'Enter' to confirm settings:\n")
if pause != '':
    raise Exception("Please check settings and try again")

# run RSA for either a single subject or a group of subjects
for sub in sub_num:
    sub_name = f'sub-{sub:>02}'
    full_RSA(RSA_selection,model_selection,first_phase,first_run,second_phase,second_run,mask_selection,
             data_path,base_out_path,suffix,sl_rad,max_blk_edge,pool_size,sub_name,first_run_name,
             second_run_name,all_dots_height)
