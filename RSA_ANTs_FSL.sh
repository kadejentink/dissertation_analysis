#!/bin/bash
# files need to be in a loose BIDS-style format
# DESCRIPTION
# analysis pipeline from single trial t-maps to group t-test
# general order of operations:
# -activate proper anaconda virtual environment
# -run RSA analysis
# -transform subject space RSA results to MNI space
# -run group one-sample t-test

# activate specific environment which has the dependencies
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE"/etc/profile.d/conda.sh
conda activate brainiak

# run the python pipeline that executes the RSA
# SCRIPT NEEDS PARAMETERS SET - CHECK BEFORE RUNNING
# some should match the variables initialized below - recommend initialize parameters in this script and pipeline at the same time
# (e.g., the value for "rad" needs to match the value that was used in the python script)
python pipeline.py

# pipeline.py outputs the RSA results to a specific name_of_project_folder
results_path='/path/to/RSA/results'
# this is the name of the project folder created in pipeline.py
project_name='name_of_project_folder'
# pipeline.py runs either a comparison against a "model_matrix" or compares two "neural_matrices"
RSA_selection='RSA_type'
# these would be the two components of the RSA - this name is generated in pipeline.py
# e.g., "ses-01_run-01-model_matrix" would mean that you are comparing the RDM for run-01 against a model matrix
run_comparison_name='name_of_both_RSA_components'
# same value used in pipeline.py - type of functional data mask used for RSA
mask_selection='whole_brain'
# same value used in pipeline.py - radius of searchlight
rad='3'

# where is the fMRIPrep derivatives folder containing the transform file (from T1w to MNI space)
transform_path='/path/to/fMRIPrep/derivatives/folder'
# the standard portion of the text in the filename
# if the file is called "sub-01_transform-file.h5" then "_transform-file.h5" is the standard portion
transform_file='_standard_text_for_transformation_file_mode-image_xfm.h5'

# run the transformation to MNI space using ANTs
# for the number of files in the folder (i.e., the number of subjects you ran a searchlight RSA on)...
for (( c=1; c<=$(ls "$results_path"/"$project_name"/"$RSA_selection"/"$run_comparison_name"/*.nii.gz | wc -l); c++)); do

	# check if output directory exists, if not, make it
	if [ ! -d "$results_path"/"$project_name"/"$RSA_selection"/"$run_comparison_name"/output_folder ]; then
  	mkdir "$results_path"/"$project_name"/"$RSA_selection"/"$run_comparison_name"/output_folder
	fi

	# create formatted value for subject ID
	subID=$(printf "%02d" $c)

	# apply ANTs transform function using transform file generated during fMRIPrep
	# uses RSA output as input (i), transform file from fMRIPrep (t), standard atlas as reference (r), and generates an output in standard space (o)
	antsApplyTransforms \
	-i "$results_path"/"$project_name"/"$RSA_selection"/"$run_comparison_name"/sub-"$subID"_"$run_comparison_name"_mask-"$mask_selection"_rad-"$rad".nii.gz \
	-t "$transform_path"/sub-"$subID"/ses-01/anat/sub-"$subID""$transform_file" \
	-r /media/shareDrive/data/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii \
	-o "$results_path"/"$project_name"/"$RSA_selection"/"$run_comparison_name"/output_folder/sub-"$subID"_"$run_comparison_name"_mask-"$mask_selection"_rad-"$rad"_space-MNI.nii.gz \
	-v
done

# move to the folder containing the transformed RSA analysis results
cd ${results_path}/${project_name}/${RSA_selection}/${run_comparison_name}/output_folder/
# merge the transformed files into one file - the necessary format for FSL to run a t-test
fslmerge -t sub-all_${run_comparison_name}_mask-${mask_selection}_rad-${rad}_space-MNI.nii.gz sub*.nii.gz

# run a one-sample t-test using FSL randomise function on the concatenated file data
randomise -i sub-all_${run_comparison_name}_mask-${mask_selection}_rad-${rad}_space-MNI.nii.gz -o sub-all_${run_comparison_name}_mask-${mask_selection}_rad-${rad}_space-MNI_randomise-T_v-5.nii.gz -1 -v 5 -T

echo ALL done
