from utils import cat12_nii2npy

## Dataset name for the generated output .npy and .csv files
dataset = 'abide2'
## Initial .nii files
nii_regex_path = '/neurospin/psy_sbox/'+dataset+'/derivatives/cat12-12.6_vbm/sub-*/ses-*/anat/mri/mwp1*.nii'
## Phenotype obtained with <phenotypes_make_dataset.py>
phenotype_path = '/neurospin/psy_sbox/{dataset}/{dataset_u}_t1mri_mwp1_participants.csv'.format(dataset=dataset,
                                                                                                dataset_u=dataset.upper())
## Separator the the phenotype.csv file
sep ='\t'
## Where do we put the generated files ?
output_path = '/neurospin/tmp/psy_sbox/all_studies/derivatives/arrays'
## Add the QC file if available
qc_path = '/neurospin/psy_sbox/{dataset}/derivatives/cat12-12.6_vbm_qc/qc.tsv'.format(dataset=dataset)
## How the participant's id are formatted ? (Either <int> or <str>)
id_type = int

cat12_nii2npy(nii_regex_path, phenotype_path, dataset, output_path,
            qc=qc_path, sep=sep, id_type=id_type, check=dict(shape=(121, 145, 121),zooms=(1.5, 1.5, 1.5)))