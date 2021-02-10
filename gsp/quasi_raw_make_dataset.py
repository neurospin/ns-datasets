from utils import quasi_raw_nii2npy
from gsp.cat12vbm_make_dataset import phenotype_path, sep, dataset, output_path, id_type

## Initial .nii files
nii_regex_path = '/neurospin/psy_sbox/'+dataset.upper()+'/derivatives/quasi-raw/sub-*/ses-*/anat/*preproc-linear*.nii.gz'
## Eventual QC file
qc_path = '/neurospin/psy_sbox/'+dataset+'/derivatives/cat12-12.6_vbm_qc/qc.tsv'

quasi_raw_nii2npy(nii_regex_path, phenotype_path, dataset, output_path,
            qc=qc_path, sep=sep, id_type=id_type, check=dict(shape=(182, 218, 182), zooms=(1, 1, 1)))