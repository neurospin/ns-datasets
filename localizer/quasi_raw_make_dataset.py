import sys
sys.path.append("/neurospin/psy_sbox/git/ns-datasets")
import os
import pandas as pd
from utils import quasi_raw_nii2npy
from localizer.cat12vbm_make_dataset import sep, dataset, output_path, id_type


## Select LOCALIZER phenotype
LOCALIZER_PATH = "/neurospin/psy_sbox/localizer"
## Dataset dependent
age_sex_dx_site = pd.read_csv(os.path.join(LOCALIZER_PATH, 'participants.tsv'), sep='\t')
age_sex_dx_site.sex = age_sex_dx_site.sex.map({'M':0, 'F':1})
age_sex_dx_site['study'] = 'LOCALIZER'
age_sex_dx_site = age_sex_dx_site[~age_sex_dx_site.age.isna() & ~age_sex_dx_site.age.eq('None')] # 4 participants have 'None' age
age_sex_dx_site.age = age_sex_dx_site.age.astype(float)
age_sex_dx_site['diagnosis'] = 'control'
assert age_sex_dx_site.participant_id.is_unique
## Dataset independent
tiv = pd.read_csv(os.path.join(LOCALIZER_PATH,'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
## Merge all
age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex_dx_site, on='participant_id', how='left', sort=False, validate='1:1')
phenotype_pd = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.TIV.isna() &
                                                      ~age_sex_dx_site_study_tiv.diagnosis.isna()]
assert len(phenotype_pd) == len(tiv) - 4

## Initial .nii files
nii_regex_path = '/neurospin/psy_sbox/'+dataset+'/derivatives/quasi-raw/sub-*/ses-*/anat/*preproc-linear*.nii.gz'
## Eventual QC file
qc_path = '/neurospin/psy_sbox/'+dataset+'/derivatives/cat12-12.6_vbm_qc/qc.tsv'

quasi_raw_nii2npy(nii_regex_path, phenotype_pd, dataset, output_path,
            qc=qc_path, sep=sep, id_type=id_type, check=dict(shape=(182, 218, 182), zooms=(1, 1, 1)))
