import os
import pandas as pd
import numpy as np

## Make ABIDE2 phenotype with TIV
ABIDE2_PATH = '/neurospin/psy_sbox/abide2/'
OUTPUT_PATH = '/neurospin/psy_sbox/abide2/ABIDE2_t1mri_mwp1_participants.csv'
pheno_abide2 = pd.read_csv(os.path.join(ABIDE2_PATH, 'participants.tsv'), sep='\t')
pheno_abide2.participant_id = pheno_abide2.participant_id.astype(str)
assert pheno_abide2.participant_id.is_unique
## Dataset-independent
tiv = pd.read_csv(os.path.join(ABIDE2_PATH, 'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
## Merge all
age_sex_dx_site_study_tiv = pd.merge(tiv, pheno_abide2, on='participant_id', how='left', sort=False, validate='m:1')
age_sex_dx_site_study_tiv = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.TIV.isna() &
                                                      ~age_sex_dx_site_study_tiv.diagnosis.isna()]
assert len(age_sex_dx_site_study_tiv) == len(tiv) - 11
age_sex_dx_site_study_tiv.to_csv(os.path.join(ABIDE2_PATH, 'ABIDE2_t1mri_mwp1_participants.csv'), sep='\t', index=False)
