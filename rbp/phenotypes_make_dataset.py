import os
import pandas as pd
import numpy as np

## Make RBP phenotype
RBP_PATH = '/neurospin/psy_sbox/rbp'
## Dataset dependent
age_sex = pd.read_csv(os.path.join(RBP_PATH, 'participants.tsv'), sep='\t')
age_sex = age_sex.rename(columns = {'Age': 'age', 'Gender': 'sex', 'ID': 'participant_id'})
age_sex.sex = age_sex.sex.map({'M': 0, 'F': 1})
age_sex['study']='RBP'
age_sex['site']='RBP'
age_sex['diagnosis']='control'
age_sex.participant_id = age_sex.participant_id.astype(str)
assert age_sex.participant_id.is_unique
## Dataset-independent
tiv = pd.read_csv(os.path.join(RBP_PATH,'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str) # 9 bad segmentations with no TIV
## Merge all
age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex, on='participant_id', how='left', sort=False, validate='m:1')
age_sex_dx_site_study_tiv = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.TIV.isna()]
assert len(age_sex_dx_site_study_tiv) == len(tiv) - 2 # 2 participants not found in participants.tsv (18,40)
age_sex_dx_site_study_tiv.to_csv(os.path.join(RBP_PATH, 'RBP_t1mri_mwp1_participants.csv'), sep='\t', index=False)
