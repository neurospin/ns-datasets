import os
import pandas as pd
import numpy as np

## Make ICBM phenotype
ICBM_PATH = "/neurospin/psy_sbox/icbm"
## Dataset dependent
age_sex_dx = pd.read_csv(os.path.join(ICBM_PATH, 'participants.tsv'), sep='\t')
age_sex_dx['participant_id'] = age_sex_dx['Subject'].str.replace('_','')
age_sex_dx = age_sex_dx.rename(columns={'Sex': 'sex', 'Age': 'age'})
age_sex_dx.sex = age_sex_dx.sex.map({'M': 0, 'F': 1})
age_sex_dx['diagnosis'] = 'control'
age_sex_dx.drop_duplicates(subset='participant_id', keep='first', inplace=True)
assert len(age_sex_dx.participant_id) == len(set(age_sex_dx.participant_id)) == 640
age_sex_dx['study'] = 'ICBM'
age_sex_dx['site'] = 'ICBM'
## Dataset independent
tiv = pd.read_csv(os.path.join(ICBM_PATH,'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
## Merge all
age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex_dx, on='participant_id', how='left', sort=False, validate='m:1')
age_sex_dx_site_study_tiv = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.TIV.isna()]
assert len(age_sex_dx_site_study_tiv) == len(tiv) == 982
age_sex_dx_site_study_tiv.to_csv(os.path.join(ICBM_PATH, 'ICBM_t1mri_mwp1_participants.csv'), sep='\t', index=False)
