import os
import pandas as pd
import numpy as np

## Make CNP phenotype
CNP_PATH = "/neurospin/psy_sbox/cnp"
## Dataset dependent
age_sex_dx = pd.read_csv(os.path.join(CNP_PATH, 'participants.tsv'), sep='\t')
age_sex_dx.participant_id = age_sex_dx.participant_id.str.replace('sub-', '')
age_sex_dx.diagnosis = age_sex_dx.diagnosis.map({'CONTROL': 'control', 'SCHZ': 'schizophrenia', 'BIPOLAR': 'bipolar',
                                                 'ADHD': 'adhd'})
age_sex_dx = age_sex_dx.rename(columns={'gender': 'sex'})
age_sex_dx.sex = age_sex_dx.sex.map({'M': 0, 'F': 1})
age_sex_dx['site'] = 'CNP'
age_sex_dx['study'] = 'CNP'
assert age_sex_dx.participant_id.is_unique
## Dataset independent
tiv = pd.read_csv(os.path.join(CNP_PATH,'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
## Merge all
age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex_dx, on='participant_id', how='left', sort=False, validate='1:1')
age_sex_dx_site_study_tiv = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.TIV.isna() &
                                                      ~age_sex_dx_site_study_tiv.diagnosis.isna()]
assert len(age_sex_dx_site_study_tiv) == len(tiv) == 265
age_sex_dx_site_study_tiv.to_csv(os.path.join(CNP_PATH, 'CNP_t1mri_mwp1_participants.csv'), sep='\t', index=False)
