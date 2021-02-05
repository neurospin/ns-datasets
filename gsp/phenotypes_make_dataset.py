import os
import pandas as pd
import numpy as np

## Make GSP phenotype
GSP_PATH = '/neurospin/psy_sbox/GSP'
## Dataset dependent
age_sex = pd.read_csv(os.path.join(GSP_PATH, 'participants_ses-1.tsv'), sep='\t')
age_sex_rescanned = pd.read_csv(os.path.join(GSP_PATH, 'participants_ses-1_ses-2.tsv'), sep='\t')
age_sex['participant_id'] = age_sex.Subject_ID.str.extract('Sub([0-9]+)_*')
age_sex_rescanned['participant_id'] = age_sex_rescanned.Subject_ID.str.extract('Sub([0-9]+)_*')
# Ensures that we have all the data we need from age_sex
assert age_sex.participant_id.is_unique and set(age_sex.participant_id) >= set(age_sex_rescanned.participant_id)
assert np.all(~age_sex.Age_Bin.isna()) and np.all(~age_sex.Sex.isna())
age_sex = age_sex.rename(columns={'Age_Bin': 'age', 'Sex': 'sex'})
age_sex.sex = age_sex.sex.map({'M': 0, 'F': 1})
age_sex.participant_id = age_sex.participant_id.astype(int).astype(str)
age_sex['diagnosis'] = 'control'
age_sex['study'] = 'GSP'
age_sex['site'] = 'HUV'
## Dataset-independent
tiv = pd.read_csv(os.path.join(GSP_PATH,'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
## Merge all
age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex, on='participant_id', how='left', sort=False, validate='m:1')
age_sex_dx_site_study_tiv = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.TIV.isna()]
assert len(age_sex_dx_site_study_tiv) == len(tiv)
age_sex_dx_site_study_tiv.to_csv(os.path.join(GSP_PATH, 'GSP_t1mri_mwp1_participants.csv'), sep='\t', index=False)
