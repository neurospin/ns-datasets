import os
import pandas as pd
import numpy as np

## Make CoRR phenotype with TIV
CORR_PATH = '/neurospin/psy_sbox/CoRR'
## Dataset-dependent
sex_site = pd.read_csv(os.path.join(CORR_PATH, 'participants.tsv'), sep='\t')
age = pd.read_csv(os.path.join(CORR_PATH, 'MR_sessions.tsv'), sep='\t')
filter_age = age['MR ID'].str.contains('baseline')
age = age[filter_age]
sex_site = sex_site.rename(columns={'M/F': 'sex'})
sex_site.sex = sex_site.sex.map({'M':0, 'F':1})
sex_site = sex_site[~sex_site.sex.isna()] # Erases 7 participants
sex_site['site'] = sex_site['Subject'].str.extract(r'(\w+)_([0-9]+)', expand=True)[0]
sex_site['participant_id'] = sex_site['Subject'].str.extract(r'(\w+)_([0-9]+)', expand=True)[1]
assert len(sex_site.participant_id) == len(set(sex_site.participant_id)) == 1379 # Unique participant_id
age_sex_dx_site_study = pd.merge(sex_site, age, on='Subject', how='left', sort=False, validate='1:1')
age_sex_dx_site_study = age_sex_dx_site_study.rename(columns={'Age': 'age'})
age_sex_dx_site_study['study'] = 'CoRR'
age_sex_dx_site_study['diagnosis'] = 'control'
assert np.all(~age_sex_dx_site_study.age.isna() & ~age_sex_dx_site_study.sex.isna())
assert len(age_sex_dx_site_study.participant_id) == len(set(age_sex_dx_site_study.participant_id)) == 1379
## Dataset-independent
tiv = pd.read_csv(os.path.join(CORR_PATH, 'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
## Merge everything
age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex_dx_site_study, on='participant_id', how='left', sort=False,
                                     validate='m:1')
assert len(age_sex_dx_site_study_tiv) == len(tiv) == 2698
age_sex_dx_site_study_tiv = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.TIV.isna()]
assert len(age_sex_dx_site_study_tiv) == len(tiv) - 13
age_sex_dx_site_study_tiv.to_csv(os.path.join(CORR_PATH, 'CoRR_t1mri_mwp1_participants.csv'), sep='\t', index=False)
