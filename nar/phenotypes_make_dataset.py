import os
import pandas as pd
import numpy as np

## Make NAR phenotype
NAR_path = '/neurospin/psy_sbox/nar'
## Dataset dependent
age_sex_dx = pd.read_csv(os.path.join(NAR_path, 'participants.tsv'), sep='\t')
assert len(age_sex_dx) == 315
age_sex_dx.sex =  age_sex_dx.sex.str.split(',', expand=True)[0]
age_sex_dx.sex = age_sex_dx.sex.map({'M':0, 'F':1})
age_sex_dx.age = age_sex_dx.age.str.replace('(n\/a,)*(n\/a)?', '') # Removes the n/a unformatted
age_sex_dx = age_sex_dx[~age_sex_dx.age.isna() & (age_sex_dx.age.str.len()>0)] # Removes 4 participants
age_sex_dx.age = age_sex_dx.age.str.split(',').apply(lambda x: np.mean([int(e) for e in x]))
assert len(age_sex_dx) == 311 and age_sex_dx.participant_id.is_unique
assert np.all(~age_sex_dx.age.isna() & ~age_sex_dx.sex.isna())
age_sex_dx['site'] = 'NAR'
age_sex_dx['study'] = 'NAR'
age_sex_dx['diagnosis'] = 'control'
age_sex_dx.participant_id = age_sex_dx.participant_id.str.replace('sub-', '').astype(int).astype(str)
## Dataset-independent
tiv = pd.read_csv(os.path.join(NAR_path, 'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
## Merge all
age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex_dx, on='participant_id', how='left', sort=False, validate='m:1')
age_sex_dx_site_study_tiv = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.TIV.isna()]
assert len(age_sex_dx_site_study_tiv) == len(tiv) - 4
age_sex_dx_site_study_tiv.to_csv(os.path.join(NAR_path, 'NAR_t1mri_mwp1_participants.csv'), sep='\t', index=False)
