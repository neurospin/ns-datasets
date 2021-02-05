import os
import pandas as pd
import numpy as np

## Make MPI_LEIPZIG phenotype
MPI_PATH = '/neurospin/psy_sbox/mpi-leipzig'
## Dataset dependent
age_sex_dx = pd.read_csv(os.path.join(MPI_PATH, 'participants.tsv'), sep='\t')
age_sex_dx = age_sex_dx.rename(columns={'gender': 'sex', 'age (5-year bins)': 'age'})
age_sex_dx.sex = age_sex_dx.sex.map({'M': 0, 'F': 1})
age_sex_dx['participant_id'] = age_sex_dx['participant_id'].str.replace('sub-','').astype(int).astype(str)
assert len(age_sex_dx) == 318
age_sex_dx = age_sex_dx[~age_sex_dx.age.isna()]
assert len(age_sex_dx) == 316 and age_sex_dx.participant_id.is_unique
age_sex_dx.age = age_sex_dx.age.str.split('-').apply(lambda x: np.mean([int(e) for e in x]))
age_sex_dx['site'] = 'MPI-LEIPZIG'
age_sex_dx['study'] = 'MPI-LEIPZIG'
age_sex_dx['diagnosis'] = 'control'
## Dataset independent
tiv = pd.read_csv(os.path.join(MPI_PATH,'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
## Merge all
age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex_dx, on='participant_id', how='left', sort=False, validate='m:1')
age_sex_dx_site_study_tiv = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.TIV.isna() &
                                                      ~age_sex_dx_site_study_tiv.diagnosis.isna()]
assert len(age_sex_dx_site_study_tiv) == len(tiv) - 2
age_sex_dx_site_study_tiv.to_csv(os.path.join(MPI_PATH, 'MPI-LEIPZIG_t1mri_mwp1_participants.csv'), sep='\t', index=False)
