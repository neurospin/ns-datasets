import os
import pandas as pd
import numpy as np

## Make OASIS 3 phenotype
OASIS3_PATH = '/neurospin/psy_sbox/oasis3'
## Dataset dependent
dx = pd.read_csv(os.path.join(OASIS3_PATH, 'adrc_clinical_data.csv'), sep=',')
sex = pd.read_csv(os.path.join(OASIS3_PATH, 'participants.tsv'), sep='\t')
age = pd.read_csv(os.path.join(OASIS3_PATH, 'MR_sessions.tsv'), sep='\t')
age = age.rename(columns={'Age': 'age', 'Subject': 'participant_id'})
age['session'] = age['MR ID'].str.extract('(OAS[0-9]+)\_MR\_(d[0-9]+)')[1]
sex = sex.rename(columns={'Subject': 'participant_id', 'M/F': 'sex'})
sex.sex = sex.sex.map({'M': 0, 'F': 1})
age_sex = pd.merge(age, sex, on='participant_id', how='left', sort=False, validate='m:1')
age_sex = age_sex[~age_sex.age.isna() & ~age_sex.sex.isna()]
dx = dx.rename(columns={'Subject': 'participant_id'})
dx.drop(columns=['Age', 'Date'], inplace=True)

# Selects only patients who kept their CDR constant (no CTL to AD or AD to CTL and AD is defined as CDR > 0)
# cf Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI Data in Young, Middle Aged, Nondemented,
# and Demented Older Adults, Daniel S. Marcus, Journal of Cognitive Neuroscience, 2007
cdr_min = dx.groupby('participant_id', as_index=False)['cdr'].agg(np.min)
cdr_max = dx.groupby('participant_id', as_index=False)['cdr'].agg(np.max)
constant_cdr_participants_id = cdr_min[cdr_min.cdr == cdr_max.cdr]
# In line with https://www.oasis-brains.org/files/OASIS-3_Imaging_Data_Dictionary_v1.8.pdf
assert (constant_cdr_participants_id.cdr==0).sum() == 605 and (constant_cdr_participants_id.cdr==0.5).sum() == 66
# Filters the participants
age_sex = age_sex[age_sex.participant_id.isin(constant_cdr_participants_id.participant_id)]
age_sex['diagnosis'] = 'control'
filter_ad = age_sex.participant_id.isin(constant_cdr_participants_id[constant_cdr_participants_id.cdr > 0].participant_id)
age_sex.loc[filter_ad, 'diagnosis'] = 'AD'
assert np.all(age_sex[age_sex.diagnosis.eq('control')].participant_id.isin(constant_cdr_participants_id[constant_cdr_participants_id.cdr==0].participant_id))
assert np.all(age_sex[age_sex.diagnosis.eq('AD')].participant_id.isin(constant_cdr_participants_id[constant_cdr_participants_id.cdr>0].participant_id))
age_sex['study'] = 'OASIS3'
age_sex['site'] = 'OASIS3'
## Dataset-independent
tiv = pd.read_csv(os.path.join(OASIS3_PATH,'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
## Merge all
age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex, on=['participant_id', 'session'], how='left', sort=False, validate='m:1')
age_sex_dx_site_study_tiv = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.TIV.isna() &
                                                      ~age_sex_dx_site_study_tiv.diagnosis.isna()]
age_sex_dx_site_study_tiv.to_csv(os.path.join(OASIS3_PATH, 'OASIS3_t1mri_mwp1_participants.csv'), sep='\t', index=False)

