import os
import pandas as pd

## Make CANDI phenotype
CANDI_PATH = '/neurospin/psy_sbox/candi'
age_sex_dx = pd.read_csv(os.path.join(CANDI_PATH, 'SchizBull_2008_Demographics_V1.1.csv'), sep=',')
age_sex_dx = age_sex_dx.rename(columns={'Gender': 'sex', 'Age': 'age'})
age_sex_dx['participant_id'] = age_sex_dx.Subject.str.replace('_', '')
age_sex_dx['diagnosis'] = age_sex_dx.Subject.str.extract('(\w+)\_[0-9]+')[0]
age_sex_dx.diagnosis = age_sex_dx.diagnosis.map({'HC': 'control', 'BPDwoPsy': 'bipolar disorder without psychosis',
                                                 'BPDwPsy': 'bipolar disorder with psychosis', 'SS': 'schizophrenia'})
age_sex_dx.sex = age_sex_dx.sex.map({'male': 0, 'female': 1})
age_sex_dx['site'] = 'CANDI'
age_sex_dx['study'] = 'CANDI'
## Dataset independent
tiv = pd.read_csv(os.path.join(CANDI_PATH,'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
## Merge all
age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex_dx, on='participant_id', how='left', sort=False, validate='m:1')
age_sex_dx_site_study_tiv = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.TIV.isna() &
                                                      ~age_sex_dx_site_study_tiv.diagnosis.isna()]
assert len(age_sex_dx_site_study_tiv) == len(tiv)
age_sex_dx_site_study_tiv.to_csv(os.path.join(CANDI_PATH, 'CANDI_t1mri_mwp1_participants.csv'), sep='\t', index=False)
