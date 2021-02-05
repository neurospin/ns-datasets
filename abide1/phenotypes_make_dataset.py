import os
import pandas as pd
import numpy as np

## Make ABIDE1 phenotype
## Dataset dependent
ABIDE1_PATH = '/neurospin/psy_sbox/abide1/'
age_sex_dx_site = pd.read_csv(os.path.join(ABIDE1_PATH, 'Phenotypic_V1_0b.csv'), sep=',')
age_sex_dx_site = age_sex_dx_site.rename(columns={"AGE_AT_SCAN": 'age', 'SEX': 'sex', 'SITE_ID': 'site',
                                                  "DX_GROUP": "diagnosis", "SUB_ID": 'participant_id'})
age_sex_dx_site.diagnosis = age_sex_dx_site.diagnosis.map({1: 'autism', 2:'control'})
age_sex_dx_site.participant_id = age_sex_dx_site.participant_id.astype(str)
age_sex_dx_site.sex = age_sex_dx_site.sex.map({1:0, 2:1}) # 1: Male, 2: Female
age_sex_dx_site['study'] = 'ABIDE1'
assert age_sex_dx_site.participant_id.is_unique

## Dataset-independent
tiv = pd.read_csv(os.path.join(ABIDE1_PATH, 'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
## Merge all
age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex_dx_site, on='participant_id', how='left', sort=False, validate='m:1')
age_sex_dx_site_study_tiv = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.TIV.isna() &
                                                      ~age_sex_dx_site_study_tiv.diagnosis.isna() &
                                                      ~age_sex_dx_site_study_tiv.site.isna()
                                                      ]
assert len(age_sex_dx_site_study_tiv) == len(tiv) - 1 # No TIV available for participant 50818
age_sex_dx_site_study_tiv.to_csv(os.path.join(ABIDE1_PATH, 'ABIDE1_t1mri_mwp1_participants.csv'), sep='\t', index=False)
