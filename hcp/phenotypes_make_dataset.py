import os
import pandas as pd

## Make HCP phenotype
HCP_PATH = '/neurospin/psy_sbox/hcp'
sex_dx_site = pd.read_csv(os.path.join(HCP_PATH, 'participants.tsv'), sep='\t')
age = pd.read_csv(os.path.join(HCP_PATH, 'hcp_restricted_data.csv'), sep=',')
age = age.rename(columns={'Subject': 'participant_id', 'Age_in_Yrs': 'age'})
sex_dx_site.drop(columns='age', inplace=True)
age_sex_dx_site_study_tiv = pd.merge(sex_dx_site, age, on="participant_id", how='left', validate='1:1')
age_sex_dx_site_study_tiv = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.tiv.isna() &
                                                      ~age_sex_dx_site_study_tiv.diagnosis.isna() &
                                                      ~age_sex_dx_site_study_tiv.site.isna()
                                                      ]
assert len(age_sex_dx_site_study_tiv) == 1113 and age_sex_dx_site_study_tiv.participant_id.is_unique
age_sex_dx_site_study_tiv.to_csv(os.path.join(HCP_PATH, 'HCP_t1mri_mwp1_participants.csv'), sep='\t', index=False)