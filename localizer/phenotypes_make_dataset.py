import os
import pandas as pd

## Make LOCALIZER phenotype
LOCALIZER_PATH = "/neurospin/psy_sbox/localizer"
## Dataset dependent
age_sex_dx_site = pd.read_csv(os.path.join(LOCALIZER_PATH, 'participants.tsv'), sep='\t')
age_sex_dx_site.sex = age_sex_dx_site.sex.map({'M':0, 'F':1})
age_sex_dx_site['study'] = 'LOCALIZER'
age_sex_dx_site.age = age_sex_dx_site.age.astype(float, errors='ignore')
age_sex_dx_site['diagnosis'] = 'control'
assert age_sex_dx_site.participant_id.is_unique
## Dataset independent
tiv = pd.read_csv(os.path.join(LOCALIZER_PATH,'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
## Merge all
age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex_dx_site, on='participant_id', how='left', sort=False, validate='1:1')
participants_columns_list = list(age_sex_dx_site.columns)
participants_columns_list = [participants_columns_list[0]]+["session", "run"]+participants_columns_list[1:]
rois_columns_list = list(tiv.columns)
age_sex_dx_site_study = age_sex_dx_site_study_tiv[participants_columns_list]
rois = age_sex_dx_site_study_tiv[rois_columns_list]
age_sex_dx_site_study.reset_index().to_csv(os.path.join(os.path.join(LOCALIZER_PATH,"phenotype"), 'participants_phenotypes.tsv'), sep='\t', index=False, columns=participants_columns_list)
rois.reset_index().to_csv(os.path.join(os.path.join(LOCALIZER_PATH, "phenotype"), 'participants_ROIS.tsv'), sep='\t', index=False, columns=rois_columns_list)
