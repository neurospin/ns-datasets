import sys
sys.path.append("/neurospin/psy_sbox/git/ns-datasets")
from utils import cat12_nii2npy
import os
import pandas as pd

## Select LOCALIZER phenotype
LOCALIZER_PATH = "/neurospin/psy_sbox/localizer"
## Dataset dependent
age_sex_dx_site = pd.read_csv(os.path.join(LOCALIZER_PATH, 'participants.tsv'), sep='\t')
age_sex_dx_site.sex = age_sex_dx_site.sex.map({'M':0, 'F':1})
age_sex_dx_site['study'] = 'LOCALIZER'
age_sex_dx_site = age_sex_dx_site[~age_sex_dx_site.age.isna() & ~age_sex_dx_site.age.eq('None')] # 4 participants have 'None' age
age_sex_dx_site.age = age_sex_dx_site.age.astype(float)
age_sex_dx_site['diagnosis'] = 'control'
assert age_sex_dx_site.participant_id.is_unique
## Dataset independent
tiv = pd.read_csv(os.path.join(LOCALIZER_PATH,'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
## Merge all
age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex_dx_site, on='participant_id', how='left', sort=False, validate='1:1')
phenotype_pd = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.TIV.isna() &
                                                      ~age_sex_dx_site_study_tiv.diagnosis.isna()]
assert len(phenotype_pd) == len(tiv) - 4

## Dataset name for the generated output .npy and .csv files
dataset = 'localizer'
## Initial .nii files
nii_regex_path = '/neurospin/psy/'+dataset+'/derivatives/cat12-12.6_vbm/sub-*/ses-*/anat/mri/mwp1*.nii'
## Separator the the phenotype.csv file
sep ='\t'
## Where do we put the generated files ?
output_path = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm_julietest'
## Add the QC file if available
qc_path = '/neurospin/psy_sbox/{dataset}/derivatives/cat12-12.6_vbm_qc/qc.tsv'.format(dataset=dataset)
## How the participant's id are formatted ? (Either <int> or <str>)
id_type = str

cat12_nii2npy(nii_regex_path, phenotype_pd, dataset, output_path,
            qc=qc_path, sep=sep, id_type=id_type, check=dict(shape=(121, 145, 121),zooms=(1.5, 1.5, 1.5)))


