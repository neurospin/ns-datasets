import os
import pandas as pd

## Make IXI phenotype with TIV
IXI_PATH = '/neurospin/psy_sbox/ixi/'
OUTPUT_PATH = '/neurospin/psy_sbox/hc/ixi/IXI_t1mri_mwp1_participants.csv'
pheno_ixi = pd.read_csv(os.path.join(IXI_PATH, 'participants.tsv'), sep='\t')
meta_data_ixi = pd.read_csv(os.path.join(IXI_PATH, 'tiv.csv'), sep=',')
pheno_ixi.sex = pheno_ixi.sex.map({'M': 0, 'F': 1})
pheno_ixi['study'] = 'IXI'
pheno_ixi['site'] = 'LONDON'
pheno_ixi['diagnosis'] = 'control'
meta_data_ixi = meta_data_ixi.rename(columns={'TIV': 'tiv'})
pheno_ixi = pd.merge(pheno_ixi, meta_data_ixi, on='participant_id', how='left', sort=False)

pheno_ixi.to_csv(OUTPUT_PATH, sep='\t', index=False)
