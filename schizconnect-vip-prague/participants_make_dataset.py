#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 09:42:31 CET 2021

@author: edouard.duchesnay@cea.fr

Sources:


Population description
             TIV        age
sex
0.0  1499.599429  33.117413
1.0  1326.079475  31.387403
                       TIV        age
diagnosis
FEP            1444.038633  29.186047
control        1432.053401  31.370235
schizophrenia  1421.522848  34.506979
{'control': 420,
 'schizophrenia': 275,
 'FEP': 43}
"""

import os
import os.path
import glob
import click
import numpy as np
import pandas as pd

from nitk import bids


#%% INPUTS:

STUDY_PATH = '/neurospin/psy_sbox/schizconnect-vip-prague'
CLINIC_CSV = '/neurospin/psy_sbox/all_studies/phenotype/phenotypes_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START_20201223.tsv'
NII_FILENAMES = glob.glob("/neurospin/psy/schizconnect-vip-prague/derivatives/cat12-12.6_vbm/sub-*/mri/mwp1*.nii")
N_SUBJECTS = 738
assert len(NII_FILENAMES) == 738

#%% OUTPUTS:

OUTPUT_DIR = STUDY_PATH


#%% Make participants file

@click.command()
@click.option('--output', type=str, help='Output dir', default=OUTPUT_DIR)
@click.option('--dry', is_flag=True, help='Dry Run: no files are written, only check')
def make_participants(output, dry):
    """Make participants file.
    1. Read CLINIC_CSV agregated by Anton, checked with Laurie Anne,
    2. Read VBM images
    3. => Intersection
    6. Save as participants.tsv

    Returns
    -------
    DataFrame: participants
    """

    #%% 1. Read Clinic

    participants = pd.read_csv(CLINIC_CSV, sep='\t')
    participants.participant_id = participants.participant_id.astype(str)

    assert participants.shape == (3857, 46)
    # rm subjects with missing sex, age, site or diagnosis
    participants = participants[participants.sex.notnull() &
                                participants.age.notnull() &
                                participants.site.notnull() &
                                participants.diagnosis.notnull()]
    assert participants.shape == (2663, 46)

    participants = participants[participants.study.isin(['SCHIZCONNECT-VIP', 'PRAGUE'])]
    assert participants.shape == (739, 46)


    #%% Read mwp1
    ni_schizconnect_filenames = NII_FILENAMES
    assert len(ni_schizconnect_filenames) == N_SUBJECTS

    ni_schizconnect_df = pd.DataFrame([pd.Series(bids.get_keys(filename))
                                for filename in ni_schizconnect_filenames])

    # Keep only participants with processed T1
    participants = pd.merge(participants, ni_schizconnect_df, on="participant_id")
    assert participants.shape == (N_SUBJECTS, 48)

    #%% Read Total Imaging volumes
    vol_cols = ["participant_id", 'TIV', 'CSF_Vol', 'GM_Vol', 'WM_Vol']

    tivo_schizconnect = pd.read_csv(os.path.join(STUDY_PATH,
        'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')[vol_cols]
    tivo_schizconnect.participant_id = tivo_schizconnect.participant_id.astype(str)

    # assert tivo_icaar.shape == (171, 6)
    # assert len(ni_icaar_filenames) == 171

    assert tivo_schizconnect.shape ==  (N_SUBJECTS, 5)

    participants = pd.merge(participants, tivo_schizconnect, on="participant_id")
    assert participants.shape == (N_SUBJECTS, 52)

    # Save This one as the participants file
    participants_filename = os.path.join(output, "participants.tsv")
    if not dry:
        print("======================")
        print("= Save data to: %s" % participants_filename)
        print("======================")
        participants.to_csv(participants_filename, index=False, sep="\t")
    else:
        print("= Dry run do not save to %s" % participants_filename)

    # Sex mapping:
    # participants['sex'] = participants.sex.map({0:"M", 1:"F"})
    print(participants[["sex", "TIV", 'age']].groupby('sex').mean())
    print(participants[["diagnosis", "TIV", 'age']].groupby('diagnosis').mean())
    print({lev:np.sum(participants["diagnosis"]==lev) for lev in participants["diagnosis"].unique()})

    return participants

if __name__ == "__main__":
    # execute only if run as a script
    participants = make_participants()
