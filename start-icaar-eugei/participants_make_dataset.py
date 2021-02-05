#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 09:45:51 CET 2021

@author: edouard.duchesnay@cea.fr

Sources:


Population description

diagnosis
Non-UHR-NC           3
Psychotic            1
Retard_Mental        1
UHR-C               32
UHR-NC              67
UHR-NaN             17
bipolar disorder     2
control             16
schizophrenia       22
dtype: int64
             TIV        age
sex
F    1365.035388  21.915254
M    1535.555070  21.617647
                          TIV        age
diagnosis
Non-UHR-NC        1428.439931  24.333333
Psychotic         1363.883739  30.000000
Retard_Mental     1486.156753  18.000000
UHR-C             1449.962017  20.187500
UHR-NC            1495.255581  22.402985
UHR-NaN           1484.890756  21.470588
bipolar disorder  1620.096916  20.500000
control           1431.030670  21.312500
schizophrenia     1457.618582  21.954545
"""

import os
import os.path
import glob
import click
import numpy as np
import pandas as pd

from nitk import bids

#%% INPUTS:
STUDY = 'start-icaar-eugei'
STUDY_PATH = '/neurospin/psy_sbox/%s' % STUDY
CLINIC_CSV = '/neurospin/psy_sbox/all_studies/phenotype/phenotypes_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START_20201223.tsv'
NII_FILENAMES = glob.glob(
    "/neurospin/psy/%s/derivatives/cat12-12.6_vbm/sub-*/ses-*/anat/mri/mwp1*.nii" % STUDY)

N_SUBJECTS = 161 # Some subject have 2 != participants_id => 2 time points
# participants.irm.unique() ['M0', 'MF'] == Inclusion/final
N_SCANS = 171

assert len(NII_FILENAMES) == N_SCANS

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

    participants = participants[participants.study.isin(['ICAAR_EUGEI_START'])]
    assert participants.shape == (N_SUBJECTS, 46)


    #%% Read mwp1
    ni_icaar_filenames = NII_FILENAMES
    assert len(ni_icaar_filenames) == N_SCANS

    ni_icaar_df = pd.DataFrame([pd.Series(bids.get_keys(filename))
                                for filename in ni_icaar_filenames])

    # Keep only participants with processed T1
    participants = pd.merge(participants, ni_icaar_df, on="participant_id")
    assert participants.shape == (N_SUBJECTS, 48)

    #%% Read Total Imaging volumes
    vol_cols = ["participant_id", 'TIV', 'CSF_Vol', 'GM_Vol', 'WM_Vol']

    tivo_icaar = pd.read_csv(os.path.join(STUDY_PATH,
        'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')[vol_cols]
    tivo_icaar.participant_id = tivo_icaar.participant_id.astype(str)

    # assert tivo_icaar.shape == (171, 6)
    # assert len(ni_icaar_filenames) == 171

    assert tivo_icaar.shape ==  (N_SCANS, 5)

    participants = pd.merge(participants, tivo_icaar, on="participant_id")
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
    participants['sex'] = participants.sex.map({0:"M", 1:"F"})
    print(participants[['diagnosis', 'irm']].groupby(['diagnosis', 'irm']).size())
    print(participants[["sex", "TIV", 'age']].groupby('sex').mean())
    print(participants[["diagnosis", "TIV", 'age']].groupby('diagnosis').mean())

    return participants

if __name__ == "__main__":
    # execute only if run as a script
    participants = make_participants()
