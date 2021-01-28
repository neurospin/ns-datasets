#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 15:45:22 CET 2021

@author: edouard.duchesnay@cea.fr

Population description

             TIV        age
sex
0.0  1458.560863  35.909910
1.0  1283.825687  40.947635
                                                            TIV        age
diagnosis
control                                             1383.028836  38.643216
psychotic bipolar disorder                          1351.313738  37.401709
relative of proband with psychotic bipolar diso...  1350.709520  40.117647
relative of proband with schizoaffective disorder   1336.081444  41.504065
relative of proband with schizophrenia              1331.678799  43.622857
schizoaffective disorder                            1327.429595  36.252252
schizophrenia                                       1400.200306  34.281250
{'control': 199,
 'schizophrenia': 192,
 'relative of proband with schizoaffective disorder': 123,
 'schizoaffective disorder': 111,
 'psychotic bipolar disorder': 117,
 'relative of proband with schizophrenia': 175,
 'relative of proband with psychotic bipolar disorder': 119}
"""

import os
import os.path
import glob
import re
import click
import numpy as np
import pandas as pd

from nitk import bids


#%% INPUTS:

STUDY_PATH = '/neurospin/psy/bsnip1'
CLINIC_CSV = '/neurospin/psy/all_studies/phenotype/phenotypes_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START_20201223.tsv'
NII_FILENAMES = glob.glob(
    os.path.join(STUDY_PATH, "derivatives/cat12-12.6_vbm/sub-*/ses-V1/anat/mri/mwp1*.nii"))
assert len(NII_FILENAMES) == 1042

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
    # rm subjects with missing age or site
    participants = participants[participants.sex.notnull() &
                                participants.age.notnull() &
                                participants.site.notnull() &
                                participants.diagnosis.notnull()]
    assert participants.shape == (2663, 46)

    participants = participants[participants.study == 'BSNIP']
    assert participants.shape == (1094, 46)


    #%% Read mwp1
    ni_bsnip_filenames = NII_FILENAMES
    assert len(ni_bsnip_filenames) == 1042

    ni_bsnip_df = pd.DataFrame([pd.Series(bids.get_keys(filename))
                                for filename in ni_bsnip_filenames])

    # Keep only participants with processed T1
    participants = pd.merge(participants, ni_bsnip_df, on="participant_id")
    assert participants.shape == (1042, 48)

    #%% Read Total Imaging volumes
    vol_cols = ["participant_id", 'TIV', 'CSF_Vol', 'GM_Vol', 'WM_Vol']

    tivo_bsnip = pd.read_csv(os.path.join(STUDY_PATH,
        'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')[vol_cols]
    tivo_bsnip.participant_id = tivo_bsnip.participant_id.astype(str)

    # assert tivo_icaar.shape == (171, 6)
    # assert len(ni_icaar_filenames) == 171

    # assert tivo_schizconnect.shape == (738, 6)
    # assert len(ni_schizconnect_filenames) == 738

    # assert tivo_bsnip.shape == (1042, 6)
    # assert len(ni_bsnip_filenames) == 1042

    assert tivo_bsnip.shape ==  (1036, 5)

    participants = pd.merge(participants, tivo_bsnip, on="participant_id")
    assert participants.shape == (1036, 52)

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
