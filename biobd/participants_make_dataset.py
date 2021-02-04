#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:11:34 2019

@author: edouard.duchesnay@cea.fr

Sources:

cp /neurospin/lnao/Pdiff/josselin/laurie-anne/pull_data/bipolar_transcoding_psy_keep.csv /neurospin/psy/bipolar/biobd/code/bipolar_transcoding_remove_duclicates_laurie-anne_20190415.csv
cp /neurospin/lnao/Pdiff/josselin/laurie-anne/pull_data/biodb_clinical.csv /neurospin/psy/bipolar/biobd/code/biodb_clinical_laurie-anne_20190415.csv

Population description:

diagnosis
ADHD                  1
ADHD, SU              1
EDM                   1
MDE, ADHD, panic      1
MDE, PTSD             1
SU, panic             1
bipolar disorder    307
control             356
dtype: int64
             TIV        age
sex
0.0  1480.877380  39.969044
1.0  1365.625379  39.252639
                          TIV        age
diagnosis
ADHD              1615.635245  20.000000
ADHD, SU          1397.842228  18.000000
EDM               1299.802870  17.000000
MDE, ADHD, panic  1717.615129  16.000000
MDE, PTSD         1397.231961  23.000000
SU, panic         1794.500488  20.000000
bipolar disorder  1401.771065  39.656550
control           1428.363433  39.847359
"""

import os
import os.path
import glob
import click
import numpy as np
import pandas as pd

from nitk import bids


#%% INPUTS:
STUDY = 'biobd'
STUDY_PATH = '/neurospin/psy_sbox/%s' % STUDY
CLINIC_CSV = '/neurospin/psy/all_studies/phenotype/phenotypes_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START_20201223.tsv'
NII_FILENAMES = glob.glob(
    "/neurospin/psy/%s/derivatives/cat12-12.6_vbm/sub-*/ses-V1/anat/mri/mwp1*.nii" % STUDY)

assert len(NII_FILENAMES) == 746

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

    participants = participants[participants.study == 'BIOBD']
    assert participants.shape == (669, 46)

    #%% Read mwp1
    ni_biobd_filenames = NII_FILENAMES
    assert len(ni_biobd_filenames) == 746

    ni_biobd_df = pd.DataFrame([pd.Series(bids.get_keys(filename))
                                for filename in ni_biobd_filenames])

    # Keep only participants with processed T1
    participants = pd.merge(participants, ni_biobd_df, on="participant_id")
    assert participants.shape == (669, 48)

    #%% Read Total Imaging volumes
    vol_cols = ["participant_id", 'TIV', 'CSF_Vol', 'GM_Vol', 'WM_Vol']

    tivo_biobd = pd.read_csv(os.path.join(STUDY_PATH,
        'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')[vol_cols]
    tivo_biobd.participant_id = tivo_biobd.participant_id.astype(str)

    # assert tivo_icaar.shape == (171, 6)
    # assert len(ni_icaar_filenames) == 171

    # assert tivo_schizconnect.shape == (738, 6)
    # assert len(ni_schizconnect_filenames) == 738

    assert tivo_biobd.shape ==  (746, 5)

    participants = pd.merge(participants, tivo_biobd, on="participant_id")
    assert participants.shape == (669, 52)

    print(participants.shape)

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
    print(participants[["diagnosis"]].groupby('diagnosis').size())
    print(participants[["sex", "TIV", 'age']].groupby('sex').mean())
    print(participants[["diagnosis", "TIV", 'age']].groupby('diagnosis').mean())

    return participants

if __name__ == "__main__":
    # execute only if run as a script
    participants = make_participants()
