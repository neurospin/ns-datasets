#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:11:34 2019

@author: edouard.duchesnay@cea.fr

Sources:

cp /neurospin/lnao/Pdiff/josselin/laurie-anne/pull_data/bipolar_transcoding_psy_keep.csv /neurospin/psy/bipolar/biobd/code/bipolar_transcoding_remove_duclicates_laurie-anne_20190415.csv
cp /neurospin/lnao/Pdiff/josselin/laurie-anne/pull_data/biodb_clinical.csv /neurospin/psy/bipolar/biobd/code/biodb_clinical_laurie-anne_20190415.csv

Population description

             TIV        age
sex
0.0  1485.537341  39.024742
1.0  1366.405783  38.558532
                          TIV        age
diagnosis
ADHD              1615.635245  20.000000
ADHD, SU          1397.842228  18.000000
EDM               1299.802870  17.000000
MDE, ADHD, panic  1717.615129  16.000000
MDE, PTSD         1397.231961  23.000000
SU, panic         1794.500488  20.000000
bipolar disorder  1401.497266  39.665231
control           1428.363433  39.847359

{'control': 356,
 'bipolar disorder': 306,
 'ADHD, SU': 1,
 nan: 0,
 'EDM': 1,
 'MDE, ADHD, panic': 1,
 'SU, panic': 1, '
 MDE, PTSD': 1,
 'ADHD': 1}
"""

import os
import os.path
import glob
import click
import numpy as np
import pandas as pd

from nitk import bids


#%% INPUTS:

STUDY_PATH = '/neurospin/psy/biobd'
CLINIC_CSV = '/neurospin/psy/all_studies/phenotype/phenotypes_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START_20201223.tsv'
NII_FILENAMES = glob.glob(
    os.path.join(STUDY_PATH, "derivatives/cat12-12.6_vbm/sub-*/ses-V1/anat/mri/mwp1*.nii"))
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
    # rm subjects with missing age or site
    participants = participants[participants.sex.notnull() & participants.age.notnull()]
    assert participants.shape == (2697, 46)

    participants = participants[participants.study == 'BIOBD']
    assert participants.shape == (697, 46)


    #%% Read mwp1
    ni_biobd_filenames = NII_FILENAMES
    assert len(ni_biobd_filenames) == 746

    # def _get_participants_sesssion(filenames):
    #     participant_re = re.compile("sub-([^_/]+)")
    #     session_re = re.compile("ses-([^_/]+)/")
    #     return pd.DataFrame([[participant_re.findall(filename)[0], session_re.findall(filename)[0]] + [filename]
    #             for filename in filenames], columns=["participant_id", "session", "mwp1_filename"])

    # ni_biobd_df = _get_participants_sesssion(ni_biobd_filenames)

    ni_biobd_df = pd.DataFrame([pd.Series(bids.get_keys(filename))
                                for filename in ni_biobd_filenames])

    # Keep only participants with processed T1
    participants = pd.merge(participants, ni_biobd_df, on="participant_id")
    assert participants.shape == (697, 48)

    #%% Read Total Imaging volumes
    # Read schizconnect to remove duplicates
    # tivo_icaar = pd.read_csv(os.path.join(BASE_PATH_icaar, 'stats', 'cat12_tissues_volumes.tsv'), sep='\t')
    #tivo_schizconnect = pd.read_csv(os.path.join(STUDY_PATH_schizconnect, 'stats', 'cat12_tissues_volumes.tsv'), sep='\t')
    vol_cols = ["participant_id", 'TIV', 'CSF_Vol', 'GM_Vol', 'WM_Vol']

    # tivo_bsnip = pd.read_csv(os.path.join(BASE_PATH_bsnip, 'stats', 'cat12_tissues_volumes.tsv'), sep='\t')
    tivo_biobd = pd.read_csv(os.path.join(STUDY_PATH,
        'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')[vol_cols]
    tivo_biobd.participant_id = tivo_biobd.participant_id.astype(str)

    # assert tivo_icaar.shape == (171, 6)
    # assert len(ni_icaar_filenames) == 171

    # assert tivo_schizconnect.shape == (738, 6)
    # assert len(ni_schizconnect_filenames) == 738

    assert tivo_biobd.shape ==  (746, 5)

    participants = pd.merge(participants, tivo_biobd, on="participant_id")
    assert participants.shape == (697, 52)

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
