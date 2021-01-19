#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 11:07:29 2020

@author: edouard.duchesnay@cea.fr
"""

import os
import numpy as np
import pandas as pd
import glob
#import nibabel as nib  # import generate a FutureWarning
#import matplotlib.pyplot as plt
import os.path
#import scipy.io
#import scipy.linalg
#import shutil
#import xml.etree.ElementTree as ET
import subprocess
import re
import glob
from nitk.image import img_to_array, global_scaling, compute_brain_mask, rm_small_clusters, img_plot_glass_brain


STUDY_PATH = '/neurospin/psy_sbox/bipolar-biobd'
NII_FILENAMES = glob.glob("/neurospin/psy/biobd/derivatives/cat12/vbm/sub-*/ses-V1/anat/mri/mwp1*.nii")

# STUDY_PATH_schizconnect = '/neurospin/psy/schizconnect-vip-prague'
# CLINIC_CSV = os.path.join(STUDY_PATH, 'phenotype/build_dataset/phenotypes_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START.tsv')

###############################################################################
#%% Read Participants

participants = pd.read_csv(os.path.join(STUDY_PATH, "participants.tsv"), sep='\t')
participants.participant_id = participants.participant_id.astype(str)
assert np.all(participants.study.isin(["BIOBD"]))
assert participants.shape[0] == 697

###############################################################################
#%% Images

imgs_arr, pop_ni, target_img = img_to_array(NII_FILENAMES)
len(NII_FILENAMES) == 746

###############################################################################
#%% Rois

rois = pd.read_csv(os.path.join(STUDY_PATH,
    'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
rois.participant_id = rois.participant_id.astype(str)
assert rois.shape ==  (746, 290)



OLDIES
#%% Select participants for VBM


# Ask Laurie Anne what is in this file
"""
laurie_anne = pd.read_csv(os.path.join(BASE_PATH_biobd, "phenotype", "build_dataset",
                                       "laurie_anne_merging_201904.tsv"), sep='\t')
laurie_anne.new_ID = laurie_anne.new_ID.astype(str)
laurie_anne.insert(0, "participant_id", laurie_anne.new_ID)
laurie_anne = laurie_anne[laurie_anne.Keep == 1]
"""

#%% Read QC
qc = pd.read_csv(os.path.join(STUDY_PATH,
    'derivatives/cat12-12.6_vbm_qc/qc.tsv'), sep='\t')
qc.participant_id = qc.participant_id.astype(str)

assert qc.shape[0] == 746


#%% Apply Laurie Anne selection into QC if needed
if 'qc_laurie-anne' not in qc.columns:
    laurie = pd.read_csv(os.path.join(STUDY_PATH,
        'derivatives/cat12-12.6_vbm_qc-laurie-anne/norm_dataset_cat12_bsnip_biobd.tsv'), sep='\t')
    laurie.participant_id = laurie.participant_id.astype(str)
    assert laurie.shape == (993, 183)
    s = laurie.participant_id[0]
    laurie["is_biobd"] = [not s.startswith('INV') for s in laurie.participant_id]
    laurie = laurie[laurie["is_biobd"]]

    assert laurie.shape[0] == 677

    # 34 subject Removed
    assert len(set(participants.participant_id) - set(laurie.participant_id)) == 34

    # Apply fast QC and Laurie selection
    #fast_qc = fast_qc.drop("qc_laurie-anne", axis=1)

    qc["qc_laurie-anne"] = 0
    qc.loc[qc.participant_id.isin(laurie.participant_id), "qc_laurie-anne"] = 1
    # Make Laurie-Anne QC the new QC
    qc["qc"] = qc["qc_laurie-anne"]

    qc.to_csv(os.path.join(STUDY_PATH,
        'derivatives/cat12-12.6_vbm_qc/qc.tsv'), sep='\t', index=False)


    # OK so far




#fast_qc.to_csv(os.path.join(BASE_PATH_biobd,
#    'derivatives/cat12-12.6_vbm_qc/qc.tsv'), sep='\t', index=False)
"""
{'106155619059',
 '122453692470',
 '138432854246',
 '149945412502',
 '175092425987',
 '179056268515',
 '200348608780',
 '206749871806',
 '213889059273',
 '237316076994',
 '286261781409',
 '307816755268',
 '322297931142',
 '325766870880',
 '328878196078',
 '334345139854',
 '352160641135',
 '352572019743',
 '372933076631',
 '383341727580', out
 '391617528618',
 '396494923948',
 '400222907331',
 '406249203634',
 '434225074023',
 '442595775212',
 '455549998730',
 '461903006973',
 '467195136336',
 '468730767332',
 '478171713584',
 '491217343760',
 '501127836295',
 '542241441476',
 '564252676034',
 '570221845965',
 '589916690768',
 '613965897134',
 '617030174208',
 '645604123256',
 '651599076115', out
 '661005540862',
 '682108027037',
 '705483781994',
 '706415700286',
 '706928832977',
 '722188139700',
 '739487676161',
 '739657365501',
 '740649847210',
 '797727246006',
 '799853514370',
 '807115709508',
 '809878713473',
 '821637832591',
 '821944316944',
 '850820275021', out
 '895940602378',
 '918563591727',
 '930023532220',
 '938038137548',
 '939467895180',
 '939686012648',
 '970408525689',
 '972652377496',
 '978881310059',
 '981518352800',
 '995197582752',
 '999231381382'}
"""