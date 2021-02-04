#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 09:42:31 CET 2021

@author: edouard.duchesnay@cea.fr

===================
= Descriptive stats
===================
diagnosis
FEP               43
control          420
schizophrenia    275
dtype: int64
========================================
= Basic QC: Reload and check dimension =
========================================
============================
= Basic QC: Age prediction =
============================
rois:	CV R2:0.6080, MAE:5.7307, RMSE:7.2890
vbm:	CV R2:0.6914, MAE:5.0869, RMSE:6.4744
"""

import os
import os.path
import numpy as np
import pandas as pd
import glob
import click

# Neuroimaging
import nibabel
from nitk.image import img_to_array, global_scaling, compute_brain_mask, rm_small_clusters, img_plot_glass_brain
from nitk.bids import get_keys
from nitk.data import fetch_data

# sklearn for QC
import sklearn.linear_model as lm
from sklearn.model_selection import cross_validate
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold

#%% INPUTS:

STUDY = "schizconnect-vip-prague"
STUDY_PATH = '/neurospin/psy_sbox/%s' % STUDY
NII_FILENAMES = glob.glob(
    "/neurospin/psy/%s/derivatives/cat12-12.6_vbm/sub-*/mri/mwp1*.nii" % STUDY)

assert len(NII_FILENAMES) == 738

#%% OUTPUTS:

    #%% OUTPUTS:

OUTPUT_DIR = "/neurospin/tmp/psy_sbox/all_studies/derivatives/arrays"
OUTPUT_FILENAME = "{dirname}/{study}_cat12vbm_{datatype}.{ext}"

def read_data():
    """Read images, ROis, and match with participants

    Returns
    -------
    participants : DataFrame
        participants with session.
    rois : DataFrame
        ROIs.
    imgs_arr : 5D array of shape nb_subject x 1 x X x Y x Z
        Images.
    target_img : TYPE
        Reference images.
    """
    #%% Read Participants

    participants = pd.read_csv(os.path.join(STUDY_PATH, "participants.tsv"), sep='\t')
    participants.participant_id = participants.participant_id.astype(str)
    assert participants.shape[0] == 738
    assert np.all(participants.study.isin(['SCHIZCONNECT-VIP', 'PRAGUE']))
    participants["session"] = "V1" # No session, set to "V1

    #%% Select participants with QC==1

    qc = pd.read_csv(os.path.join(STUDY_PATH,
        'derivatives/cat12-12.6_vbm_qc/qc.tsv'), sep='\t')
    qc.participant_id = qc.participant_id.astype(str)

    participants = participants[participants.participant_id.isin(qc.participant_id[qc["qc"] == 1])]
    assert participants.shape[0] == 738


    #%% Read Images

    assert len(NII_FILENAMES) == 738
    imgs_arr, imgs_df, target_img = img_to_array(NII_FILENAMES)
    imgs_df["session"] = "V1" # No session, set to "V1

    #%% Select images that are in participants

    select_mask_ = imgs_df.participant_id.isin(participants.participant_id)
    imgs_arr = imgs_arr[select_mask_]
    imgs_df = imgs_df[select_mask_]
    imgs_df.reset_index(drop=True, inplace=True)
    del select_mask_
    assert imgs_df.shape[0] == 738
    assert imgs_arr.shape == (738, 1, 121, 145, 121)

    #%% Align participants with images, eventually repplicates for sessions

    # Intersect participants with images, align with images
    # No session here
    assert np.all([col in participants.columns for col in ['participant_id', "session"]]), \
        "participants does not contains some expected columns"

    participants = pd.merge(left=imgs_df[["participant_id", "session"]], right=participants, how='inner',
                    on=["participant_id", "session"])
    participants.reset_index(drop=True, inplace=True)
    assert participants.shape == (738, 52)  # Make sure no particiant is lost

    #%% Align ROIs with images

    rois = pd.read_csv(os.path.join(STUDY_PATH,
        'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
    rois.participant_id = rois.participant_id.astype(str)
    assert rois.shape ==  (738, 291)

    assert np.all([col in rois.columns for col in ['participant_id', 'session']]), \
        "Rois does not contains some expected columns"
    rois = pd.merge(left=imgs_df[["participant_id", "session"]], right=rois, how='inner',
                    on=["participant_id", "session"])
    assert rois.shape == (738, 291)


    # Final QC
    assert participants.shape[0] == rois.shape[0] == imgs_arr.shape[0] == imgs_df.shape[0] == 738
    assert np.all(participants.participant_id == rois.participant_id)
    assert np.all(rois.participant_id == imgs_df.participant_id)

    return participants, rois, imgs_arr, target_img


#%% Read Laurie-Anne QC and save it into derivatives/cat12-12.6_vbm_qc/qc.tsv

def _unused_build_qc_():
    """
    """
    participants = pd.read_csv(os.path.join(STUDY_PATH, "participants.tsv"), sep='\t')
    participants.participant_id = participants.participant_id.astype(str)
    assert participants.shape[0] == 738

    qc = pd.read_csv(os.path.join(STUDY_PATH,
         'derivatives/cat12-12.6_vbm_qc/qc.tsv'), sep= "\t")
    qc.participant_id = qc.participant_id.astype(str)
    qc.corr_mean[ qc.corr_mean.abs() > 1] = qc.corr_mean.median()
    qc["qc"] = 1
    qc.loc[(qc.NCR > 4.5) | (qc.IQR > 4.5), "qc"] = 0

    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.plot(qc.corr_mean, qc.IQR, "o")
    df = qc[['corr_mean', 'NCR', 'ICR', 'IQR']]
    sns.lmplot(x='corr_mean', y='NCR', data=df)
    sns.lmplot(x='corr_mean', y='IQR', data=df)
    sns.lmplot(x='corr_mean', y='NCR', data=df)

    csv_filename = os.path.join(STUDY_PATH,
        'derivatives/cat12-12.6_vbm_qc/qc.tsv')
    print("Save to %s" % csv_filename)
    print("Perform manual look at the data, manually discard (set qc=0) participants in %s" % csv_filename)
    qc.to_csv(csv_filename, sep='\t', index=False)


#%% make_dataset Main function

@click.command()
@click.option('--output', type=str, help='Output dir', default=OUTPUT_DIR)
@click.option('--nogs', is_flag=True, help='No global scaling to the Total Intracranial volume')
@click.option('--dry', is_flag=True, help='Dry Run: no files are written, only check')
def make_dataset(output, nogs, dry):
    """ Make BIOBD cat12BVM dataset, create mpw1, rois, mask and participants file

    Parameters
    ----------
    output : str
        Output dir.
    nogs : TYPE, optional
        No global scaling to the Total Intracranial volume (TIV).
        The default is True.

    Returns
    -------
    None.

        """
    # preprocessing string "gs": global scaling
    preproc_str = ""


    #%% Read data

    print("=============")
    print("= Read data =")
    print("=============")

    participants, rois, imgs_arr, target_img = read_data()


    #%% Global scalling

    if not nogs:
        print("===================")
        print("= Global scalling =")
        print("===================")
        preproc_str += "-gs"

        imgs_arr = global_scaling(imgs_arr, axis0_values=rois['TIV'].values, target=1500)
        gscaling = 1500 / rois['TIV']
        rois.loc[:, 'TIV':] = rois.loc[:, 'TIV':].multiply(gscaling, axis="index")

    print("==============")
    print("= Fetch mask =")
    print("==============")

    base_url='ftp://ftp.cea.fr/pub/unati/ni_ressources/masks/'
    fetch_data(files=["mni_cerebrum-gm-mask_1.5mm.nii.gz", "mni_brain-gm-mask_1.5mm.nii.gz"], dst=output, base_url=base_url, verbose=1)
    mask_img = nibabel.load(os.path.join(output, "mni_cerebrum-gm-mask_1.5mm.nii.gz"))
    assert np.all(mask_img.affine == target_img.affine), "Data shape do not match cat12VBM"

    participants_filename = OUTPUT_FILENAME.format(dirname=output, study=STUDY, datatype="participants", ext="csv")
    rois_filename = OUTPUT_FILENAME.format(dirname=output, study=STUDY, datatype="rois%s" % preproc_str, ext="csv")
    vbm_filename = OUTPUT_FILENAME.format(dirname=output, study=STUDY, datatype="mwp1%s" % preproc_str, ext="npy")

    if not dry:
        print("======================")
        print("= Save data to: %s" % output)
        print("======================")

        participants.to_csv(participants_filename, index=False)
        rois.to_csv(rois_filename, index=False)
        #target_img.save()  # No need to save the reference image since it is identical to the mask
        np.save(vbm_filename, imgs_arr)

        print(participants_filename, rois_filename, vbm_filename)
    else:
        print("= Dry run do not save to %s" % participants_filename)

    print("===================")
    print("= Descriptive stats")
    print("===================")

    print(participants[["diagnosis"]].groupby('diagnosis').size())

    #%% QC1: Basic ML brain age

    print("========================================")
    print("= Basic QC: Reload and check dimension =")
    print("========================================")

    participants = pd.read_csv(participants_filename)
    rois = pd.read_csv(rois_filename)
    imgs_arr = np.load(vbm_filename)
    mask_img = nibabel.load(os.path.join(output, "mni_cerebrum-gm-mask_1.5mm.nii.gz"))

    assert participants.shape == (738, 52)
    assert rois.shape == (738, 291)
    assert imgs_arr.shape == (738, 1, 121, 145, 121)

    print("============================")
    print("= Basic QC: Age prediction =")
    print("Expected values:")
    print("rois:	CV R2:0.6049, MAE:5.7385, RMSE:7.3214")
    print("vbm:	CV R2:0.6959, MAE:5.0062, RMSE:6.4248")
    print("============================")

    mask_arr = mask_img.get_fdata() != 0
    data = dict(rois=rois.loc[:, 'l3thVen_GM_Vol':].values,
                vbm=imgs_arr.squeeze()[:, mask_arr])
    y = participants["age"].values
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    lr = make_pipeline(preprocessing.StandardScaler(), lm.Ridge(alpha=10))

    for name, X, in data.items():
        cv_res = cross_validate(estimator=lr, X=X, y=y, cv=cv,
                                n_jobs=5,
                                scoring=['r2', 'neg_mean_absolute_error',
                                         'neg_mean_squared_error'])
        r2 = cv_res['test_r2'].mean()
        rmse = np.sqrt(np.mean(-cv_res['test_neg_mean_squared_error']))
        mae = np.mean(-cv_res['test_neg_mean_absolute_error'])
        print("%s:\tCV R2:%.4f, MAE:%.4f, RMSE:%.4f" % (name, r2, mae, rmse))


if __name__ == '__main__':
    make_dataset()