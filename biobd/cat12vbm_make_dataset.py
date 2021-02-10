#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 11:07:29 2020

@author: edouard.duchesnay@cea.fr

===================
= Descriptive stats
===================
diagnosis
bipolar disorder    307
control             356

============================
= Basic QC: Age prediction =
============================
rois:	CV R2:0.6132, MAE:6.0520, RMSE:7.8319
vbm:	CV R2:0.7305, MAE:5.1636, RMSE:6.5485
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

STUDY = "biobd"
STUDY_PATH = '/neurospin/psy_sbox/%s' % STUDY
NII_FILENAMES = glob.glob(
    "/neurospin/psy/%s/derivatives/cat12-12.6_vbm/sub-*/ses-V1/anat/mri/mwp1*.nii" % STUDY)
assert len(NII_FILENAMES) == 746

#%% OUTPUTS:

OUTPUT_DIR = "/neurospin/tmp/psy_sbox/all_studies/derivatives/arrays"
OUTPUT_FILENAME = "{dirname}/{study}_cat12vbm_{datatype}.{ext}"

N_SUBJECTS = 663

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
    assert participants.shape[0] == 669
    assert np.all(participants.study.isin(["BIOBD"]))

    #%% Keep only BD and CTL
    participants = participants[participants.diagnosis.isin(['control', 'bipolar disorder'])]
    assert participants.shape[0] == N_SUBJECTS

    #%% Select participants with QC==1
    qc = pd.read_csv(os.path.join(STUDY_PATH,
        'derivatives/cat12-12.6_vbm_qc/qc.tsv'), sep='\t')
    qc.participant_id = qc.participant_id.astype(str)


    excluded = participants.loc[participants.participant_id.isin(qc.participant_id[qc["qc"] != 1]),
                            ["participant_id", "diagnosis"]]
    assert excluded.shape[0] == 0

    participants = participants[participants.participant_id.isin(qc.participant_id[qc["qc"] == 1])]
    assert participants.shape[0] == N_SUBJECTS

# Useless but add it to be rigourous

    #%% Read Images

    assert len(NII_FILENAMES) == 746
    imgs_arr, imgs_df, target_img = img_to_array(NII_FILENAMES)

    #%% Select images that are in participants

    select_mask_ = imgs_df.participant_id.isin(participants.participant_id)
    imgs_arr = imgs_arr[select_mask_]
    imgs_df = imgs_df[select_mask_]
    imgs_df.reset_index(drop=True, inplace=True)
    del select_mask_
    assert imgs_df.shape[0] == N_SUBJECTS
    assert imgs_arr.shape == (N_SUBJECTS, 1, 121, 145, 121)

    #%% Align participants with images, eventually repplicates for sessions

    # Intersect participants with images, align with images
    assert np.all([col in participants.columns for col in ['participant_id', 'session']]), \
        "participants does not contains some expected columns"

    participants = pd.merge(left=imgs_df[["participant_id", "session"]], right=participants, how='inner',
                    on=["participant_id", "session"])
    participants.reset_index(drop=True, inplace=True)
    assert participants.shape == (N_SUBJECTS, 52)  # Make sure no particiant is lost

    #%% Align ROIs with images

    rois = pd.read_csv(os.path.join(STUDY_PATH,
        'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
    rois.participant_id = rois.participant_id.astype(str)
    assert rois.shape ==  (746, 290)

    assert np.all([col in rois.columns for col in ['participant_id', 'session']]), \
        "Rois does not contains some expected columns"
    rois = pd.merge(left=imgs_df[["participant_id", "session"]], right=rois, how='inner',
                    on=["participant_id", "session"])
    assert rois.shape == (N_SUBJECTS, 290)


    # Final QC
    assert participants.shape[0] == rois.shape[0] == imgs_arr.shape[0] == imgs_df.shape[0] == N_SUBJECTS
    assert np.all(participants.participant_id == rois.participant_id)
    assert np.all(rois.participant_id == imgs_df.participant_id)

    return participants, rois, imgs_arr, target_img


#%% Read Laurie-Anne QC and save it into derivatives/cat12-12.6_vbm_qc/qc.tsv

def build_qc_from_laurie_anne_qc():
    """Read Laurie-Anne QC and save it into derivatives/cat12-12.6_vbm_qc/qc.tsv
    See Laurie-Anne README:
    phenotype/2019_laurie_anne/readme_constitution_dataset_biobd_bsnip.odt
    """
    participants = pd.read_csv(os.path.join(STUDY_PATH, "participants.tsv"), sep='\t')
    participants.participant_id = participants.participant_id.astype(str)
    assert participants.shape[0] == 669

    # We should use the cat12_qc_laurie-anne_20190627.csv but for some reason
    # Laurie-anne used another file see bellow
    laurie = pd.read_csv(os.path.join(STUDY_PATH,
        'derivatives/cat12-12.6_vbm_qc-laurie-anne/cat12_qc_laurie-anne_20190627.csv'))
    laurie.participant_id = [s.replace("sub-", "") for s in laurie.participant_id.astype(str)]
    laurie.participant_id = laurie.participant_id.astype(str)

    laurie = laurie[laurie.qc_cat12 == 1]
    #laurie = laurie[laurie.participant_id.isin(participants.participant_id)]
    assert laurie.shape[0] == 728

    # Note that all participants are OK in Laurie-Anne QC
    assert participants.participant_id.isin(laurie.participant_id).sum() == participants.shape[0]

    # Check with selected by Laurie:
    laurie2 = pd.read_csv(os.path.join(STUDY_PATH,
        'derivatives/cat12-12.6_vbm_qc-laurie-anne/norm_dataset_cat12_bsnip_biobd.tsv'), sep='\t')
    laurie2.participant_id = laurie2.participant_id.astype(str)
    assert laurie2.shape == (993, 183)
    laurie2["is_biobd"] = [not s.startswith('INV') for s in laurie2.participant_id]
    laurie2 = laurie2[laurie2["is_biobd"]]
    assert laurie2.shape[0] == 677

    # Defferences are explaind by
    # - 14 vip_duplicated_in_biobd
    # - 6 with non BD DX
    # So participants selected by Laurie-Anne match participants with Laurie ANNE QC
    diff = pd.DataFrame(dict(participant_id=participants.participant_id.append(laurie.participant_id.append(laurie2.participant_id)).unique(),
                        participants=0,
                        cat12_qc_laurie_anne_20190627=0, norm_dataset_cat12_bsnip_biobd=0))
    diff.loc[diff.participant_id.isin(participants.participant_id), "participants"] = 1
    diff.loc[diff.participant_id.isin(laurie.participant_id), "cat12_qc_laurie_anne_20190627"] = 1
    diff.loc[diff.participant_id.isin(laurie2.participant_id), "norm_dataset_cat12_bsnip_biobd"] = 1

    diff["select_cat12_qc_laurie_anne_20190627"] = ((diff.participants == 1) & (diff.cat12_qc_laurie_anne_20190627)).astype(int)
    vip_duplicated_in_biobd = ['341879365063', '156634941156', '611954003219', '999412570656', '435432648506', '186334059458', '870810930661', '153138320244', '726278928908', '611553851411', '942465208526', '148210353882', '419555247213', '544435731463']
    diff["vip_duplicated_in_biobd"] = 0
    diff.vip_duplicated_in_biobd[diff.participant_id.isin(vip_duplicated_in_biobd)] = 1

    diff = pd.merge(diff, participants[["participant_id", "diagnosis", "age", "sex", "site"]],
                    on="participant_id", how="outer")

    # Keep only differences not duplicated in vip
    diff = diff[diff.select_cat12_qc_laurie_anne_20190627 != diff.norm_dataset_cat12_bsnip_biobd]
    diff = diff[diff.vip_duplicated_in_biobd != 1]

    # 6 subjects are differents non are 'control', 'bipolar disorder'
    assert diff.shape[0] == 6 and np.all(~diff.diagnosis.isin(['control', 'bipolar disorder']))
    # diff.to_csv(os.path.join(STUDY_PATH,
    #        'derivatives/cat12-12.6_vbm_qc-laurie-anne/diff__cat12_qc_laurie-anne_20190627__norm_dataset_cat12_bsnip_biobd.tsv'), index=False, sep='\t')
    # 25    722188139700             1  ...  1.0  geneve
    # 113   706928832977             1  ...  1.0  geneve
    # 297   589916690768             1  ...  1.0  geneve
    # 530   467195136336             1  ...  0.0  geneve
    # 544   739657365501             1  ...  1.0  geneve
    # 605   237316076994             1  ...  0.0  geneve

    # Save QC = participants + apply Laurie-QC (Please not that 6 subjects non
    # control nor bipolar disorder haven't been chached)
    qc = pd.read_csv(os.path.join(STUDY_PATH,
         'derivatives/cat12-12.6_vbm_qc/qc.tsv'), sep= "\t")
    qc.participant_id = qc.participant_id.astype(str)
    qc = qc.loc[:, :'corr_mean']

    qc["qc"] = 0
    qc.loc[qc.participant_id.isin(laurie.participant_id), "qc"] = 1

    # Recheck that participants are OK in Laurie-Anne QC
    assert participants.participant_id.isin(qc.participant_id[qc["qc"] == 1]).sum() == participants.shape[0]
    assert len(qc.participant_id[qc["qc"] == 1]) == 728

    qc.to_csv(os.path.join(STUDY_PATH,
        'derivatives/cat12-12.6_vbm_qc/qc.tsv'), sep='\t', index=False)


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
    print("Arguments")
    print("output", output)
    print("nogs", nogs)

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
    fetch_data(files=["mni_cerebrum-gm-mask_1.5mm.nii.gz", "mni_brain-gm-mask_1.5mm.nii.gz"],
               dst=output, base_url=base_url, verbose=1)
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

    assert participants.shape == (N_SUBJECTS, 52)
    assert rois.shape == (N_SUBJECTS, 290)
    assert imgs_arr.shape == (N_SUBJECTS, 1, 121, 145, 121)

    print("============================")
    print("= Basic QC: Age prediction =")
    print("Expected values:")
    print("rois:	CV R2:0.6132, MAE:6.0520, RMSE:7.8319")
    print("vbm:	CV R2:0.7305, MAE:5.1636, RMSE:6.5485")
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