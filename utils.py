#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: benoit.dufumier


"""

# TODO: Libraries pylearn-mulm, brainomics needed for these functions. Do we hard copy them here ?
import os, sys
import numpy as np
import pandas as pd
import brainomics.image_preprocessing as preproc
from brainomics.image_statistics import univ_stats, plot_univ_stats, residualize, ml_predictions
import matplotlib
matplotlib.use('Agg')
import glob

def OUTPUT_CAT12(dataset, output_path, modality='t1mri', mri_preproc='mwp1', scaling=None, harmo=None, type=None, ext=None):
    # scaling: global scaling? in "raw", "gs"
    # harmo (harmonization): in [raw, ctrsite, ressite, adjsite]
    # type data64, or data32
    return os.path.join(output_path, dataset + "_" + modality+ "_" + mri_preproc +
                 ("" if scaling is None else "_" + scaling) +
                 ("" if harmo is None else "-" + harmo) +
                 ("" if type is None else "_" + type) + "." + ext)


def OUTPUT_QUASI_RAW(dataset, output_path, modality='t1mri', mri_preproc='quasi_raw', type=None, ext=None):
    # type data64, or data32
    return os.path.join(output_path, dataset + "_" + modality+ "_" + mri_preproc +
                 ("" if type is None else "_" + type) + "." + ext)


def quasi_raw_nii2npy(nii_path, phenotype_path, dataset_name, output_path, qc=None, sep='\t', id_type=str,
            check = dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5))):
    ########################################################################################################################


    phenotype = pd.read_csv(phenotype_path, sep=sep)
    qc = pd.read_csv(qc, sep=sep) if qc is not None else None

    if 'TIV' in phenotype:
        phenotype.rename(columns={'TIV': 'tiv'}, inplace=True)

    keys_required = ['participant_id', 'age', 'sex', 'tiv', 'diagnosis']

    assert set(keys_required) <= set(phenotype.columns), \
        "Missing keys in {} that are required to compute the npy array: {}".format(phenotype_path,
                                                                                   set(keys_required)-set(phenotype.columns))

    ## TODO: change this condition according to session and run in phenotype.csv
    #assert len(set(phenotype.participant_id)) == len(phenotype), "Unexpected number of participant_id"


    null_or_nan_mask = [False for _ in range(len(phenotype))]
    for key in keys_required:
        null_or_nan_mask |= getattr(phenotype, key).isnull() | getattr(phenotype, key).isna()
    if null_or_nan_mask.sum() > 0:
        print('Warning: {} participant_id will not be considered because of missing required values:\n{}'. \
              format(null_or_nan_mask.sum(), list(phenotype[null_or_nan_mask].participant_id.values)))

    participants_df = phenotype[~null_or_nan_mask]

    ########################################################################################################################
    #  Neuroimaging niftii and TIV
    #  mwp1 files
      #  excpected image dimensions
    NI_filenames = glob.glob(nii_path)
    ########################################################################################################################
    #  Load images, intersect with pop and do preprocessing and dump 5d npy
    print("###########################################################################################################")
    print("#", dataset_name)

    print("# 1) Read images")
    scaling, harmo = 'raw', 'raw'
    print("## Load images")
    NI_arr, NI_participants_df, ref_img = preproc.load_images(NI_filenames,check=check)
    print('--> {} img loaded'.format(len(NI_participants_df)))
    print("## Merge nii's participant_id with participants.csv")
    NI_arr, NI_participants_df = preproc.merge_ni_df(NI_arr, NI_participants_df, participants_df,
                                                         qc=qc, id_type=id_type)
    print('--> Remaining samples: {} / {}'.format(len(NI_participants_df), len(participants_df)))

    print("## Save the new participants.csv")
    NI_participants_df.to_csv(OUTPUT_QUASI_RAW(dataset_name, output_path, type="participants", ext="csv"),
                              index=False)
    print("## Save the raw npy file (with shape {})".format(NI_arr.shape))
    np.save(OUTPUT_QUASI_RAW(dataset_name, output_path, type="data64", ext="npy"), NI_arr)

    ######################################################################################################################
    # Deallocate the memory
    del NI_arr

def cat12_nii2npy(nii_path, phenotype_path, dataset_name, output_path, qc=None, sep='\t', id_type=str,
            check = dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5))):
    ########################################################################################################################
    # Read phenotypes

    phenotype = pd.read_csv(phenotype_path, sep=sep)
    qc = pd.read_csv(qc, sep=sep) if qc is not None else None

    if 'TIV' in phenotype:
        phenotype.rename(columns={'TIV': 'tiv'}, inplace=True)

    keys_required = ['participant_id', 'age', 'sex', 'tiv', 'diagnosis']

    assert set(keys_required) <= set(phenotype.columns), \
        "Missing keys in {} that are required to compute the npy array: {}".format(phenotype_path,
                                                                                   set(keys_required)-set(phenotype.columns))

    ## TODO: change this condition according to session and run in phenotype.csv
    #assert len(set(phenotype.participant_id)) == len(phenotype), "Unexpected number of participant_id"


    null_or_nan_mask = [False for _ in range(len(phenotype))]
    for key in keys_required:
        null_or_nan_mask |= getattr(phenotype, key).isnull() | getattr(phenotype, key).isna()
    if null_or_nan_mask.sum() > 0:
        print('Warning: {} participant_id will not be considered because of missing required values:\n{}'. \
              format(null_or_nan_mask.sum(), list(phenotype[null_or_nan_mask].participant_id.values)))

    participants_df = phenotype[~null_or_nan_mask]

    ########################################################################################################################
    #  Neuroimaging niftii and TIV
    #  mwp1 files
      #  excpected image dimensions
    NI_filenames = glob.glob(nii_path)
    ########################################################################################################################
    #  Load images, intersect with pop and do preprocessing and dump 5d npy
    print("###########################################################################################################")
    print("#", dataset_name)

    print("# 1) Read images")
    scaling, harmo = 'raw', 'raw'
    print("## Load images")
    NI_arr, NI_participants_df, ref_img = preproc.load_images(NI_filenames,check=check)
    print('--> {} img loaded'.format(len(NI_participants_df)))
    print("## Merge nii's participant_id with participants.csv")
    NI_arr, NI_participants_df = preproc.merge_ni_df(NI_arr, NI_participants_df, participants_df,
                                                         qc=qc, id_type=id_type)
    print('--> Remaining samples: {} / {}'.format(len(NI_participants_df), len(participants_df)))

    print("## Save the new participants.csv")
    NI_participants_df.to_csv(OUTPUT_CAT12(dataset_name, output_path, scaling=None, harmo=None, type="participants", ext="csv"),
                              index=False)
    print("## Save the raw npy file (with shape {})".format(NI_arr.shape))
    np.save(OUTPUT_CAT12(dataset_name, output_path, scaling=scaling, harmo=harmo, type="data64", ext="npy"), NI_arr)
    NI_arr = np.load(OUTPUT_CAT12(dataset_name, output_path, scaling=scaling, harmo=harmo, type="data64", ext="npy"), mmap_mode='r')

    print("## Compute brain mask")
    mask_img = preproc.compute_brain_mask(NI_arr, ref_img, mask_thres_mean=0.1, mask_thres_std=1e-6,
                                          clust_size_thres=10,
                                          verbose=1)
    mask_arr = mask_img.get_data() > 0
    print("## Save the mask")
    mask_img.to_filename(OUTPUT_CAT12(dataset_name, output_path, scaling=None, harmo=None, type="mask", ext="nii.gz"))

    ########################################################################################################################
    print("# 2) Raw data")
    # Univariate stats

    # design matrix: Set missing diagnosis to 'unknown' to avoid missing data(do it once)
    dmat_df = NI_participants_df[['age', 'sex', 'tiv']]
    assert np.all(dmat_df.isnull().sum() == 0)
    print("## Do univariate stats on age, sex and TIV")
    univmods, univstats = univ_stats(NI_arr.squeeze()[:, mask_arr], formula="age + sex + tiv", data=dmat_df)

    # %time univmods, univstats = univ_stats(NI_arr.squeeze()[:, mask_arr], formula="age + sex + diagnosis + tiv + site", data=dmat_df)
    pdf_filename = OUTPUT_CAT12(dataset_name, output_path, scaling=scaling, harmo=harmo, type="univstats", ext="pdf")
    plot_univ_stats(univstats, mask_img, data=dmat_df, grand_mean=NI_arr.squeeze()[:, mask_arr].mean(axis=1),
                    pdf_filename=pdf_filename, thres_nlpval=3,
                    skip_intercept=True)

    ########################################################################################################################
    print("# 3) Global scaling")
    scaling, harmo = 'gs', 'raw'

    print("## Apply global scaling")
    NI_arr = preproc.global_scaling(NI_arr, axis0_values=np.array(NI_participants_df.tiv), target=1500)
    # Save
    print("## Save the new .npy array")
    np.save(OUTPUT_CAT12(dataset_name, output_path, scaling=scaling, harmo=harmo, type="data64", ext="npy"), NI_arr)
    NI_arr = np.load(OUTPUT_CAT12(dataset_name, output_path, scaling=scaling, harmo=harmo, type="data64", ext="npy"), mmap_mode='r')

    # Univariate stats
    print("## Recompute univariate stats on age, sex and TIV")
    univmods, univstats = univ_stats(NI_arr.squeeze()[:, mask_arr], formula="age + sex + tiv", data=dmat_df)
    pdf_filename = OUTPUT_CAT12(dataset_name, output_path, scaling=scaling, harmo=harmo, type="univstats", ext="pdf")
    plot_univ_stats(univstats, mask_img, data=dmat_df, grand_mean=NI_arr.squeeze()[:, mask_arr].mean(axis=1),
                    pdf_filename=pdf_filename, thres_nlpval=3,
                    skip_intercept=True)
    # Deallocate the memory
    del NI_arr




