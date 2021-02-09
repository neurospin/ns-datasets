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

"""
Format:

<study>_<software>_<output>[-<options>][_resolution]

study := cat12vbm | quasiraw
output := mwp1 | rois
options := gs: global scaling
resolution := 1.5mm | 1mm

Examples:

bsnip1_cat12vbm_mwp1-gs_1.5mm.npy
bsnip1_cat12vbm_rois-gs.csv
bsnip1_cat12vbm_participants.csv
"""

# TODO Julie/Benoit: modify OUTPUT_CAT12 and OUTPUT_QUASI_RAW to match format
# TODO Julie/Benoit: split int participants.tsv, rois and vbm
# TODO Edouard Add 10 line of age prediction

def OUTPUT_CAT12(dataset, output_path, modality='cat12vbm', mri_preproc='mwp1', scaling=None, harmo=None, type=None, ext=None):
    # scaling: global scaling? in "raw", "gs"
    # harmo (harmonization): in [raw, ctrsite, ressite, adjsite]
    # type data64, or data32
    return os.path.join(output_path, dataset + "_" + modality+ "_" + mri_preproc +
                 ("" if scaling is None else "_" + scaling) +
                 ("" if harmo is None else "-" + harmo) +
                 ("" if type is None else "_" + type) + "." + ext)


def OUTPUT_QUASI_RAW(dataset, output_path, modality='cat12vbm', mri_preproc='quasi_raw', type=None, ext=None):
    # type data64, or data32
    return os.path.join(output_path, dataset + "_" + modality+ "_" + mri_preproc +
                 ("" if type is None else "_" + type) + "." + ext)

def merge_ni_df(NI_arr, NI_participants_df, participants_df, qc=None, participant_id="participant_id", id_type=str,
                merge_ni_path=True):
    """
    Select participants of NI_arr and NI_participants_df participants that are also in participants_df

    Parameters
    ----------
    NI_arr:  ndarray, of shape (n_subjects, 1, image_shape).
    NI_participants_df: DataFrame, with at leas 2 columns: participant_id, "ni_path"
    participants_df: DataFrame, with 2 at least 1 columns participant_id
    qc: DataFrame, with at least 1 column participant_id
    participant_id: column that identify participant_id
    id_type: the type of participant_id and session, eventually, that should be used for every DataFrame

    Returns
    -------
     NI_arr (ndarray) and NI_participants_df (DataFrame) participants that are also in participants_df


    >>> import numpy as np
    >>> import pandas as pd
    >>> import brainomics.image_preprocessing as preproc
    >>> NI_filenames = ['/neurospin/psy/start-icaar-eugei/derivatives/cat12/vbm/sub-ICAAR017/ses-V1/mri/mwp1sub-ICAAR017_ses-V1_acq-s03_T1w.nii',
    '/neurospin/psy/start-icaar-eugei/derivatives/cat12/vbm/sub-ICAAR033/ses-V1/mri/mwp1sub-ICAAR033_ses-V1_acq-s07_T1w.nii',
    '/neurospin/psy/start-icaar-eugei/derivatives/cat12/vbm/sub-STARTRA160489/ses-V1/mri/mwp1sub-STARTRA160489_ses-v1_T1w.nii']
    >>> NI_arr, NI_participants_df, ref_img = preproc.load_images(NI_filenames, check=dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5)))
    >>> NI_arr.shape
    (3, 1, 121, 145, 121)
    >>> NI_participants_df
      participant_id                                            ni_path
    0       ICAAR017  /neurospin/psy/start-icaar-eugei/derivatives/c...
    1       ICAAR033  /neurospin/psy/start-icaar-eugei/derivatives/c...
    2  STARTRA160489  /neurospin/psy/start-icaar-eugei/derivatives/c...
    >>> other_df=pd.DataFrame(dict(participant_id=['ICAAR017', 'STARTRA160489']))
    >>> NI_arr2, NI_participants_df2 = preproc.merge_ni_df(NI_arr, NI_participants_df, other_df)
    >>> NI_arr2.shape
    (2, 1, 121, 145, 121)
    >>> NI_participants_df2
      participant_id                                            ni_path
    0       ICAAR017  /neurospin/psy/start-icaar-eugei/derivatives/c...
    1  STARTRA160489  /neurospin/psy/start-icaar-eugei/derivatives/c...
    >>> np.all(NI_arr[[0, 2], ::] == NI_arr2)
    True
    """

    # 1) Extracts the session + run if available in participants_df/qc from <ni_path> in NI_participants_df
    unique_key_pheno = [participant_id]
    unique_key_qc = [participant_id]
    NI_participants_df.participant_id = NI_participants_df.participant_id.astype(id_type)
    participants_df.participant_id = participants_df.participant_id.astype(id_type)
    if 'session' in participants_df or (qc is not None and 'session' in qc):
        NI_participants_df['session'] = NI_participants_df.ni_path.str.extract('ses-([^_/]+)/')[0].astype(id_type)
        if 'session' in participants_df:
            participants_df.session = participants_df.session.astype(id_type)
            unique_key_pheno.append('session')
        if qc is not None and 'session' in qc:
            qc.session = qc.session.astype(id_type)
            unique_key_qc.append('session')
    if 'run' in participants_df or (qc is not None and 'run' in qc):
        NI_participants_df['run'] = NI_participants_df.ni_path.str.extract('run-([^_/]+)\_.*nii')[0].fillna(1).astype(str)
        if 'run' in participants_df:
            unique_key_pheno.append('run')
            participants_df.run = participants_df.run.astype(str)
        if qc is not None and 'run' in qc:
            unique_key_qc.append('run')
            qc.run = qc.run.astype(str)

    # 2) Keeps only the matching (participant_id, session, run) from both NI_participants_df and participants_df by
    #    preserving the order of NI_participants_df
    # !! Very import to have a clean index (to retrieve the order after the merge)
    NI_participants_df = NI_participants_df.reset_index(drop=True).reset_index() # stores a clean index from 0..len(df)
    NI_participants_merged = pd.merge(NI_participants_df, participants_df, on=unique_key_pheno,
                                      how='inner', validate='m:1')
    print('--> {} {} have missing phenotype'.format(len(NI_participants_df)-len(NI_participants_merged),
          unique_key_pheno))

    # 3) If QC is available, filters out the (participant_id, session, run) who did not pass the QC
    if qc is not None:
        assert np.all(qc.qc.eq(0) | qc.qc.eq(1)), 'Unexpected value in qc.csv'
        qc = qc.reset_index(drop=True) # removes an old index
        qc_val = qc.qc.values
        if np.all(qc_val==0):
            raise ValueError('No participant passed the QC !')
        elif np.all(qc_val==1):
            pass
        else:
            # Modified this part, indeed, the old code assumes that all subject
            # after idx_first_occurence should be removed, why ?
            # idx_first_occurence = len(qc_val) - (qc_val[::-1] != 1).argmax()
            # assert np.all(qc.iloc[idx_first_occurence:].qc == 1)
            # keep = qc.iloc[idx_first_occurence:][unique_key_qc]
            # New code simply select qc['qc'] == 1
            keep = qc[qc['qc'] == 1][unique_key_qc]
            init_len = len(NI_participants_merged)
            # Very important to have 1:1 correspondance between the QC and the NI_participant_array
            NI_participants_merged = pd.merge(NI_participants_merged, keep, on=unique_key_qc,
                                              how='inner', validate='1:1')
            print('--> {} {} did not pass the QC'.format(init_len - len(NI_participants_merged), unique_key_qc))

    # if merge_ni_path and 'ni_path' in participants_df:
    #     # Keep only the matching session and acquisition nb according to <participants_df>
    #     sub_sess_to_keep = NI_participants_merged['ni_path_y'].str.extract(r".*/.*sub-(\w+)_ses-(\w+)_.*")
    #     sub_sess = NI_participants_merged['ni_path_x'].str.extract(r".*/.*sub-(\w+)_ses-(\w+)_.*")
    #     # Some participants have only one acq, in which case it is not mentioned
    #     acq_to_keep = NI_participants_merged['ni_path_y'].str.extract(r"(acq-[a-zA-Z0-9\-\.]+)").fillna('')
    #     acq = NI_participants_merged['ni_path_x'].str.extract(r"(acq-[a-zA-Z0-9\-\.]+)").fillna('')

    #     assert not (sub_sess.isnull().values.any() or sub_sess_to_keep.isnull().values.any()), \
    #         "Extraction of session_id or participant_id failed"

    #     keep_unique_participant_ids = sub_sess_to_keep.eq(sub_sess).all(1).values.flatten() & \
    #                                   acq_to_keep.eq(acq).values.flatten()

    #     NI_participants_merged = NI_participants_merged[keep_unique_participant_ids]
    #     NI_participants_merged.drop(columns=['ni_path_y'], inplace=True)
    #     NI_participants_merged.rename(columns={'ni_path_x': 'ni_path'}, inplace=True)


    unique_key = unique_key_qc if set(unique_key_qc) >= set(unique_key_pheno) else unique_key_pheno
    assert len(NI_participants_merged.groupby(unique_key)) == len(NI_participants_merged), \
        '{} similar pairs {} found'.format(len(NI_participants_merged)-len(NI_participants_merged.groupby(unique_key)),
                                           unique_key)

    # Get back to NI_arr using the indexes kept in NI_participants through all merges
    idx_to_keep = NI_participants_merged['index'].values

    # NI_participants_merged.drop('index')
    return NI_arr[idx_to_keep], NI_participants_merged

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

    # Rm participants with missing keys_required
    null_or_nan_mask = [False for _ in range(len(phenotype))]
    for key in keys_required:
        null_or_nan_mask |= getattr(phenotype, key).isnull() | getattr(phenotype, key).isna()
    if null_or_nan_mask.sum() > 0:
        print('Warning: {} participant_id will not be considered because of missing required values:\n{}'. \
              format(null_or_nan_mask.sum(), list(phenotype[null_or_nan_mask].participant_id.values)))

    participants_df = phenotype[~null_or_nan_mask]

    ########################################################################################################################
    #  Neuroimaging niftii and TIV
    #  mwp1 files
      #  excpected image dimensions
    NI_filenames = glob.glob(nii_path)
    ########################################################################################################################
    #  Load images, intersect with pop and do preprocessing and dump 5d npy
    print("###########################################################################################################")
    print("#", dataset_name)

    print("# 1) Read images")
    scaling, harmo = 'raw', 'raw'
    print("## Load images")
    NI_arr, NI_participants_df, ref_img = preproc.load_images(NI_filenames,check=check)

    print('--> {} img loaded'.format(len(NI_participants_df)))
    print("## Merge nii's participant_id with participants.csv")
    NI_arr, NI_participants_df = merge_ni_df(NI_arr, NI_participants_df, participants_df,
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
    #  Neuroimaging niftii and TIV
    #  mwp1 files
      #  excpected image dimensions
    NI_filenames = glob.glob(nii_path)
    ########################################################################################################################
    #  Load images, intersect with pop and do preprocessing and dump 5d npy
    print("###########################################################################################################")
    print("#", dataset_name)

    print("# 1) Read images")
    scaling, harmo = 'raw', 'raw'
    print("## Load images")
    # MODIF 1:
    #NI_arr, NI_participants_df, ref_img = preproc.load_images(NI_filenames,check=check)
    NI_arr, NI_participants_df, ref_img = img_to_array(NI_filenames)
    assert np.all(NI_arr == imgs_arr)
    print('--> {} img loaded'.format(len(NI_participants_df)))

    #imgs_df.participant_id.equals(NI_participants_df.participant_id)

    print("## Merge nii's participant_id with participants.csv")
    # MODIF 2:
    # NI_arr_, NI_participants_df_ = preproc.merge_ni_df(NI_arr, NI_participants_df, participants_df,
    #                                                     qc=qc, id_type=id_type)
    NI_arr, NI_participants_df = merge_ni_df(NI_arr, NI_participants_df, participants_df,
                                                         qc=qc, id_type=id_type)

    print('--> Remaining samples: {} / {}'.format(len(NI_participants_df), len(participants_df)))

    print("## Save the new participants.csv")
    NI_participants_df.to_csv(OUTPUT_CAT12(dataset_name, output_path, scaling=None, harmo=None, type="participants", ext="csv"),
                              index=False)
    print("## Save the raw npy file (with shape {})".format(NI_arr.shape))
    np.save(OUTPUT_CAT12(dataset_name, output_path, scaling=scaling, harmo=harmo, type="data64", ext="npy"), NI_arr)
    NI_arr = np.load(OUTPUT_CAT12(dataset_name, output_path, scaling=scaling, harmo=harmo, type="data64", ext="npy"), mmap_mode='r')

    # print("## Compute brain mask")
    # mask_img = preproc.compute_brain_mask(NI_arr, ref_img, mask_thres_mean=0.1, mask_thres_std=1e-6,
    #                                      clust_size_thres=10,
    #                                      verbose=1)
    # mask_arr = mask_img.get_data() > 0
    # print("## Save the mask")
    # mask_img.to_filename(OUTPUT_CAT12(dataset_name, output_path, scaling=None, harmo=None, type="mask", ext="nii.gz"))

    ########################################################################################################################
    print("# 2) Raw data")
    # Univariate stats

    # design matrix: Set missing diagnosis to 'unknown' to avoid missing data(do it once)
    dmat_df = NI_participants_df[['age', 'sex', 'tiv']]
    assert np.all(dmat_df.isnull().sum() == 0)
    # print("## Do univariate stats on age, sex and TIV")
    # univmods, univstats = univ_stats(NI_arr.squeeze()[:, mask_arr], formula="age + sex + tiv", data=dmat_df)

    # %time univmods, univstats = univ_stats(NI_arr.squeeze()[:, mask_arr], formula="age + sex + diagnosis + tiv + site", data=dmat_df)
    # pdf_filename = OUTPUT_CAT12(dataset_name, output_path, scaling=scaling, harmo=harmo, type="univstats", ext="pdf")
    # plot_univ_stats(univstats, mask_img, data=dmat_df, grand_mean=NI_arr.squeeze()[:, mask_arr].mean(axis=1),
    #                pdf_filename=pdf_filename, thres_nlpval=3,
    #               skip_intercept=True)

    ########################################################################################################################
    print("# 3) Global scaling")
    scaling, harmo = 'gs', 'raw'

    print("## Apply global scaling")
    NI_arr = preproc.global_scaling(NI_arr, axis0_values=np.array(NI_participants_df.tiv), target=1500)
    # Save
    # RM data64 always in 64
    # RM harmo no harmonization
    print("## Save the new .npy array")
    np.save(OUTPUT_CAT12(dataset_name, output_path, scaling=scaling, harmo=harmo, type="data64", ext="npy"), NI_arr)
    NI_arr = np.load(OUTPUT_CAT12(dataset_name, output_path, scaling=scaling, harmo=harmo, type="data64", ext="npy"), mmap_mode='r')

    # # Univariate stats
    # print("## Recompute univariate stats on age, sex and TIV")
    # univmods, univstats = univ_stats(NI_arr.squeeze()[:, mask_arr], formula="age + sex + tiv", data=dmat_df)
    # pdf_filename = OUTPUT_CAT12(dataset_name, output_path, scaling=scaling, harmo=harmo, type="univstats", ext="pdf")
    # plot_univ_stats(univstats, mask_img, data=dmat_df, grand_mean=NI_arr.squeeze()[:, mask_arr].mean(axis=1),
    #                 pdf_filename=pdf_filename, thres_nlpval=3,
    #                 skip_intercept=True)
    # Deallocate the memory
    del NI_arr




