# NeuroSpin Datasets

Processing scripts of NeuroSpin dataset

# Images organization

```
study/
├── README (date of data acquisition/upload, what has been done on the data)
├── participants.tsv 
├── sourcedata ("raw" unorganized data)
│   ├── images
│   └── phenotypes
├── rawdata (niftti image in bids format)
├── phenotypes cured "clean" phenotypes
└── derivatives (processed data)
    ├── <soft>-<vesion>_<output> ex: cat12-12.6_vbm
    └── <soft>-<vesion>_<output>_qc ex: cat12-12.6_vbm_qc
        └─ qc.tsv
```

## Unorganized data: directory `sourcedata`

Contains "raw" **unorganized data**, both **images** and **phenotypes**.

When you receive new data:

1. Put everything in `sourcedata/year_<data description>` ex: `2021_t1mri_from_london_iop`
2. Write a scripts `source_to_<bids|phenotypes|...>_<t1mri|dwi|...>.py` that re-organize data into bids or cured phenotypes. This script copy unorganized data into organized folders:
    * `rawdata` for images.
    * `phenotypes` for phenotypes.

## Rawdata: directory `rawdata`

**Unprocessed Niftti images** organized in **[BIDS](https://bids-specification.readthedocs.io/en/stable/)** format see detailed description **https://bioproj.cea.fr/nextcloud/f/118432**

## Processed images: directory `derivatives`

**Processed Niftti images** organized in **[BIDS](https://bids-specification.readthedocs.io/en/stable/)** format see detailed description **https://bioproj.cea.fr/nextcloud/f/118432**

General organization

- Data: `<soft>-<vesion>_<output>` ex: `cat12-12.6_vbm`
- Quality check `<soft>-<vesion>_<output>_qc` ex: `cat12-12.6_vbm_qc`. This directory MUST contain a file `qc.tsv` with at least two column `participant_id` and  `qc` in `[0, 1]`.
 
# Phenotypes: directories `sourcedata/phenotypes` and `phenotypes`

1. **Unorganized "dirty" phenotypes** go in `sourcedata/phenotypes/year_some_description`.
2. **Cured** phenotypes are saved in dicrectory `phenotypes/year_some_description*`.
   Cured phenotypes `.tsv` files must contains a `participant_id` column.
  Ideally, curation should be made by scripts (ex: `phenotypes_make_dataset_<year>_<some_description>.py`) saved in this git repository.

# Scripts

```
<study>/
├── phenotypes_make_dataset_<year>_<some_description>.py: re-organize file from source data to rawdata
├── participants_make_dataset.py: build the participants.tsv file
├── source_to_bids_2021_t1mri_from_london_iop.py: re-organize file from source data to rawdata
├── <soft>_<01_first_processing>.py: do some pre-processing
├── <soft>_<02_second_processing>.py: do some pre-processing
├── <soft>_make_dataset.py (ex: cat12vbm_make_dataset.py): build `array` dataset
```

# Psy datasets

The psy dataset are stored into two directories :
- **/neurospin/psy** : Safe shared data, finished and completed work. Ask read Read acces.
- **/neurospin/psy_sbox** : Mirror (sandbox) of the psy directory. Ask read Read/write acces.

Note : Due to insufficient memory space, the synchronisation between these two directories is stopped for the time being.

The organisation of these directories is in **[BIDS](https://bids-specification.readthedocs.io/en/stable/)**. Please for more details refer to the links below:
https://bioproj.cea.fr/nextcloud/f/118432 , file `data_procedures.docx`.


## Monitoring :

Please refer to this table for more information about a **dataset**:
1. https://bioproj.cea.fr/nextcloud/f/117205 file `neurospin_repository_monitor.xlsx`: global overview of the datasets (size, modality, pre-processing, etc.).
2. http://mart.intra.cea.fr/neurospin_datasets/: Who is working on what.

## Input array datasets

Ready-to-use array datasets. 

```
/neurospin/tmp/psy_sbox/all_studies/derivatives/arrays/
├── biobd_cat12vbm_mwp1-gs.npy
├── biobd_cat12vbm_participants.csv
├── biobd_cat12vbm_rois-gs.csv
├── bsnip1_cat12vbm_mwp1-gs.npy
├── bsnip1_cat12vbm_participants.csv
├── bsnip1_cat12vbm_rois-gs.csv
├── ...
├── mni_brain-gm-mask_1.5mm.nii.gz
└── mni_cerebrum-gm-mask_1.5mm.nii.gz
```

## Output data

- `/neurospin/tmp/psy_sbox/analysis/year_study-name` : Make your experiment here: those are preliminary data that may change. The direcory **`/neurospin/tmp`, has  no backup**. This save disk space. You can use **`/neurospin/tmp/<user>`**.

- `/neurospin/psy_sbox/analysis/year_study-name` : When you are happy sync your results here. Make your experiment here: those are preliminary data that may change. The direcory **has backup**.
