# NeusoSpin Datasets

Processing scripts of NeusoSpin dataset

[Summary of psy datasets in NeuroSpin](https://bioproj.cea.fr/nextcloud/f/118432)


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

**Unprocessed Niftti images** organized in **[BIDS](https://bids-specification.readthedocs.io/en/stable/)** format see detailed description [JULIE ADD A LINK HERE TO NEXTCLOUD FILE](https://).

## Processed images: directory `derivatives`

**Processed Niftti images** organized in **[BIDS](https://bids-specification.readthedocs.io/en/stable/)** format see detailed description [JULIE ADD A LINK HERE TO NEXTCLOUD FILE](https://).

General organization

- Data: `<soft>-<vesion>_<output>` ex: `cat12-12.6_vbm`
- Quality check `<soft>-<vesion>_<output>_qc` ex: `cat12-12.6_vbm_qc`. This directory MUST contain a file `qc.tsv` with at least two column `participant_id` and  `qc` in `[0, 1]`.
 
# Phenotypes

Contained **cured** phenotypes, `.tsv` files must contains a `participant_id` column. The "dirty" phenotypes must be saved in `sourcedata/phenotypes`.

# Scripts

```
<study>/
├── participants_make_dataset.py: build the participants.tsv file
├── source_to_bids_2021_t1mri_from_london_iop.py: re-organize file from source data to rawdata 
├── <soft>_<01_first_processing>.py: do some pre-processing
├── <soft>_<02_second_processing>.py: do some pre-processing
├── <soft>_make_dataset.py (ex: cat12vbm_make_dataset.py): build `array` dataset
```
