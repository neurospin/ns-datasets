# NeusoSpin Datasets

Processing scripts of NeusoSpin dataset

The psy dataset is divided in two directories :
- **/neurospin/psy** : Safe shared data.(finished and complete work)
- **/neurospin/psy_sbox** : Mirror of the psy directory. (pending work)
Note : Due to insufficient memory space, the synchronisation between these two directories is stopped for the time being.
The organisation of these directories is in **bids format**. Please for more details refer to the links below:
https://bioproj.cea.fr/nextcloud/f/118432
https://bids-specification.readthedocs.io/en/stable/

monitoring :
Please refer to this table for more information about a **dataset**:
http://mart.intra.cea.fr/neurospin_datasets/
Please refer to this table for more information about the **preprocessings** done in a dataset:
https://bioproj.cea.fr/nextcloud/f/117205

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
 
# Phenotypes

Contained **cured** phenotypes, `.tsv` files must contains a `participant_id` column. The "dirty" phenotypes must be saved in `sourcedata/phenotypes`. See detailed procedure **https://bioproj.cea.fr/nextcloud/f/118432**

# Scripts

```
<study>/
├── participants_make_dataset.py: build the participants.tsv file
├── source_to_bids_2021_t1mri_from_london_iop.py: re-organize file from source data to rawdata 
├── <soft>_<01_first_processing>.py: do some pre-processing
├── <soft>_<02_second_processing>.py: do some pre-processing
├── <soft>_make_dataset.py (ex: cat12vbm_make_dataset.py): build `array` dataset
```
