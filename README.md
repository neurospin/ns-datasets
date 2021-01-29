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
└── derivatives (processed data)
    ├── <soft>-<vesion>_<output> ex: cat12-12.6_vbm
    └── <soft>-<vesion>_<output>_qc ex: cat12-12.6_vbm_qc
        └─ qc.tsv
```

## sourcedata

Put your "raw" unorganized data, both images and **phenotypes**.

## rawdata

niftti image in bids format

## derivatives

### Data

`<soft>-<vesion>_<output> ex: cat12-12.6_vbm` data organized in BIDS.

### Quality check (QC)

`<soft>-<vesion>_<output>_qc ex: cat12-12.6_vbm_qc` Quality check: contains a file `qc.tsv` with at least two column `participant_id` and  `qc` in `[0, 1]`.
  
# Phenotypes

Should contained *cured* phenotypes, `.tsv` files must contains a `participant_id` column.

# Scripts

```
<study>/
├── participants_make_dataset.py: build the participants.tsv file
├── <soft>_<01_first_processing>.py: do some pre-processing
├── <soft>_<02_second_processing>.py: do some pre-processing
├── <soft>_make_dataset.py (ex: cat12vbm_make_dataset.py): build `array` dataset
```
