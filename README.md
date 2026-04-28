# Traditional bioacoustic analyses and machine-learning methods indicate weak vocal dimorphism in four Cerrado antbird species
## _Paper in review at the Journal of Ornithology_
This repository contains all code and scripts used to reproduce the analyses presented in the manuscript on acoustic sexual dimorphism in birds.

The project integrates:

* Traditional acoustic analyses (R; Raven Pro measurements)
* Machine learning classification (Python; BirdNET embeddings, MFCCs, and standard features)
* Signal processing (MFCC extraction from audio files)

---

## Repository Structure

```
Dimorphism-paper-JORN/
│
├── data/
│   ├── raven_exports/        # Raven Pro exported acoustic measurements
│   └── metadata/             
│
├── scripts/
│   ├── python/
│   │   ├── classification/
│   │   │   ├── classify.py
│   │   │   └── classification_and_graphs.py
│   │   └── mfcc/
│   │       └── extract_mfcc.py
│   │
│   └── r/
│       ├── 01_preprocess_and_pca.R
│       ├── 02_lmm_models.R
│       └── 03_species_specific.R
│
├── results/
│   ├── figures/              # Output plots (accuracy, recall, etc.)
│   └── tables/               # Model summaries and statistics
│
├── environment/
│   ├── environment.yml       # Conda environment (recommended)
│   └── requirements.txt      # Python dependencies
│
└── README.md
```

---

## Reproducing the Analyses

### 1. Clone the repository

```
git clone https://github.com/yourusername/Dimorphism-paper-JORN.git
cd Dimorphism-paper-JORN
```

---

## Python Analyses

### Classification (SVM vs Dummy baseline)

Runs classification pipelines using different feature sets (e.g., BirdNET embeddings, MFCCs, standard acoustic features).

```
python scripts/python/classification/run_classifiers.py
```

Outputs:

* Accuracy distributions (violin plots)
* Recall distributions (violin plots)

Saved in:

```
results/figures/
```

---

### MFCC Extraction

Extract Mel-frequency cepstral coefficients (MFCCs) from audio recordings.

```
python scripts/python/mfcc/extract_mfcc.py
```

---

## R Analyses (Traditional Acoustic Approach)

These scripts analyze manually annotated acoustic parameters exported from Raven Pro.

### Run full pipeline

```
Rscript scripts/r/01_preprocess_and_pca.R
Rscript scripts/r/02_lmm_models.R
Rscript scripts/r/03_species_specific.R
```

### Input data

Place Raven exports in:

```
data/raven_exports/tabg.csv
```

### Outputs

* PCA objects
* Linear mixed model summaries
* Species-specific analyses

Saved in:

```
results/
```

---

## Environment Setup

### Option 1: Conda (recommended)

```
conda env create -f environment/environment.yml
conda activate dimorphism-env
```

### Option 2: pip

```
pip install -r environment/requirements.txt
```

---

## Notes on Reproducibility

* Analyses use fixed random seeds where applicable
* Classification uses **group-aware splits** to avoid data leakage across individuals
* Dummy classifiers are included as baselines for comparison
* **Audio data is available upon request**

---

## Contact

For questions or issues, please open a GitHub issue or contact the author through:

* [enrico.breviglieri@unesp.br](mailto:enrico.breviglieri@unesp.br)
* [enricolopesbrevi@ufl.edu](mailto:enricolopesbrevi@ufl.edu)
