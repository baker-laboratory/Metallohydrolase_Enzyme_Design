# 🧬 Computational Design of Metallohydrolases

This repository contains code and data to reproduce the key results from the manuscript:

## Computational Design of Metallohydrolases
Donghyo Kim†, Seth M. Woodbury†, Woody Ahern†, Doug Tischer, Nikita Hanikel, Saman Salike, Jason Yim, Samuel J. Pellock, Anna Lauko, Indrek Kalvet*, Donald Hilvert*, David Baker*  
†Co-first authors, *Corresponding authors

> 📄 _The manuscript apllies RFdiffusion2, a generative AI method for de novo enzyme design, to build highly active zinc-dependent hydrolases from quantum chemistry-based active site geometries._

---

## 📁 Repository Overview

- `design_zn_hydrolase.ipynb`  
  A Jupyter Notebook to reproduce the core design and analysis pipeline for zinc metallohydrolases.

- `env/zinc_hydro.yml`  
  Conda environment specification file.

- `software/`  
  Folder containing required software dependencies, such as `RFdiffusion2`, `ProteinMPNN`, `PLACER`, and utility scripts.

---

## 🧪 Setup Instructions

### 1. Clone the repository and submodules. 
```bash
git clone https://github.com/SethWoodbury/Computational_Design_of_Metallohydrolases_PrivateGitHub.git
cd Computational_Design_of_Metallohydrolases_PrivateGitHub
git submodule init 
git submodule update
````

### 2. Install Conda (if not already installed)
[Conda Installation Guide](https://docs.conda.io/en/latest/miniconda.html)

### 3. Create and activate the environment
```bash
conda env create -f env/zinc_hydro.yml -n zinc_hydro
conda activate zinc_hydro
```
### 4. Register as Jupyter kernel
```bash
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=zinc_hydro
```

---

## 🔗 Data & Models
- AlphaFold2 parameters are required and must be downloaded using:
  ```bash
  bash software/superfold/scripts/download_all_data.sh ./software/superfold/alphafold_weights.pth
  ```
- This requires `aria2c`, which can be installed with:
  ```bash
  sudo apt install aria2
  ```
- Additionally, make sure you have access to the following pre-trained models
  - [RFdiffusion2](https://github.com/RosettaCommons/RFdiffusion2)
  - [Protein/LignadMPNN](https://github.com/dauparas/LigandMPNN)
  - [PLACER](https://github.com/baker-laboratory/PLACER)
  - [openfold](https://github.com/aqlaboratory/openfold)

---

## 🧾 Citation
- If you use this repository, please cite:
  `Kim, Woodbury, Ahern, et al. Computational Design of Metallohydrolases.`
- DOI will be added upon publication
