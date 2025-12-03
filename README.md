# 🧬 Computational Design of Metallohydrolases

🚧🚧 **THIS REPOSITORY IS UNDER ACTIVE CONSTRUCTION** 🚧🚧  
We are currently **restructuring, cleaning, and unifying** this repository following the acceptance of our manuscript in **Nature**.  
Over the next several weeks, we will be:

- Harmonizing folder organization and naming conventions  
- Updating dependencies and removing legacy paths  
- Clarifying whether Apptainer-based execution is required for full reproducibility  
- Improving tutorial readability for new users  
- Uploading data and documentation  

Please expect rapid updates—thank you for your patience! Please reach out to Seth Woodbury (woodbuse@uw.edu) or Donghyo Kim (donghyo@uw.edu) for questions, concerns, bugs, or collaboration. Happy designing!! ✨

---

## 📄 Nature Publication

**Computational Design of Metallohydrolases**  
**Published in *Nature* on December 3, 2025**

**DOI:** https://doi.org/10.1038/s41586-025-09746-w  
**URL:** https://www.nature.com/articles/s41586-025-09746-w

## 👥 Authors
**Donghyo Kim‡, Seth M. Woodbury‡, Woody Ahern‡, Doug Tischer, Alex Kang, Emily Joyce,  
Asim K. Bera, Nikita Hanikel, Saman Salike, Rohith Krishna, Jason Yim,  
Samuel J. Pellock, Anna Lauko, Indrek Kalvet\*, Donald Hilvert\*, David Baker\***

‡Co-first authors, \*Corresponding authors

> 📄 _This manuscript applies RFdiffusion2, a generative AI model for de novo enzyme design, to build highly active zinc-dependent hydrolases from quantum-chemistry-defined active site geometries._

---

## 📁 Repository Overview

This repository contains:

- **Tutorials & Reproduction Pipelines**
  - A JupyterHub notebook tutorial on how to use RFdiffusion2 (`RFdiffusion2_Tutorial_JuptyerNotebook.ipynb`) starting from your own input from a pre-existing PDB or a quantum chemistry-derived theozyme (worked examples for each case). Many scripts have been made to streamline this process.
  - A reproduction tutorial for **Design Campaign 1**, nearly identical to what was provided to peer-reviewers (`design_zn_hydrolase.ipynb`). This contains the most important pipeline steps that were performed in the first design campaign, although we are working on modernizing it, to make it more user-friendly, and releasing the notebook for **Design Campaign 2**.

- **Dry Lab Data**
  - DFT-optimized theozymes  
  - Design models of the ordered & tested designs  

- **Wet Lab Data**
  - DNA & protein sequences  
  - Expression sequences  
  - Kinetic measurements (kcat, KM, kcat/KM)  
  - Other wetlab data & all analysis/plotting! 

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
conda env create -f Environment/zinc_hydro.yml -n zinc_hydro
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
- Additionally, make sure you have access to the following pre-trained models (which are available as submodules in this repo in the Software subdirectory)
  - [RFdiffusion2](https://github.com/RosettaCommons/RFdiffusion2)
  - [Protein/LigandMPNN](https://github.com/dauparas/LigandMPNN)
  - [PLACER](https://github.com/baker-laboratory/PLACER)
  - [openfold](https://github.com/aqlaboratory/openfold)

---

## 🧾 Citation
- If you use this repository, please cite:
- DOI will be added upon publication