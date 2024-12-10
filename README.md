# Transcriptomic changes in retinal ganglion cell types associated with the disruption of cholinergic retinal waves

This repository contains code and processed files used to produce the analyses published in [**Transcriptomic changes in retinal ganglion cell types associated with the disruption of cholinergic retinal waves**](https://www.biorxiv.org/content/10.1101/2024.12.05.627027v1).

## Repository Structure

The repository is structured according to the order in which scripts should be run. All scripts are provided as Jupyter Notebooks:
* **1-DataProcessing**: Contains processing and classification steps (associated with Figures 1 and 2).
* **2-ExpressionChanges**: Contains analysis of wave-dependent and developmental changes at the global and type-specific levels (associated with Figure 3).
* **3-SubclassComparison**: Contains classification of direction-selective ganglion cell types and analysis of subclass-level changes (associated with Figure 4).
* **4-FISHAnalysis**: Contains analysis of fluorescence *in situ* hybridization (FISH) images (associated with Figure 5).

## Usage

The numbering of the folders and files indicates the order in which code should be run. File paths must be edited to reflect the paths used on the local machine. Matrix files required for the analysis as well as final h5ad files containing log-normalized counts and relevant metadata can be found here:
https://drive.google.com/drive/folders/1H4-ibHUhqa4l_vi_y7ZXOB3-lf_Fp4sW?usp=drive_link

## Contact

For questions about the data or code, please email [Matthew Po](mailto:matthew.po@berkeley.edu) or [Karthik Shekhar](mailto:kshekhar@berkeley.edu).

## Cite

If you make use of our code or findings, please cite this work as:
R. D. Somaiya, M. A. Po, M. B. Feller, and K. Shekhar (2024). Transcriptomic changes in retinal ganglion cell types associated with the disruption of cholinergic retinal waves. *In submission*.
