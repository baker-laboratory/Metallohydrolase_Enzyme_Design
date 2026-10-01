# ZAPP-1 intact-protein mass spectrometry

These are original instrument-exported PNG images of deconvoluted intact-protein ESI mass spectra. Image bytes are preserved; the TEO filenames identify the displayed view. The [manifest](manifest.csv) records each original filename and SHA-256 checksum. Numeric spectra and vendor acquisition files are not included.

## Tagless ZAPP-1

[Original mass-spectrum image](ZAPP1_tagless/SMW_ZAPP_tagless.png). The instrument labels the principal peak at 22,466.27 Da. The final supplementary figure combines the mass spectrum with the tagless-protein kinetics: [PDF](../wetlab_data_plots/paper_figures/pdf/PTE__v2SI__ZAPP1_tagless.pdf), [PNG](../wetlab_data_plots/paper_figures/png/PTE__v2SI__ZAPP1_tagless.png), and [source data](../supplemental_data/ZAPP1_tagless/README.md).

## TEO treatment of ZAPP-1

TEO denotes triethyloxonium tetrafluoroborate. The original sample labels are retained, and suffixes 1 and 2 identify the two labeled replicates of each treatment. Each pair contains two views of the same acquisition, rather than two additional replicates. TEO:protein molar equivalents are approximate and follow the manuscript methods; the untreated control contains no TEO. Sample labels and treatment equivalents are also provided in [sample_metadata.csv](sample_metadata.csv).

| Sample label | TEO:protein equivalents | Overview | Zoom |
| --- | --- | --- | --- |
| SMW_P1D1_Control | 0 | [PNG](TEO/SMW_P1D1_Control__overview.png) | [PNG](TEO/SMW_P1D1_Control__zoom.png) |
| SMW_P1D1_High_TEO_1 | ≈3,270 | [PNG](TEO/SMW_P1D1_High_TEO_1__overview.png) | [PNG](TEO/SMW_P1D1_High_TEO_1__zoom.png) |
| SMW_P1D1_High_TEO_2 | ≈3,270 | [PNG](TEO/SMW_P1D1_High_TEO_2__overview.png) | [PNG](TEO/SMW_P1D1_High_TEO_2__zoom.png) |
| SMW_P1D1_Low_TEO_1 | ≈1,635 | [PNG](TEO/SMW_P1D1_Low_TEO_1__overview.png) | [PNG](TEO/SMW_P1D1_Low_TEO_1__zoom.png) |
| SMW_P1D1_Low_TEO_2 | ≈1,635 | [PNG](TEO/SMW_P1D1_Low_TEO_2__overview.png) | [PNG](TEO/SMW_P1D1_Low_TEO_2__zoom.png) |

Peak labels and acquisition annotations are retained as supplied by the instrument software. These image exports do not provide site-specific modification assignments.
