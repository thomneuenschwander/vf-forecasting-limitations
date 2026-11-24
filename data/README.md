# Data

This study uses the public **UWHVF** dataset (Montesano et al., 2022), distributed
under the 3-clause BSD license at <https://github.com/uw-biomedical-ml/uwhvf>.

The dataset is not versioned here. Download these three files from that repository
and place them in `data/raw/`:

| File | Used for |
|---|---|
| `alldata.json` | exams (sensitivities, TD, age) and grid coordinates |
| `Coord_242.csv` | location metadata; cluster 0 marks the two blind spots |
| `Series_Summary.csv` | MTD slope per eye, used only to balance the folds |

`scripts/01_prepare_data.py` checks the sha256 of each file against the version used
in the paper and stops if they differ.
