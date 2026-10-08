# Translational V7 Poster Figures

These figures are generated from the schema-8 V7 metrics file by:

```bash
python scripts/41_create_v7_poster_figures.py
```

## Recommended use

- `01_v7_poster_graphical_abstract`: Main poster figure. It combines the three
  datasets, native-montage processing, V7 fusion and patient calibration,
  binary output, headline metrics, domain results, and validation boundary.
- `02_v7_poster_dataset_pipeline`: Methods figure. It provides a larger,
  presentation-readable dataset-to-output workflow.
- `03_v7_poster_results_dashboard`: Results figure. It emphasizes the V5/V6/V7
  progression, domain variation, auxiliary weights, and operating-point
  tradeoff.

Each figure is exported as a 300-DPI PNG and as vector SVG and PDF files.

## Interpretation boundary

All values are development-observed and cue-locked. Specificity is measured per
rest window, not as continuous false activations per minute. The locked external
cohort and continuous no-control evaluation remain unavailable.
