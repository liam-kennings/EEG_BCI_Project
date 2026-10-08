# V7 Poster Section Visuals

Standalone visuals for Sections 3 through 7 of
`docs/final/EEG_BCI_V7_POSTER_CONTENT.md`.

- `03_datasets_and_cohorts`: cohort composition, retained windows, and negative-state semantics.
- `04_signal_processing_and_features`: native-montage QC and parallel physiological branches.
- `05_v7_decoder_and_patient_calibration`: population fusion, 30-window calibration, auxiliary direction, and binary output.
- `06_evaluation_protocol_and_leakage_control`: subject roles, recording boundary, and primary metrics.
- `07_v7_results_and_domain_variation`: headline metrics, domain results, adaptation effect, and version comparison.

Each visual is exported as a 300-DPI PNG and vector SVG/PDF. Regenerate with:

```bash
python scripts/42_create_v7_poster_section_visuals.py
```

All results are development observed and cue locked.
