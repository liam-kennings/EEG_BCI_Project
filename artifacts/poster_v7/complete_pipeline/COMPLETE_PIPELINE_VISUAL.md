# Complete V7 Pipeline Visual

`09_v7_complete_pipeline` combines the complete V7 workflow in one figure:

1. Multi-dataset population EEG.
2. Native-montage quality control and harmonization.
3. Parallel MRCP, SMR, and Riemannian representations.
4. Controller-selected general-model fusion.
5. Thirty-window patient calibration from recording 0.
6. Patient-adapted intent probability.
7. Binary `REST` or `INTENT` safety gate.
8. Later-recording evaluation and headline metrics.

The figure also marks the validation boundary and states that the reported
result is development observed and cue locked.

Regenerate the PNG, SVG, and PDF exports with:

```bash
python scripts/45_create_v7_complete_pipeline_visual.py
```
