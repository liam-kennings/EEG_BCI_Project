# Dataset Links

These links make the final project runnable without duplicating approximately
41 GB of EEG recordings.

| Link | Dataset |
|---|---|
| `EEG_BCI_Project_PhysioNet_EEGMMIDB` | PhysioNet EEG Motor Movement/Imagery Database |
| `EEG_BCI_Project_OpenBMI` | OpenBMI motor-imagery dataset |
| `EEG_BCI_Project_Liu_Stroke` | Liu acute-stroke motor-imagery dataset |

The same datasets are linked into the paths expected by the scripts:

```text
files/eegmmidb/1.0.0
files/external/openbmi/nm000338
Liu acute-stroke dataset
```

If the project or datasets move, run
`EEG_BCI_Project_CONFIGURE_DATA.command` from the project root with the three
new dataset paths. An optional fourth path can point to the folder containing
the existing `translational_windows_v3.npz` cache.

