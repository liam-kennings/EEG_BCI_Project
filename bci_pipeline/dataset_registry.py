"""Curated motor-intent datasets and integration metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


CANONICAL_LABELS = ("rest", "left_hand", "right_hand")
CORE_MOTOR_CHANNELS = (
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
)


@dataclass(frozen=True)
class MotorImageryDataset:
    """Research metadata required before a dataset is used by the project."""

    key: str
    name: str
    cohort: str
    subjects: int
    sessions: str
    eeg_channels: str
    sampling_hz: str
    tasks: tuple[str, ...]
    rest_support: str
    file_formats: tuple[str, ...]
    license: str
    access_url: str
    publication_url: str
    priority_tier: int
    recommended_role: str
    integration_status: str
    caveat: str
    modalities: tuple[str, ...] = ("EEG",)
    movement_ground_truth: str = "Cue and protocol event markers only"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


DATASET_CATALOG = (
    MotorImageryDataset(
        key="eegmmidb",
        name="PhysioNet EEG Motor Movement/Imagery Dataset",
        cohort="Healthy",
        subjects=109,
        sessions="14 runs",
        eeg_channels="64",
        sampling_hz="160",
        tasks=(
            "left/right fist imagery",
            "both-fists/both-feet imagery",
            "matched execution",
        ),
        rest_support="Annotated T0 intervals and separate baseline runs",
        file_formats=("EDF+",),
        license="Open Data Commons Attribution",
        access_url="https://physionet.org/content/eegmmidb/1.0.0/",
        publication_url="https://physionet.org/content/eegmmidb/1.0.0/",
        priority_tier=0,
        recommended_role="Fixed reference benchmark and ontology anchor",
        integration_status="Native project loader",
        caveat="T1/T2 meanings depend on run number; cue onset is not measured movement onset.",
        modalities=("EEG",),
        movement_ground_truth="Cue markers; no EMG or kinematic movement-onset measurement",
    ),
    MotorImageryDataset(
        key="yang2025",
        name="WBCIC Multi-day High-quality Motor Imagery",
        cohort="Healthy",
        subjects=51,
        sessions="3 days",
        eeg_channels="59 raw / 58 processed",
        sampling_hz="1000 raw / 250 processed",
        tasks=("left hand", "right hand"),
        rest_support="Timed inter-trial rest and resting-state recordings",
        file_formats=("BDF", "MAT", "BIDS"),
        license="CC BY 4.0",
        access_url="https://doi.org/10.25452/figshare.plus.22671172",
        publication_url="https://doi.org/10.1038/s41597-025-04826-y",
        priority_tier=1,
        recommended_role="Multi-session healthy pretraining and session-shift validation",
        integration_status="BIDS/BDF adapter required",
        caveat="Use the 51-subject two-class cohort first; the 11-subject three-class cohort is separate.",
    ),
    MotorImageryDataset(
        key="liu2024",
        name="Acute Stroke Hand-grip Motor Imagery",
        cohort="Acute stroke",
        subjects=50,
        sessions="1",
        eeg_channels="29 EEG + 2 EOG",
        sampling_hz="500",
        tasks=("left hand grip", "right hand grip"),
        rest_support="Within-trial non-imagery intervals; no independent rest class",
        file_formats=("MAT", "EDF", "BIDS"),
        license="CC BY 4.0",
        access_url="https://ww2.nemar.org/dataset/nm000158",
        publication_url="https://doi.org/10.1038/s41597-023-02787-8",
        priority_tier=1,
        recommended_role="Clinical transfer learning and acute-stroke external validation",
        integration_status="Local EDF adapter and harmonized trial manifest available",
        caveat="Only 40 trials per patient; published data-quality warnings identify several artifact-heavy subjects.",
    ),
    MotorImageryDataset(
        key="uet175",
        name="UET175 Post-stroke Motor Imagery",
        cohort="Post-stroke",
        subjects=30,
        sessions="1-3",
        eeg_channels="22",
        sampling_hz="128",
        tasks=("left arm", "right arm", "left leg", "right leg"),
        rest_support="Explicit 2-second relax periods and 20-second run breaks",
        file_formats=("CSV", "JSON", "TXT"),
        license="CC BY 4.0",
        access_url="https://github.com/nmk-k66-uet/UET175",
        publication_url="https://doi.org/10.3389/fnins.2025.1580931",
        priority_tier=1,
        recommended_role="Low-cost clinical-device domain adaptation and artifact robustness",
        integration_status="CSV protocol adapter required",
        caveat="Archive extraction requires contacting the authors for a password; only arm trials map to this project's direction task.",
    ),
    MotorImageryDataset(
        key="hs_stroke_2026",
        name="HS-Stroke Upper-limb Motor Imagery",
        cohort="Chronic stroke",
        subjects=14,
        sessions="20 planned per participant; 278 reported sessions",
        eeg_channels="11",
        sampling_hz="256",
        tasks=("left hand", "right hand"),
        rest_support="Explicit 9-second preparation stage before each 4-second imagery stage",
        file_formats=("MAT", "BIDS"),
        license="CC BY 4.0",
        access_url="https://doi.org/10.6084/m9.figshare.30747806.v6",
        publication_url="https://doi.org/10.1038/s41597-026-07742-x",
        priority_tier=1,
        recommended_role="Primary locked chronic-stroke external validation followed by longitudinal patient adaptation",
        integration_status="Locked raw-MAT adapter and first frozen-v4 external evaluation completed; cohort is now development-observed",
        caveat="The corrected v6 release reports 23,382 trials rather than 57,902; repeated sessions are not independent participants, and preparation is structured rest rather than natural idle.",
    ),
    MotorImageryDataset(
        key="mrcp_eeg_emg_2025",
        name="MRCP Hand Gesture EEG-EMG Dataset",
        cohort="Healthy",
        subjects=40,
        sessions="5 sessions; 10 movements/session",
        eeg_channels="32 EEG + bipolar forearm EMG",
        sampling_hz="128 EEG / 440 EMG",
        tasks=("right-hand fist closure",),
        rest_support="Pre-movement rest within synchronized trials",
        file_formats=("CSV",),
        license="CC BY-NC-SA 4.0",
        access_url="https://data.mendeley.com/datasets/y23s2xg6x4/1",
        publication_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC12969298/",
        priority_tier=1,
        recommended_role="First multimodal adapter: MRCP pretraining, EMG-onset labels, and session-drift validation",
        integration_status="Native CSV synchronization, EMG-onset, and privileged-transfer adapter available",
        caveat="Execution-only and right-hand-only; use EMG as timing supervision, not as proof of EEG imagery decoding.",
        modalities=("EEG", "EMG", "event triggers"),
        movement_ground_truth="Synchronized forearm EMG and visual-interface event triggers",
    ),
    MotorImageryDataset(
        key="way_eeg_gal",
        name="WAY-EEG-GAL Grasp-and-Lift",
        cohort="Healthy",
        subjects=12,
        sessions="10 series; 3,936 recorded trials",
        eeg_channels="32 EEG + 5 EMG",
        sampling_hz="500 EEG/kinematics/force / 4000 EMG",
        tasks=("reach", "precision grasp", "lift", "hold", "replace", "release"),
        rest_support="Return-to-rest phase and 16 behaviorally defined event times",
        file_formats=("MAT",),
        license="CC BY 4.0",
        access_url="https://doi.org/10.1038/sdata.2014.47",
        publication_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC4365902/",
        priority_tier=1,
        recommended_role="Multimodal onset teacher and force/kinematic validation with explicit modality ablation",
        integration_status="Official local release verified; continuous MAT privileged-transfer adapter and EEG-only student benchmark available",
        caveat="Overt right-hand execution in 12 participants; recent LOSO work found weight/surface decoding predominantly EMG-driven.",
        modalities=("EEG", "EMG", "kinematics", "grip/load force"),
        movement_ground_truth="EMG, hand/object kinematics, force, and 16 trial event timestamps",
    ),
    MotorImageryDataset(
        key="nebula_reaching_2025",
        name="NeBULA Standardized Reaching EEG-EMG",
        cohort="Healthy",
        subjects=40,
        sessions="1 session; 3 conditions x 10 repetitions",
        eeg_channels="127 EEG + 11 EMG",
        sampling_hz="1000 EEG / 2000 EMG",
        tasks=("free reaching", "low exoskeleton assistance", "high exoskeleton assistance"),
        rest_support="GO, reach, and end-of-trial markers",
        file_formats=("BIDS", "BrainVision", "CSV", "TSV"),
        license="CC BY 4.0",
        access_url="https://doi.org/10.6084/m9.figshare.27301629",
        publication_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC12092770/",
        priority_tier=1,
        recommended_role="Human-robot artifact, assistance-domain shift, and compensation robustness benchmark",
        integration_status="BIDS EEG/EMG adapter required; large download",
        caveat="Overt right-arm reaching with only 10 repetitions per condition; do not pool its labels with hand imagery.",
        modalities=("EEG", "EMG", "exoskeleton telemetry"),
        movement_ground_truth="GO/reach markers, upper-limb EMG, and exoskeleton endpoint/joint telemetry",
    ),
    MotorImageryDataset(
        key="wristmotion_2026",
        name="EEG-EMG WristMotion Pointing",
        cohort="Healthy",
        subjects=45,
        sessions="320 trials/participant",
        eeg_channels="8 EEG + 4 bipolar EMG",
        sampling_hz="Multirate; verify BIDS sidecars",
        tasks=("three-degree-of-freedom wrist pointing",),
        rest_support="Pre/post baseline and synchronized trial timestamps",
        file_formats=("BIDS", "EDF", "MAT", "TSV"),
        license="Verify current Figshare record",
        access_url="https://doi.org/10.6084/M9.FIGSHARE.29666735.v3",
        publication_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC13287717/",
        priority_tier=1,
        recommended_role="Direction, latency, and continuous kinematic-teacher benchmark",
        integration_status="BIDS/EDF multimodal adapter required",
        caveat="Overt wrist movement in healthy participants; the eight-channel montage limits direct spatial transfer.",
        modalities=("EEG", "EMG", "robot encoders", "kinematics"),
        movement_ground_truth="Synchronized EMG, robotic encoders, event markers, and trajectory metrics",
    ),
    MotorImageryDataset(
        key="prehension_eeg_emg_2026",
        name="High-density EEG-EMG Object Prehension",
        cohort="Healthy",
        subjects=14,
        sessions="3 grip types; dynamic and holding phases",
        eeg_channels="64 EEG + 13 EMG",
        sampling_hz="Verify OSF acquisition metadata",
        tasks=("object prehension", "three grip types", "holding"),
        rest_support="Visual GO and sensorized-object phase markers",
        file_formats=("MAT",),
        license="Verify OSF data record",
        access_url="https://doi.org/10.17605/OSF.IO/RSV4Z",
        publication_url="https://doi.org/10.1038/s41597-026-07242-y",
        priority_tier=2,
        recommended_role="Muscle-synergy and corticomuscular-coherence research benchmark",
        integration_status="OSF multimodal adapter required",
        caveat="Small overt-movement cohort; useful for physiological features, not cross-subject imagery claims.",
        modalities=("EEG", "EMG", "sensorized grasping box"),
        movement_ground_truth="Thirteen-muscle EMG plus object contact and lift-completion sensing",
    ),
    MotorImageryDataset(
        key="stroke_attempt_eeg_emg_2025",
        name="Severe Stroke Attempted-Movement EEG-EMG Cohort",
        cohort="Chronic severe stroke",
        subjects=35,
        sessions="4-6 blocks; 17 trials/block",
        eeg_channels="16 EEG + bilateral upper-limb EMG + EOG",
        sampling_hz="Publication protocol; request raw metadata",
        tasks=("attempted paretic hand opening/closing", "rest"),
        rest_support="Randomized 4-5 second rest, 4 second attempt, and 8-9 second inter-trial intervals",
        file_formats=("Not publicly confirmed",),
        license="Article open access; participant data access not confirmed",
        access_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC12557861/",
        publication_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC12557861/",
        priority_tier=1,
        recommended_role="Clinical protocol template and potential author-request external validation cohort",
        integration_status="Contact authors or locate repository before implementation",
        caveat="Do not assume the participant-level recordings are downloadable; published results are within-patient block validation.",
        modalities=("EEG", "EMG", "EOG"),
        movement_ground_truth="Attempt cue plus bilateral task-related and compensatory-muscle EMG",
    ),
    MotorImageryDataset(
        key="cho2017",
        name="Cho et al. Large-scale Motor Imagery",
        cohort="Healthy",
        subjects=52,
        sessions="1",
        eeg_channels="64 EEG",
        sampling_hz="512",
        tasks=("left hand", "right hand"),
        rest_support="Cue-locked baseline intervals; no independent rest class",
        file_formats=("MAT",),
        license="CC BY 4.0",
        access_url="https://doi.org/10.5524/100295",
        publication_url="https://doi.org/10.1093/gigascience/gix034",
        priority_tier=2,
        recommended_role="Large healthy cross-subject pretraining pool",
        integration_status="MAT adapter required",
        caveat="Single-session design cannot test day-to-day drift; verify subject exclusions used by each benchmark.",
    ),
    MotorImageryDataset(
        key="openbmi",
        name="OpenBMI Motor Imagery",
        cohort="Healthy",
        subjects=54,
        sessions="2 days",
        eeg_channels="62 EEG + 4 EMG",
        sampling_hz="1000",
        tasks=("left hand", "right hand"),
        rest_support="Resting-state recording and within-trial baseline intervals",
        file_formats=("MAT", "BIDS mirror"),
        license="CC BY 4.0 data article; toolbox GPL-3.0",
        access_url="https://doi.org/10.5524/100542",
        publication_url="https://doi.org/10.1093/gigascience/giz002",
        priority_tier=2,
        recommended_role="Healthy cross-subject and cross-session pretraining",
        integration_status="Local NEMAR BIDS adapter and raw-window cache builder available",
        caveat="The local NEMAR zip contains 116 of 216 expected BDF recordings; use only complete subjects until missing signals are retrieved.",
        modalities=("EEG", "EMG"),
        movement_ground_truth="Cue markers; EMG should remain silent during imagery and can audit covert movement",
    ),
    MotorImageryDataset(
        key="bcic_iv_2a",
        name="BCI Competition IV Dataset 2a",
        cohort="Healthy",
        subjects=9,
        sessions="2",
        eeg_channels="22 EEG + 3 EOG",
        sampling_hz="250",
        tasks=("left hand", "right hand", "both feet", "tongue"),
        rest_support="Timed baseline and inter-trial intervals",
        file_formats=("GDF",),
        license="CC BY-ND 4.0",
        access_url="https://bnci-horizon-2020.eu/database/data-sets",
        publication_url="https://www.bbci.de/competition/iv/",
        priority_tier=3,
        recommended_role="Small standard benchmark and format-adapter validation",
        integration_status="GDF adapter required",
        caveat="Small cohort and reduced montage make it a benchmark, not the main route to broader patient generalization.",
    ),
    MotorImageryDataset(
        key="bcic_iv_1",
        name="BCI Competition IV Dataset 1",
        cohort="Healthy",
        subjects=7,
        sessions="Calibration and evaluation",
        eeg_channels="64",
        sampling_hz="1000",
        tasks=("left hand", "right hand", "foot"),
        rest_support="Continuous evaluation recordings include idle periods",
        file_formats=("MAT",),
        license="Competition research terms",
        access_url="https://www.bbci.de/competition/iv/",
        publication_url="https://www.bbci.de/competition/iv/",
        priority_tier=3,
        recommended_role="Asynchronous false-activation and idle-state stress test",
        integration_status="MAT adapter required",
        caveat="Only seven subjects; use primarily to test continuous intent gating rather than pooled accuracy.",
    ),
)


def dataset_catalog(priority_tiers: Iterable[int] | None = None) -> tuple[MotorImageryDataset, ...]:
    """Return the catalog, optionally restricted to selected priority tiers."""
    if priority_tiers is None:
        return DATASET_CATALOG
    selected = set(priority_tiers)
    return tuple(dataset for dataset in DATASET_CATALOG if dataset.priority_tier in selected)


def dataset_by_key(key: str) -> MotorImageryDataset:
    normalized = key.strip().lower()
    for dataset in DATASET_CATALOG:
        if dataset.key == normalized:
            return dataset
    raise KeyError(f"Unknown motor-imagery dataset: {key}")


def catalog_as_dicts() -> list[dict[str, object]]:
    return [dataset.as_dict() for dataset in DATASET_CATALOG]
