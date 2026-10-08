"""Machine-readable study, event, split, and endpoint contracts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


CONTRACT_SCHEMA_VERSION = 1
DEVELOPMENT_ROLE = "development"
CONTROLLER_ROLE = "controller_validation"
DEVELOPMENT_OBSERVED_ROLE = "development_observed"
LOCKED_EXTERNAL_ROLE = "locked_external"


@dataclass(frozen=True)
class EventSemanticsContract:
    """Declare what intent and no-control windows mean in one dataset."""

    dataset: str
    cohort: str
    intent_definition: str
    negative_definition: str
    negative_subtype: str
    cue_locked: bool
    continuous_idle_available: bool
    auxiliary_validation: tuple[str, ...]
    deployment_match: str
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationContract:
    """Prevent exploratory cohorts from being reported as untouched tests."""

    primary_endpoint: str
    secondary_endpoint: str
    operating_point: str
    split_roles: tuple[str, ...]
    current_stroke_test_status: str
    locked_external_required: bool
    locked_external_available: bool
    confirmation_rule: str


@dataclass(frozen=True)
class ProspectiveAcquisitionTemplate:
    """Configuration placeholder for a later synchronized acquisition study."""

    enabled: bool = False
    mode: str = "preparation_only"
    warning_seconds: float = 1.0
    instruction_delay_range_seconds: tuple[float, float] = (0.8, 1.8)
    go_delay_range_seconds: tuple[float, float] = (0.8, 1.8)
    imagery_seconds: float = 4.0
    rest_seconds: float = 5.0
    no_go_fraction: float = 0.20
    required_auxiliary_channels: tuple[str, ...] = (
        "vertical_eog",
        "horizontal_eog",
        "left_emg",
        "right_emg",
        "force_or_grip",
        "limb_imu",
        "hardware_event_trigger",
    )
    safety_note: str = (
        "Configuration only. No clinical command or autonomous device output is enabled."
    )


EVENT_CONTRACTS: Mapping[str, EventSemanticsContract] = {
    "physionet": EventSemanticsContract(
        dataset="physionet",
        cohort="healthy",
        intent_definition="T1/T2 cue-locked unilateral fist imagery",
        negative_definition="T0 intervals inside repetitive imagery runs",
        negative_subtype="protocol_rest",
        cue_locked=True,
        continuous_idle_available=False,
        auxiliary_validation=(),
        deployment_match="low",
        caveats=(
            "T0 is not unrestricted idle.",
            "The cue is synchronized but attempted movement onset is not measured.",
            "No EOG, EMG, force, or motion validation is available.",
        ),
    ),
    "openbmi": EventSemanticsContract(
        dataset="openbmi",
        cohort="healthy",
        intent_definition="left/right hand imagery after a visual instruction",
        negative_definition="fixed post-imagery interval from the same trial",
        negative_subtype="post_task_rest",
        cue_locked=True,
        continuous_idle_available=False,
        auxiliary_validation=("emg",),
        deployment_match="low",
        caveats=(
            "Post-task rest is temporally predictable and not unrestricted idle.",
            "EMG channels are retained for validation but excluded from EEG decoding.",
        ),
    ),
    "liu2024": EventSemanticsContract(
        dataset="liu2024",
        cohort="acute_stroke",
        intent_definition="left/right hand-grip imagery during protocol stage 2",
        negative_definition="protocol break/rest during stage 3",
        negative_subtype="protocol_break",
        cue_locked=True,
        continuous_idle_available=False,
        auxiliary_validation=("horizontal_eog", "vertical_eog"),
        deployment_match="moderate",
        caveats=(
            "Rest is a structured trial stage rather than natural clinical idle.",
            "Healthy/stroke comparisons are confounded by acquisition and protocol.",
            "No simultaneous EMG, force, or motion ground truth is distributed.",
        ),
    ),
    "hard_negative": EventSemanticsContract(
        dataset="hard_negative",
        cohort="dataset_specific",
        intent_definition="none",
        negative_definition=(
            "continuous no-control EEG containing ordinary idle and documented artifacts"
        ),
        negative_subtype="continuous_no_control",
        cue_locked=False,
        continuous_idle_available=True,
        auxiliary_validation=("dataset_specific",),
        deployment_match="high",
        caveats=(
            "Recordings must be held out by subject/session.",
            "Artifact labels must not be treated as intent labels.",
        ),
    ),
}


DEFAULT_EVALUATION_CONTRACT = EvaluationContract(
    primary_endpoint=(
        "event sensitivity at a fixed continuous false-activation budget"
    ),
    secondary_endpoint="left/right accuracy conditional on accepted motor intent",
    operating_point=(
        "<=1 false activation/minute, >=60% sensitivity, <=2.5 s median latency"
    ),
    split_roles=(
        DEVELOPMENT_ROLE,
        CONTROLLER_ROLE,
        DEVELOPMENT_OBSERVED_ROLE,
        LOCKED_EXTERNAL_ROLE,
    ),
    current_stroke_test_status=DEVELOPMENT_OBSERVED_ROLE,
    locked_external_required=True,
    locked_external_available=False,
    confirmation_rule=(
        "A claim-bearing result requires a never-inspected subject/session or external "
        "dataset after every preprocessing, feature, model, fusion, and threshold choice."
    ),
)


def event_contract(key: str) -> EventSemanticsContract:
    normalized = key.strip().lower()
    if normalized not in EVENT_CONTRACTS:
        raise KeyError(f"Unknown event-semantics contract: {key}")
    return EVENT_CONTRACTS[normalized]


def validate_split_roles(
    subject_roles: Mapping[str, Sequence[str]],
    *,
    require_locked_external: bool = False,
) -> None:
    """Ensure subjects are not assigned to multiple experimental roles."""
    allowed = set(DEFAULT_EVALUATION_CONTRACT.split_roles)
    seen: dict[str, str] = {}
    for role, subjects in subject_roles.items():
        if role not in allowed:
            raise ValueError(f"Unsupported split role: {role}")
        for subject in subjects:
            identifier = str(subject)
            if identifier in seen:
                raise ValueError(
                    f"Subject {identifier} appears in both {seen[identifier]} and {role}."
                )
            seen[identifier] = role
    if require_locked_external and not subject_roles.get(LOCKED_EXTERNAL_ROLE):
        raise ValueError(
            "A locked external cohort is required for a claim-bearing evaluation."
        )


def write_research_contract(
    output_dir: str | Path,
    *,
    evaluation: EvaluationContract = DEFAULT_EVALUATION_CONTRACT,
    acquisition: ProspectiveAcquisitionTemplate = ProspectiveAcquisitionTemplate(),
) -> tuple[Path, Path]:
    """Write JSON and Markdown contracts beside experiment outputs."""
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "research_contract.json"
    markdown_path = destination / "RESEARCH_CONTRACT.md"
    payload = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "event_semantics": {
            key: asdict(value) for key, value in EVENT_CONTRACTS.items()
        },
        "evaluation": asdict(evaluation),
        "prospective_acquisition": asdict(acquisition),
    }
    json_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="ascii",
    )

    lines = [
        "# Translational Motor-Intent Research Contract",
        "",
        "## Evaluation Status",
        "",
        f"- Current stroke test status: `{evaluation.current_stroke_test_status}`.",
        f"- Locked external test available: `{evaluation.locked_external_available}`.",
        f"- Primary endpoint: {evaluation.primary_endpoint}.",
        f"- Operating point: {evaluation.operating_point}.",
        f"- Confirmation rule: {evaluation.confirmation_rule}",
        "",
        "## Event Semantics",
        "",
        "| Dataset | Intent | Negative state | Subtype | Continuous idle | Match |",
        "|---|---|---|---|---:|---|",
    ]
    for contract in EVENT_CONTRACTS.values():
        lines.append(
            f"| {contract.dataset} | {contract.intent_definition} | "
            f"{contract.negative_definition} | `{contract.negative_subtype}` | "
            f"{'yes' if contract.continuous_idle_available else 'no'} | "
            f"{contract.deployment_match} |"
        )
    lines.extend(
        (
            "",
            "## Prospective Acquisition",
            "",
            f"- Mode: `{acquisition.mode}`.",
            f"- Enabled: `{acquisition.enabled}`.",
            f"- Safety: {acquisition.safety_note}",
            "",
            "This file is a decision and provenance record. It does not make the "
            "retrospective datasets equivalent to a prospective clinical recording.",
        )
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return json_path, markdown_path
