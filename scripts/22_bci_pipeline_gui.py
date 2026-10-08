#!/usr/bin/env python3
"""Desktop research console for EEG upload, audit, experiments, and live capture."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(PROJECT_ROOT / ".numba_cache"))
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(PROJECT_ROOT))
(PROJECT_ROOT / ".matplotlib").mkdir(exist_ok=True)
(PROJECT_ROOT / ".numba_cache").mkdir(exist_ok=True)

import matplotlib.image as mpimg
import mne
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import filedialog, messagebox, ttk

from bci_pipeline.dataset_registry import (
    MotorImageryDataset,
    catalog_as_dicts,
    dataset_catalog,
)
from bci_pipeline.gui_backend import (
    RecordingAudit,
    append_hard_negative_manifest,
    build_decoder_bakeoff_step,
    build_emg_supervised_transfer_steps,
    build_multisource_privileged_ensemble_step,
    build_openbmi_adapter_step,
    build_pipeline_steps,
    build_translational_v3_step,
    build_translational_v4_neural_step,
    build_translational_v5_patient_adaptive_step,
    build_translational_v6_intent_only_step,
    build_translational_v7_auxiliary_direction_step,
    build_v7h_mind_step,
    build_way_eeg_ssl_step,
    build_way_eeg_ssl_seed_stability_step,
    build_way_eeg_gal_transfer_steps,
    inspect_eegmmidb_folder,
    inspect_openbmi_folder,
    load_recording,
    optional_adapter_status,
    summarize_recording,
    audit_recording,
    write_prospective_preparation,
)
from bci_pipeline.mrcp_eeg_emg import discover_synchronized_recordings
from bci_pipeline.mind_dataset import inspect_mind_dataset
from bci_pipeline.way_eeg_gal import discover_way_series, resolve_way_eeg_gal_root
from bci_pipeline.way_eeg_ssl import discover_way_eeg_series
from bci_pipeline.live_sources import (
    BrainFlowSource,
    FileReplaySource,
    LSLSource,
    discover_lsl_streams,
    raw_from_live_chunks,
)
from bci_pipeline.research_contract import ProspectiveAcquisitionTemplate
from bci_pipeline.torch_device import mps_is_available


PALETTE = {
    "ink": "#102a43",
    "blue": "#12355b",
    "teal": "#2f8f83",
    "gold": "#e9a23b",
    "paper": "#f7f4ed",
    "white": "#ffffff",
    "mist": "#e8eef2",
    "muted": "#627d98",
    "danger": "#b23a48",
}


class MotorIntentConsole(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Motor Intent EEG Research Console")
        self.geometry("1440x900")
        self.minsize(1120, 720)
        self.configure(background=PALETTE["paper"])
        self.protocol("WM_DELETE_WINDOW", self._close)

        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.raw: mne.io.BaseRaw | None = None
        self.recording_path: Path | None = None
        self.preview_canvas: FigureCanvasTkAgg | None = None
        self.pipeline_process: subprocess.Popen[str] | None = None
        self.pipeline_stop = threading.Event()
        self.live_source: Any = None
        self.live_chunks: list[np.ndarray] = []
        self.live_sample_count = 0
        self.live_poll_job: str | None = None
        self.catalog_selection: MotorImageryDataset | None = None

        self._configure_styles()
        self._build_header()
        self._build_tabs()
        self._build_status_bar()
        self.after(100, self._drain_events)

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", font=("Avenir Next", 11), background=PALETTE["paper"])
        style.configure("TFrame", background=PALETTE["paper"])
        style.configure(
            "Card.TFrame",
            background=PALETTE["white"],
            relief="flat",
        )
        style.configure(
            "Title.TLabel",
            background=PALETTE["blue"],
            foreground=PALETTE["white"],
            font=("Avenir Next", 24, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=PALETTE["blue"],
            foreground="#d9e6ef",
            font=("Georgia", 11, "italic"),
        )
        style.configure(
            "CardTitle.TLabel",
            background=PALETTE["white"],
            foreground=PALETTE["ink"],
            font=("Avenir Next", 14, "bold"),
        )
        style.configure(
            "CardText.TLabel",
            background=PALETTE["white"],
            foreground=PALETTE["muted"],
        )
        style.configure(
            "Accent.TButton",
            background=PALETTE["teal"],
            foreground=PALETTE["white"],
            font=("Avenir Next", 11, "bold"),
            padding=(14, 8),
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#24736a"), ("disabled", "#a8bbb8")],
        )
        style.configure(
            "Secondary.TButton",
            background=PALETTE["mist"],
            foreground=PALETTE["ink"],
            padding=(12, 7),
        )
        style.map("Secondary.TButton", background=[("active", "#d5e0e7")])
        style.configure(
            "Danger.TButton",
            background=PALETTE["danger"],
            foreground=PALETTE["white"],
            padding=(12, 7),
        )
        style.configure(
            "TNotebook",
            background=PALETTE["paper"],
            borderwidth=0,
        )
        style.configure(
            "TNotebook.Tab",
            background=PALETTE["mist"],
            foreground=PALETTE["ink"],
            padding=(18, 10),
            font=("Avenir Next", 11, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", PALETTE["white"])],
            foreground=[("selected", PALETTE["teal"])],
        )
        style.configure(
            "Treeview",
            rowheight=29,
            background=PALETTE["white"],
            fieldbackground=PALETTE["white"],
            foreground=PALETTE["ink"],
        )
        style.configure(
            "Treeview.Heading",
            background=PALETTE["blue"],
            foreground=PALETTE["white"],
            font=("Avenir Next", 10, "bold"),
        )
        style.map("Treeview.Heading", background=[("active", PALETTE["blue"])])

    def _build_header(self) -> None:
        header = tk.Frame(self, background=PALETTE["blue"], padx=28, pady=18)
        header.pack(fill="x")
        text = tk.Frame(header, background=PALETTE["blue"])
        text.pack(side="left", fill="x", expand=True)
        ttk.Label(
            text,
            text="Motor Intent EEG Research Console",
            style="Title.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            text,
            text="Inspect the signal first. Benchmark the decoder second. Preserve the patient split.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(3, 0))
        adapters = optional_adapter_status()
        chips = tk.Frame(header, background=PALETTE["blue"])
        chips.pack(side="right")
        self._status_chip(chips, "MNE FILES", True).pack(side="left", padx=4)
        self._status_chip(chips, "LSL", adapters["lsl"]).pack(side="left", padx=4)
        self._status_chip(chips, "BRAINFLOW", adapters["brainflow"]).pack(
            side="left", padx=4
        )

    def _status_chip(self, parent: tk.Widget, label: str, available: bool) -> tk.Label:
        color = PALETTE["teal"] if available else PALETTE["muted"]
        text = f"{label}  {'READY' if available else 'OPTIONAL'}"
        return tk.Label(
            parent,
            text=text,
            background=color,
            foreground=PALETTE["white"],
            font=("Avenir Next", 9, "bold"),
            padx=10,
            pady=6,
        )

    def _build_tabs(self) -> None:
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=22, pady=(18, 10))
        self.recording_tab = ttk.Frame(self.notebook)
        self.dataset_tab = ttk.Frame(self.notebook)
        self.live_tab = ttk.Frame(self.notebook)
        self.protocol_tab = ttk.Frame(self.notebook)
        self.library_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.recording_tab, text="Recording Workspace")
        self.notebook.add(self.dataset_tab, text="Dataset Experiments")
        self.notebook.add(self.live_tab, text="Live Device")
        self.notebook.add(self.protocol_tab, text="Study Preparation")
        self.notebook.add(self.library_tab, text="Dataset Library")
        self._build_recording_tab()
        self._build_dataset_tab()
        self._build_live_tab()
        self._build_protocol_tab()
        self._build_library_tab()

    def _card(self, parent: tk.Widget, title: str, subtitle: str = "") -> ttk.Frame:
        card = ttk.Frame(parent, style="Card.TFrame", padding=18)
        ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w")
        if subtitle:
            ttk.Label(
                card,
                text=subtitle,
                style="CardText.TLabel",
                wraplength=580,
                justify="left",
            ).pack(anchor="w", pady=(2, 14))
        return card

    def _labeled_entry(
        self,
        parent: tk.Widget,
        label: str,
        variable: tk.StringVar,
        row: int,
        *,
        width: int = 44,
    ) -> ttk.Entry:
        ttk.Label(
            parent,
            text=label,
            background=PALETTE["white"],
            foreground=PALETTE["ink"],
        ).grid(row=row, column=0, sticky="w", pady=5)
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row, column=1, sticky="ew", padx=(12, 0), pady=5)
        return entry

    def _text_panel(self, parent: tk.Widget, height: int = 12) -> tk.Text:
        panel = tk.Text(
            parent,
            height=height,
            wrap="word",
            borderwidth=0,
            background="#fbfcfd",
            foreground=PALETTE["ink"],
            font=("Menlo", 10),
            padx=12,
            pady=10,
        )
        return panel

    def _build_recording_tab(self) -> None:
        self.recording_tab.columnconfigure(0, weight=2)
        self.recording_tab.columnconfigure(1, weight=3)
        self.recording_tab.rowconfigure(0, weight=1)
        controls = self._card(
            self.recording_tab,
            "Load and verify a recording",
            "Standard EEG files retain their metadata. CSV and NumPy uploads require a sampling rate and voltage unit.",
        )
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 9))

        form = tk.Frame(controls, background=PALETTE["white"])
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        self.recording_file = tk.StringVar()
        self.array_sfreq = tk.StringVar(value="160")
        self.array_unit = tk.StringVar(value="uV")
        self.array_channels = tk.StringVar()
        self.audit_output = tk.StringVar(
            value=str(PROJECT_ROOT / "artifacts" / "gui_recording_audits")
        )
        self.hard_negative_manifest = tk.StringVar(
            value=str(
                PROJECT_ROOT
                / "artifacts"
                / "translational_motor_intent_v3"
                / "hard_negative_manifest.json"
            )
        )
        self.hard_negative_subject = tk.StringVar(value="participant-001")
        self.hard_negative_session = tk.StringVar(value="1")
        self._labeled_entry(form, "Recording", self.recording_file, 0)
        ttk.Button(
            form,
            text="Browse",
            style="Secondary.TButton",
            command=self._browse_recording,
        ).grid(row=0, column=2, padx=(8, 0))
        self._labeled_entry(form, "Array sampling (Hz)", self.array_sfreq, 1)
        ttk.Label(
            form,
            text="Array voltage unit",
            background=PALETTE["white"],
            foreground=PALETTE["ink"],
        ).grid(row=2, column=0, sticky="w", pady=5)
        ttk.Combobox(
            form,
            textvariable=self.array_unit,
            values=("uV", "mV", "V"),
            state="readonly",
            width=10,
        ).grid(row=2, column=1, sticky="w", padx=(12, 0), pady=5)
        self._labeled_entry(form, "Array channel names", self.array_channels, 3)
        self._labeled_entry(form, "Audit output", self.audit_output, 4)
        ttk.Button(
            form,
            text="Browse",
            style="Secondary.TButton",
            command=lambda: self._browse_directory(self.audit_output),
        ).grid(row=4, column=2, padx=(8, 0))
        self._labeled_entry(
            form,
            "No-control manifest",
            self.hard_negative_manifest,
            5,
        )
        self._labeled_entry(
            form,
            "No-control subject key",
            self.hard_negative_subject,
            6,
        )
        self._labeled_entry(
            form,
            "No-control session",
            self.hard_negative_session,
            7,
        )

        actions = tk.Frame(controls, background=PALETTE["white"])
        actions.pack(fill="x", pady=(14, 12))
        self.load_button = ttk.Button(
            actions,
            text="Load Recording",
            style="Accent.TButton",
            command=self._load_recording,
        )
        self.load_button.pack(side="left")
        self.audit_button = ttk.Button(
            actions,
            text="Run Signal Audit",
            style="Secondary.TButton",
            command=self._run_recording_audit,
            state="disabled",
        )
        self.audit_button.pack(side="left", padx=8)
        ttk.Button(
            actions,
            text="Register as No-Control",
            style="Secondary.TButton",
            command=self._register_hard_negative,
        ).pack(side="left")

        ttk.Separator(controls).pack(fill="x", pady=8)
        ttk.Label(
            controls,
            text="Recording summary",
            style="CardTitle.TLabel",
        ).pack(anchor="w", pady=(8, 4))
        self.recording_summary = self._text_panel(controls, height=15)
        self.recording_summary.pack(fill="both", expand=True)
        self.recording_summary.insert(
            "1.0",
            "No recording loaded.\n\nThe signal audit produces raw and processed traces, "
            "gross channel QC, a central-channel PSD, and mu/beta band summaries.",
        )
        self.recording_summary.configure(state="disabled")

        preview = self._card(
            self.recording_tab,
            "Signal audit preview",
            "This view confirms data and preprocessing quality; it does not claim a motor-intent prediction.",
        )
        preview.grid(row=0, column=1, sticky="nsew", padx=(9, 0))
        self.preview_host = tk.Frame(preview, background="#fbfcfd")
        self.preview_host.pack(fill="both", expand=True)
        self.preview_placeholder = tk.Label(
            self.preview_host,
            text="RAW EEG  ->  QC/REPAIR  ->  MRCP + SMR/CSD  ->  AVAILABILITY",
            background="#fbfcfd",
            foreground=PALETTE["muted"],
            font=("Avenir Next", 15, "bold"),
        )
        self.preview_placeholder.pack(expand=True)

    def _build_dataset_tab(self) -> None:
        self.dataset_tab.columnconfigure(0, weight=2)
        self.dataset_tab.columnconfigure(1, weight=3)
        self.dataset_tab.rowconfigure(0, weight=1)
        controls = self._card(
            self.dataset_tab,
            "Run Dataset Experiment",
            "The recommended binary workflow merges left/right imagery into "
            "rest versus intent, then applies domain-specific operating points "
            "and rapid patient calibration. Legacy workflows remain available.",
        )
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 9))
        form = tk.Frame(controls, background=PALETTE["white"])
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        self.dataset_root = tk.StringVar(
            value=str(PROJECT_ROOT / "files" / "eegmmidb" / "1.0.0")
        )
        self.experiment_output = tk.StringVar(
            value=str(PROJECT_ROOT / "artifacts" / "gui_experiments")
        )
        self.openbmi_root = tk.StringVar(
            value=str(PROJECT_ROOT / "files" / "external" / "openbmi" / "nm000338")
        )
        self.liu_root = tk.StringVar(
            value=str(PROJECT_ROOT / "Liu acute-stroke dataset")
        )
        self.mrcp_eeg_emg_root = tk.StringVar(
            value=str(PROJECT_ROOT / "EEG and EMG Dataset for Analyzing Movement-Related")
        )
        self.way_eeg_gal_root = tk.StringVar(
            value=str(PROJECT_ROOT / "WAY-EEG-GAL")
        )
        self.mind_root = tk.StringVar(value=str(PROJECT_ROOT / "MIND"))
        self.experiment_workflow = tk.StringVar(
            value="Translational v7 graph/LDA-enhanced intent (recommended)"
        )
        self.pipeline_stage = tk.StringVar(value="All stages")
        self.rebuild_pipeline = tk.BooleanVar(value=False)
        self.target_intent_sensitivity = tk.StringVar(value="0.70")
        self.temporal_confirmations = tk.StringVar(value="3")
        self.temporal_refractory_seconds = tk.StringVar(value="2.0")
        self.neural_device = tk.StringVar(
            value="mps" if mps_is_available() else "auto"
        )
        ttk.Label(
            form,
            text="Workflow",
            background=PALETTE["white"],
            foreground=PALETTE["ink"],
        ).grid(row=0, column=0, sticky="w", pady=5)
        ttk.Combobox(
            form,
            textvariable=self.experiment_workflow,
            values=(
                "Translational v7 graph/LDA-enhanced intent (recommended)",
                "V7-H hybrid EEG-fNIRS (MIND)",
                "Translational v6 intent-only adaptation",
                "Translational v5 patient-adaptive fusion",
                "Translational v4 neural-integrated",
                "Translational v3 calibrated",
                "EEG decoder model bakeoff",
                "Multi-source privileged EEG ensemble",
                "WAY EEG-only self-supervised transfer",
                "WAY EEG-only five-seed stability",
                "EMG-supervised EEG transfer",
                "WAY-EEG-GAL privileged EEG transfer",
                "Legacy feature atlas",
            ),
            state="readonly",
            width=31,
        ).grid(row=0, column=1, sticky="ew", padx=(12, 0), pady=5)
        self._labeled_entry(form, "PhysioNet root", self.dataset_root, 1)
        ttk.Button(
            form,
            text="Browse",
            style="Secondary.TButton",
            command=lambda: self._browse_directory(self.dataset_root),
        ).grid(row=1, column=2, padx=(8, 0))
        self._labeled_entry(form, "OpenBMI root", self.openbmi_root, 2)
        ttk.Button(
            form,
            text="Browse",
            style="Secondary.TButton",
            command=lambda: self._browse_directory(self.openbmi_root),
        ).grid(row=2, column=2, padx=(8, 0))
        self._labeled_entry(form, "Liu stroke root", self.liu_root, 3)
        ttk.Button(
            form,
            text="Browse",
            style="Secondary.TButton",
            command=lambda: self._browse_directory(self.liu_root),
        ).grid(row=3, column=2, padx=(8, 0))
        self._labeled_entry(form, "MRCP EEG-EMG root", self.mrcp_eeg_emg_root, 4)
        ttk.Button(
            form,
            text="Browse",
            style="Secondary.TButton",
            command=lambda: self._browse_directory(self.mrcp_eeg_emg_root),
        ).grid(row=4, column=2, padx=(8, 0))
        self._labeled_entry(form, "WAY-EEG-GAL root", self.way_eeg_gal_root, 5)
        ttk.Button(
            form,
            text="Browse",
            style="Secondary.TButton",
            command=lambda: self._browse_directory(self.way_eeg_gal_root),
        ).grid(row=5, column=2, padx=(8, 0))
        self._labeled_entry(form, "MIND EEG-fNIRS root", self.mind_root, 6)
        ttk.Button(
            form,
            text="Browse",
            style="Secondary.TButton",
            command=lambda: self._browse_directory(self.mind_root),
        ).grid(row=6, column=2, padx=(8, 0))
        self._labeled_entry(
            form,
            "No-control manifest",
            self.hard_negative_manifest,
            7,
        )
        self._labeled_entry(form, "Experiment output", self.experiment_output, 8)
        ttk.Button(
            form,
            text="Browse",
            style="Secondary.TButton",
            command=lambda: self._browse_directory(self.experiment_output),
        ).grid(row=8, column=2, padx=(8, 0))
        self._labeled_entry(
            form,
            "Target intent sensitivity",
            self.target_intent_sensitivity,
            9,
        )
        self._labeled_entry(
            form,
            "Temporal confirmations",
            self.temporal_confirmations,
            10,
        )
        self._labeled_entry(
            form,
            "Refractory period (s)",
            self.temporal_refractory_seconds,
            11,
        )
        ttk.Label(
            form,
            text="Neural compute",
            background=PALETTE["white"],
            foreground=PALETTE["ink"],
        ).grid(row=12, column=0, sticky="w", pady=5)
        ttk.Combobox(
            form,
            textvariable=self.neural_device,
            values=("mps", "auto", "cpu", "cuda"),
            state="readonly",
            width=31,
        ).grid(row=12, column=1, sticky="ew", padx=(12, 0), pady=5)
        ttk.Label(
            form,
            text="Stage",
            background=PALETTE["white"],
            foreground=PALETTE["ink"],
        ).grid(row=13, column=0, sticky="w", pady=5)
        ttk.Combobox(
            form,
            textvariable=self.pipeline_stage,
            values=(
                "All stages",
                "1. Base feature atlas",
                "2. Extended feature audit",
                "3. Controlled ablation",
            ),
            state="readonly",
            width=31,
        ).grid(row=13, column=1, sticky="ew", padx=(12, 0), pady=5)
        ttk.Checkbutton(
            form,
            text="Rebuild cached features",
            variable=self.rebuild_pipeline,
        ).grid(row=14, column=1, sticky="w", padx=(12, 0), pady=5)

        actions = tk.Frame(controls, background=PALETTE["white"])
        actions.pack(fill="x", pady=(14, 10))
        ttk.Button(
            actions,
            text="Inspect Folder",
            style="Secondary.TButton",
            command=self._inspect_dataset,
        ).pack(side="left")
        self.pipeline_run_button = ttk.Button(
            actions,
            text="Run Experiment",
            style="Accent.TButton",
            command=self._run_pipeline,
        )
        self.pipeline_run_button.pack(side="left", padx=8)
        self.pipeline_stop_button = ttk.Button(
            actions,
            text="Stop",
            style="Danger.TButton",
            command=self._stop_pipeline,
            state="disabled",
        )
        self.pipeline_stop_button.pack(side="left")
        self.dataset_summary = self._text_panel(controls, height=17)
        self.dataset_summary.pack(fill="both", expand=True, pady=(10, 0))
        self.dataset_summary.insert(
            "1.0",
            "The GUI uses the same scripts and fixed partitions as the command-line "
            "workflow. Running only stage 2 or 3 requires outputs from the previous stage.",
        )
        self.dataset_summary.configure(state="disabled")

        log_card = self._card(
            self.dataset_tab,
            "Experiment log",
            "Metrics and output paths are streamed here as each stage runs.",
        )
        log_card.grid(row=0, column=1, sticky="nsew", padx=(9, 0))
        self.pipeline_log = self._text_panel(log_card, height=30)
        self.pipeline_log.pack(fill="both", expand=True)
        self.pipeline_log.configure(state="disabled")

    def _build_live_tab(self) -> None:
        self.live_tab.columnconfigure(0, weight=2)
        self.live_tab.columnconfigure(1, weight=3)
        self.live_tab.rowconfigure(0, weight=1)
        controls = self._card(
            self.live_tab,
            "Connect an EEG source",
            "Acquisition and audit only. File replay works immediately; decoder and device commands remain disabled.",
        )
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 9))
        form = tk.Frame(controls, background=PALETTE["white"])
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        self.live_kind = tk.StringVar(value="File replay")
        self.lsl_name = tk.StringVar()
        self.lsl_type = tk.StringVar(value="EEG")
        self.brainflow_board = tk.StringVar(value="-1")
        self.brainflow_port = tk.StringVar()
        ttk.Label(
            form,
            text="Source",
            background=PALETTE["white"],
            foreground=PALETTE["ink"],
        ).grid(row=0, column=0, sticky="w", pady=5)
        ttk.Combobox(
            form,
            textvariable=self.live_kind,
            values=("File replay", "LSL", "BrainFlow"),
            state="readonly",
        ).grid(row=0, column=1, sticky="ew", padx=(12, 0), pady=5)
        self._labeled_entry(form, "LSL stream name", self.lsl_name, 1)
        self._labeled_entry(form, "LSL stream type", self.lsl_type, 2)
        self._labeled_entry(form, "BrainFlow board ID", self.brainflow_board, 3)
        self._labeled_entry(form, "Serial port", self.brainflow_port, 4)
        actions = tk.Frame(controls, background=PALETTE["white"])
        actions.pack(fill="x", pady=(14, 10))
        ttk.Button(
            actions,
            text="Discover LSL",
            style="Secondary.TButton",
            command=self._discover_lsl,
        ).pack(side="left")
        self.live_start_button = ttk.Button(
            actions,
            text="Start Capture",
            style="Accent.TButton",
            command=self._start_live,
        )
        self.live_start_button.pack(side="left", padx=8)
        self.live_stop_button = ttk.Button(
            actions,
            text="Stop + Audit",
            style="Danger.TButton",
            command=lambda: self._stop_live(run_audit=True),
            state="disabled",
        )
        self.live_stop_button.pack(side="left")
        self.live_status = self._text_panel(controls, height=18)
        self.live_status.pack(fill="both", expand=True, pady=(10, 0))
        adapters = optional_adapter_status()
        self.live_status.insert(
            "1.0",
            "Adapter availability\n"
            f"LSL: {'ready' if adapters['lsl'] else 'optional package not installed'}\n"
            f"BrainFlow: {'ready' if adapters['brainflow'] else 'optional package not installed'}\n\n"
            "A captured stream can be preprocessed and audited. Decoder commands remain "
            "disabled until channel identities, cue markers, and patient calibration are validated.",
        )
        self.live_status.configure(state="disabled")

        monitor = self._card(
            self.live_tab,
            "Capture monitor",
            "The monitor reports acquisition integrity, not a clinical device command.",
        )
        monitor.grid(row=0, column=1, sticky="nsew", padx=(9, 0))
        self.live_monitor = tk.Canvas(
            monitor,
            background="#fbfcfd",
            highlightthickness=0,
        )
        self.live_monitor.pack(fill="both", expand=True)
        self.live_monitor.bind("<Configure>", lambda event: self._draw_live_monitor())

    def _build_protocol_tab(self) -> None:
        self.protocol_tab.columnconfigure(0, weight=2)
        self.protocol_tab.columnconfigure(1, weight=3)
        self.protocol_tab.rowconfigure(0, weight=1)
        controls = self._card(
            self.protocol_tab,
            "Prepare a frozen synchronized study",
            "This exports the locked protocol, model hashes, cue controller, and replay templates. It does not start clinical acquisition or issue a device command.",
        )
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 9))
        form = tk.Frame(controls, background=PALETTE["white"])
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        self.protocol_output = tk.StringVar(
            value=str(PROJECT_ROOT / "artifacts" / "prospective_study_preparation")
        )
        self.protocol_warning = tk.StringVar(value="1.0")
        self.protocol_instruction_delay = tk.StringVar(value="0.8,1.8")
        self.protocol_go_delay = tk.StringVar(value="0.8,1.8")
        self.protocol_imagery = tk.StringVar(value="4.0")
        self.protocol_rest = tk.StringVar(value="5.0")
        self.protocol_no_go = tk.StringVar(value="0.20")
        self._labeled_entry(form, "Preparation output", self.protocol_output, 0)
        ttk.Button(
            form,
            text="Browse",
            style="Secondary.TButton",
            command=lambda: self._browse_directory(self.protocol_output),
        ).grid(row=0, column=2, padx=(8, 0))
        self._labeled_entry(form, "Warning (s)", self.protocol_warning, 1)
        self._labeled_entry(
            form,
            "Instruction delay min,max",
            self.protocol_instruction_delay,
            2,
        )
        self._labeled_entry(
            form,
            "Go delay min,max",
            self.protocol_go_delay,
            3,
        )
        self._labeled_entry(form, "Imagery/attempt (s)", self.protocol_imagery, 4)
        self._labeled_entry(form, "Rest (s)", self.protocol_rest, 5)
        self._labeled_entry(form, "No-go fraction", self.protocol_no_go, 6)
        ttk.Button(
            controls,
            text="Export Frozen Preparation Package",
            style="Accent.TButton",
            command=self._export_protocol_preparation,
        ).pack(anchor="w", pady=(16, 10))

        checklist = self._card(
            self.protocol_tab,
            "Required before command decoding",
            "The GUI is prepared for the later study, but these safeguards must be physically available and validated first.",
        )
        checklist.grid(row=0, column=1, sticky="nsew", padx=(9, 0))
        items = (
            "Full-montage EEG with known units, reference, and electrode positions",
            "Vertical and horizontal EOG",
            "Bilateral EMG over the relevant muscles",
            "Force/grip sensor and limb IMU",
            "Hardware trigger or photodiode timing verification",
            "Randomized warning/instruction/go delays",
            "No-go, cue-only, observation, and natural-idle catch trials",
            "Patient calibration followed by a locked later block/session",
            "Clinical device hazard analysis and supervised stop control",
        )
        panel = self._text_panel(checklist, height=27)
        panel.pack(fill="both", expand=True)
        panel.insert(
            "1.0",
            "PREPARATION ONLY\n\n"
            + "\n".join(f"[ ] {item}" for item in items)
            + "\n\nDecoder output remains disabled in this console.",
        )
        panel.configure(state="disabled")

    def _build_library_tab(self) -> None:
        self.library_tab.columnconfigure(0, weight=3)
        self.library_tab.columnconfigure(1, weight=2)
        self.library_tab.rowconfigure(0, weight=1)
        table_card = self._card(
            self.library_tab,
            "Compatible motor-intent datasets",
            "Tier 1 sources include imagery benchmarks and synchronized movement-teacher datasets. Their labels must remain task-specific.",
        )
        table_card.grid(row=0, column=0, sticky="nsew", padx=(0, 9))
        columns = ("tier", "cohort", "subjects", "channels", "sampling", "role")
        self.library_tree = ttk.Treeview(
            table_card,
            columns=columns,
            show="tree headings",
            selectmode="browse",
        )
        self.library_tree.heading("#0", text="Dataset")
        headings = {
            "tier": "Tier",
            "cohort": "Cohort",
            "subjects": "N",
            "channels": "EEG channels",
            "sampling": "Sampling Hz",
            "role": "Recommended role",
        }
        for column, heading in headings.items():
            self.library_tree.heading(column, text=heading)
        self.library_tree.column("#0", width=260, anchor="w")
        self.library_tree.column("tier", width=55, anchor="center")
        self.library_tree.column("cohort", width=105, anchor="w")
        self.library_tree.column("subjects", width=55, anchor="center")
        self.library_tree.column("channels", width=130, anchor="w")
        self.library_tree.column("sampling", width=120, anchor="w")
        self.library_tree.column("role", width=380, anchor="w")
        self.library_tree.pack(fill="both", expand=True)
        self.library_tree.tag_configure("tier0", background="#e9f4f1")
        self.library_tree.tag_configure("tier1", background="#fff5df")
        self.library_tree.tag_configure("tier2", background="#f5f8fa")
        self.library_tree.tag_configure("tier3", background="#ffffff")
        for dataset in dataset_catalog():
            self.library_tree.insert(
                "",
                "end",
                iid=dataset.key,
                text=dataset.name,
                values=(
                    dataset.priority_tier,
                    dataset.cohort,
                    dataset.subjects,
                    dataset.eeg_channels,
                    dataset.sampling_hz,
                    dataset.recommended_role,
                ),
                tags=(f"tier{dataset.priority_tier}",),
            )
        self.library_tree.bind("<<TreeviewSelect>>", self._select_dataset)

        details = self._card(
            self.library_tab,
            "Integration notes",
            "More data only helps when label semantics and evaluation splits remain valid.",
        )
        details.grid(row=0, column=1, sticky="nsew", padx=(9, 0))
        self.library_details = self._text_panel(details, height=28)
        self.library_details.pack(fill="both", expand=True)
        buttons = tk.Frame(details, background=PALETTE["white"])
        buttons.pack(fill="x", pady=(12, 0))
        ttk.Button(
            buttons,
            text="Open Data Access",
            style="Accent.TButton",
            command=lambda: self._open_catalog_url("access"),
        ).pack(side="left")
        ttk.Button(
            buttons,
            text="Open Publication",
            style="Secondary.TButton",
            command=lambda: self._open_catalog_url("publication"),
        ).pack(side="left", padx=8)
        self.library_tree.selection_set("yang2025")
        self._select_dataset()

    def _build_status_bar(self) -> None:
        bar = tk.Frame(self, background=PALETTE["ink"], padx=20, pady=8)
        bar.pack(fill="x", side="bottom")
        self.status_text = tk.StringVar(
            value="Research use only. No autonomous prosthetic command is issued by this console."
        )
        tk.Label(
            bar,
            textvariable=self.status_text,
            background=PALETTE["ink"],
            foreground="#d9e6ef",
            font=("Avenir Next", 10),
        ).pack(anchor="w")

    def _set_text(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _append_text(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.insert("end", text)
        widget.see("end")
        widget.configure(state="disabled")

    def _browse_recording(self) -> None:
        path = filedialog.askopenfilename(
            title="Select an EEG recording",
            filetypes=(
                ("EEG and arrays", "*.edf *.bdf *.gdf *.fif *.vhdr *.set *.cnt *.csv *.tsv *.npy *.npz"),
                ("All files", "*"),
            ),
        )
        if path:
            self.recording_file.set(path)

    def _browse_directory(self, variable: tk.StringVar) -> None:
        path = filedialog.askdirectory(initialdir=variable.get() or str(PROJECT_ROOT))
        if path:
            variable.set(path)

    def _load_recording(self) -> None:
        path = self.recording_file.get().strip()
        if not path:
            messagebox.showerror("Recording required", "Select an EEG recording first.")
            return
        self.load_button.configure(state="disabled")
        self.audit_button.configure(state="disabled")
        self.status_text.set("Loading recording metadata...")

        def worker() -> None:
            try:
                sfreq = (
                    float(self.array_sfreq.get())
                    if self.array_sfreq.get().strip()
                    else None
                )
                names = [
                    value.strip()
                    for value in self.array_channels.get().split(",")
                    if value.strip()
                ] or None
                raw = load_recording(
                    path,
                    sfreq=sfreq,
                    channel_names=names,
                    unit=self.array_unit.get(),
                )
                summary = summarize_recording(raw, path)
                self.events.put(("recording_loaded", (raw, Path(path), summary)))
            except Exception as error:
                self.events.put(("error", ("Could not load recording", str(error))))

        threading.Thread(target=worker, daemon=True).start()

    def _run_recording_audit(self) -> None:
        if self.raw is None:
            return
        self.audit_button.configure(state="disabled")
        self.status_text.set("Running signal quality and preprocessing audit...")
        raw = self.raw
        source = self.recording_path or Path("in-memory recording")
        output = self.audit_output.get()

        def worker() -> None:
            try:
                result = audit_recording(
                    raw,
                    output,
                    source_path=source,
                )
                self.events.put(("audit_done", result))
            except Exception as error:
                self.events.put(("error", ("Signal audit failed", str(error))))

        threading.Thread(target=worker, daemon=True).start()

    def _register_hard_negative(self) -> None:
        if self.recording_path is None:
            messagebox.showerror(
                "Recording required",
                "Load a saved no-control recording first.",
            )
            return
        try:
            manifest = append_hard_negative_manifest(
                self.hard_negative_manifest.get(),
                self.recording_path,
                subject_key=self.hard_negative_subject.get(),
                session=int(self.hard_negative_session.get()),
            )
        except Exception as error:
            messagebox.showerror("No-control registration failed", str(error))
            return
        self.status_text.set(f"Registered no-control recording: {manifest}")

    def _show_audit(self, audit: RecordingAudit) -> None:
        for child in self.preview_host.winfo_children():
            child.destroy()
        image = mpimg.imread(audit.figure_path)
        figure = Figure(figsize=(9, 6), dpi=100, facecolor="#fbfcfd")
        axis = figure.add_subplot(111)
        axis.imshow(image)
        axis.axis("off")
        figure.tight_layout(pad=0)
        self.preview_canvas = FigureCanvasTkAgg(figure, master=self.preview_host)
        self.preview_canvas.draw()
        self.preview_canvas.get_tk_widget().pack(fill="both", expand=True)
        details = (
            f"{audit.summary.to_text()}\n\n"
            f"Gross bad channels: {audit.quality['bad_channel_count']} "
            f"({100.0 * float(audit.quality['bad_channel_fraction']):.1f}%)\n"
            f"Median channel RMS: {float(audit.quality['median_rms_uv']):.2f} uV\n"
            f"Mu share of 4-35 Hz: {100.0 * audit.band_metrics['mu_fraction_4_35']:.1f}%\n"
            f"Beta share of 4-35 Hz: {100.0 * audit.band_metrics['beta_fraction_4_35']:.1f}%\n\n"
            f"MRCP branch: {audit.branch_metrics['mrcp_low_hz']:.1f}-"
            f"{audit.branch_metrics['mrcp_high_hz']:.1f} Hz\n"
            f"SMR branch: {audit.branch_metrics['smr_low_hz']:.1f}-"
            f"{audit.branch_metrics['smr_high_hz']:.1f} Hz\n"
            f"CSD: {audit.branch_metrics['csd_status']}\n"
            f"Auxiliary channels: {int(audit.branch_metrics['auxiliary_channels'])}\n\n"
            f"Figure: {audit.figure_path}\nReport: {audit.report_path}"
        )
        self._set_text(self.recording_summary, details)

    def _inspect_dataset(self) -> None:
        if self.experiment_workflow.get().startswith("V7-H"):
            try:
                summary = inspect_mind_dataset(self.mind_root.get())
            except Exception as error:
                messagebox.showerror("Dataset inspection failed", str(error))
                return
            text = (
                summary.to_text()
                + "\n\nV7-H will build aligned REST/INTENT features from the "
                "released EEG, HbO, and HbR derivatives. Subject 13 remains "
                "excluded under the dataset's documented fNIRS QC exception."
            )
            self._set_text(self.dataset_summary, text)
            self.status_text.set(
                "MIND V7-H derivatives ready."
                if summary.adapter_ready
                else "MIND derivatives are incomplete."
            )
            return
        if self.experiment_workflow.get().startswith("EEG decoder model bakeoff"):
            required = (
                PROJECT_ROOT.parent
                / "artifacts"
                / "gui_experiments"
                / "translational_v3"
                / "translational_windows_v3.npz",
                PROJECT_ROOT
                / "artifacts"
                / "emg_supervised_transfer"
                / "pure_eeg_target_features.npz",
                PROJECT_ROOT
                / "artifacts"
                / "multisource_privileged_ensemble"
                / "multisource_eeg_intent_ensemble.joblib",
            )
            missing = [str(path) for path in required if not path.is_file()]
            text = (
                "EEG decoder bakeoff inputs are ready.\n\n"
                "The workflow compares target-only Conv1D, EEGNet, "
                "ShallowConvNet, FBCNet, MRCP + FBCNet, WAY SSL, "
                "Riemannian, and calibrated-fusion references across seeds 42-46."
                if not missing
                else "Missing required cache(s):\n" + "\n".join(missing)
            )
            self._set_text(self.dataset_summary, text)
            self.status_text.set(
                "Decoder bakeoff inputs ready."
                if not missing
                else "Decoder bakeoff inputs incomplete."
            )
            return
        if self.experiment_workflow.get().startswith("WAY EEG-only"):
            try:
                root = resolve_way_eeg_gal_root(self.way_eeg_gal_root.get())
                series = discover_way_eeg_series(root)
            except Exception as error:
                messagebox.showerror("Dataset inspection failed", str(error))
                return
            challenge = sum(entry.challenge_series for entry in series)
            text = (
                "WAY EEG-only self-supervised source accepted.\n\n"
                f"Root: {root}\n"
                f"Continuous EEG series: {len(series)}\n"
                f"EEG-only challenge series: {challenge}\n"
                f"Participants: {len({entry.participant for entry in series})}\n\n"
                "Pretraining uses fixed-grid EEG only. EMG, kinematics, force, "
                "events, and WAY class labels are not read."
            )
            self._set_text(self.dataset_summary, text)
            self.status_text.set("WAY EEG-only SSL source accepted.")
            return
        if self.experiment_workflow.get().startswith("Multi-source"):
            caches = (
                PROJECT_ROOT
                / "artifacts"
                / "emg_supervised_transfer"
                / "mrcp_eeg_emg_source_features.npz",
                PROJECT_ROOT
                / "artifacts"
                / "way_eeg_gal_transfer"
                / "way_eeg_gal_source_features.npz",
                PROJECT_ROOT
                / "artifacts"
                / "emg_supervised_transfer"
                / "pure_eeg_target_features.npz",
            )
            missing = [str(path) for path in caches if not path.is_file()]
            if missing:
                messagebox.showerror(
                    "Dataset inspection failed",
                    "Missing standardized cache(s):\n" + "\n".join(missing),
                )
                return
            text = (
                "Multi-source privileged EEG experiment ready.\n\n"
                "Sources: MRCP EEG-EMG and WAY-EEG-GAL\n"
                "Targets: PhysioNet, OpenBMI, and Liu stroke\n"
                "Deployment inputs: 144 EEG features plus dataset identifier\n\n"
                "Source weights, dataset policy, and safety thresholds are selected "
                "only on controller-validation subjects."
            )
            self._set_text(self.dataset_summary, text)
            self.status_text.set("Multi-source standardized caches accepted.")
            return
        if self.experiment_workflow.get().startswith("WAY-EEG-GAL"):
            try:
                root = resolve_way_eeg_gal_root(self.way_eeg_gal_root.get())
                series = discover_way_series(root)
            except Exception as error:
                messagebox.showerror("Dataset inspection failed", str(error))
                return
            participants = {entry.participant for entry in series}
            text = (
                "WAY-EEG-GAL multimodal source accepted.\n\n"
                f"Root: {root}\n"
                f"Continuous paired series: {len(series)}\n"
                f"Participants: {len(participants)}\n\n"
                "Run Experiment to extract EEG windows ending before kinematic hand "
                "start, use post-onset EMG only as training evidence, and evaluate an "
                "EEG-only intent student on the existing PhysioNet/OpenBMI/Liu split."
            )
            self._set_text(self.dataset_summary, text)
            self.status_text.set("WAY-EEG-GAL dataset accepted.")
            return
        if self.experiment_workflow.get().startswith("EMG-supervised"):
            try:
                recordings, qc = discover_synchronized_recordings(
                    self.mrcp_eeg_emg_root.get()
                )
            except Exception as error:
                messagebox.showerror("Dataset inspection failed", str(error))
                return
            text = (
                "MRCP EEG-EMG synchronized source accepted.\n\n"
                f"Root: {qc['dataset_root']}\n"
                f"Paired recordings: {len(recordings)}\n"
                f"Paired participants: {qc['paired_subjects']}\n"
                f"EEG files: {qc['eeg_files']}\n"
                f"EMG files: {qc['emg_files']}\n"
                f"Missing EEG pairs: {len(qc['missing_eeg'])}\n"
                f"Missing EMG pairs: {len(qc['missing_emg'])}\n\n"
                "Run Experiment to build EMG-onset-confirmed source features and "
                "compare an EEG-only transfer student against the target-only baseline."
            )
            self._set_text(self.dataset_summary, text)
            self.status_text.set("MRCP EEG-EMG dataset accepted.")
            return
        root = self.dataset_root.get()
        try:
            summary = inspect_eegmmidb_folder(root)
        except Exception as error:
            messagebox.showerror("Dataset inspection failed", str(error))
            return
        if summary.looks_like_eegmmidb:
            text = (
                summary.to_text()
                + "\n\nEEGMMIDB native workflow ready: feature atlas, extended "
                "audit, and controlled ablation."
            )
            status = "EEGMMIDB dataset accepted."
        else:
            try:
                openbmi = inspect_openbmi_folder(root)
            except Exception:
                openbmi = None
            if openbmi is None:
                text = (
                    summary.to_text()
                    + "\n\nFolder is neither a valid EEGMMIDB root nor a recognized "
                    "OpenBMI nm000338 BIDS root."
                )
                status = "Dataset layout not recognized."
            else:
                text = (
                    openbmi.to_text()
                    + "\n\nOpenBMI workflow ready: BIDS/event harmonization, EMG "
                    "exclusion, paired imagery/rest extraction, common nine-channel "
                    "montage, and 160 Hz RawEpochDataset cache."
                )
                status = (
                    "OpenBMI adapter ready."
                    if openbmi.adapter_ready
                    else "OpenBMI metadata found, but no complete subjects."
                )
        self._set_text(
            self.dataset_summary,
            text,
        )
        self.status_text.set(status)

    def _selected_pipeline_steps(self) -> tuple[Any, ...]:
        if self.experiment_workflow.get().startswith("V7-H"):
            return (
                build_v7h_mind_step(
                    self.mind_root.get(),
                    self.experiment_output.get(),
                    target_specificity=0.93,
                    rebuild=self.rebuild_pipeline.get(),
                ),
            )
        if self.experiment_workflow.get().startswith("Translational v7"):
            manifest = Path(self.hard_negative_manifest.get()).expanduser()
            return (
                build_translational_v7_auxiliary_direction_step(
                    self.dataset_root.get(),
                    self.openbmi_root.get(),
                    self.liu_root.get(),
                    self.experiment_output.get(),
                    hard_negative_manifest=(
                        manifest if manifest.is_file() else None
                    ),
                    target_intent_sensitivity=float(
                        self.target_intent_sensitivity.get()
                    ),
                    temporal_confirmations=int(
                        self.temporal_confirmations.get()
                    ),
                    temporal_refractory_seconds=float(
                        self.temporal_refractory_seconds.get()
                    ),
                    neural_device=self.neural_device.get(),
                    rebuild=self.rebuild_pipeline.get(),
                ),
            )
        if self.experiment_workflow.get().startswith("Translational v6"):
            manifest = Path(self.hard_negative_manifest.get()).expanduser()
            return (
                build_translational_v6_intent_only_step(
                    self.dataset_root.get(),
                    self.openbmi_root.get(),
                    self.liu_root.get(),
                    self.experiment_output.get(),
                    hard_negative_manifest=(
                        manifest if manifest.is_file() else None
                    ),
                    target_intent_sensitivity=float(
                        self.target_intent_sensitivity.get()
                    ),
                    temporal_confirmations=int(
                        self.temporal_confirmations.get()
                    ),
                    temporal_refractory_seconds=float(
                        self.temporal_refractory_seconds.get()
                    ),
                    neural_device=self.neural_device.get(),
                    rebuild=self.rebuild_pipeline.get(),
                ),
            )
        if self.experiment_workflow.get().startswith("Translational v5"):
            manifest = Path(self.hard_negative_manifest.get()).expanduser()
            return (
                build_translational_v5_patient_adaptive_step(
                    self.dataset_root.get(),
                    self.openbmi_root.get(),
                    self.liu_root.get(),
                    self.experiment_output.get(),
                    hard_negative_manifest=(
                        manifest if manifest.is_file() else None
                    ),
                    target_intent_sensitivity=float(
                        self.target_intent_sensitivity.get()
                    ),
                    temporal_confirmations=int(
                        self.temporal_confirmations.get()
                    ),
                    temporal_refractory_seconds=float(
                        self.temporal_refractory_seconds.get()
                    ),
                    neural_device=self.neural_device.get(),
                    rebuild=self.rebuild_pipeline.get(),
                ),
            )
        if self.experiment_workflow.get().startswith("EEG decoder model bakeoff"):
            return (
                build_decoder_bakeoff_step(
                    self.experiment_output.get(),
                    device=self.neural_device.get(),
                    rebuild=self.rebuild_pipeline.get(),
                ),
            )
        if self.experiment_workflow.get().startswith("WAY EEG-only five-seed"):
            return (
                build_way_eeg_ssl_seed_stability_step(
                    self.way_eeg_gal_root.get(),
                    self.experiment_output.get(),
                    device=self.neural_device.get(),
                    rebuild=self.rebuild_pipeline.get(),
                ),
            )
        if self.experiment_workflow.get().startswith("WAY EEG-only"):
            return (
                build_way_eeg_ssl_step(
                    self.way_eeg_gal_root.get(),
                    self.experiment_output.get(),
                    rebuild=self.rebuild_pipeline.get(),
                    device=self.neural_device.get(),
                ),
            )
        if self.experiment_workflow.get().startswith("Multi-source"):
            return (
                build_multisource_privileged_ensemble_step(
                    self.experiment_output.get()
                ),
            )
        if self.experiment_workflow.get().startswith("WAY-EEG-GAL"):
            return build_way_eeg_gal_transfer_steps(
                self.way_eeg_gal_root.get(),
                self.experiment_output.get(),
                rebuild=self.rebuild_pipeline.get(),
            )
        if self.experiment_workflow.get().startswith("EMG-supervised"):
            return build_emg_supervised_transfer_steps(
                self.mrcp_eeg_emg_root.get(),
                self.experiment_output.get(),
                rebuild=self.rebuild_pipeline.get(),
            )
        if self.experiment_workflow.get().startswith("Translational v4"):
            manifest = Path(self.hard_negative_manifest.get()).expanduser()
            return (
                build_translational_v4_neural_step(
                    self.dataset_root.get(),
                    self.openbmi_root.get(),
                    self.liu_root.get(),
                    self.experiment_output.get(),
                    hard_negative_manifest=(
                        manifest if manifest.is_file() else None
                    ),
                    target_intent_sensitivity=float(
                        self.target_intent_sensitivity.get()
                    ),
                    temporal_confirmations=int(
                        self.temporal_confirmations.get()
                    ),
                    temporal_refractory_seconds=float(
                        self.temporal_refractory_seconds.get()
                    ),
                    neural_device=self.neural_device.get(),
                    rebuild=self.rebuild_pipeline.get(),
                ),
            )
        if self.experiment_workflow.get().startswith("Translational v3"):
            manifest = Path(self.hard_negative_manifest.get()).expanduser()
            return (
                build_translational_v3_step(
                    self.dataset_root.get(),
                    self.openbmi_root.get(),
                    self.liu_root.get(),
                    self.experiment_output.get(),
                    hard_negative_manifest=(
                        manifest if manifest.is_file() else None
                    ),
                    target_intent_sensitivity=float(
                        self.target_intent_sensitivity.get()
                    ),
                    temporal_confirmations=int(
                        self.temporal_confirmations.get()
                    ),
                    temporal_refractory_seconds=float(
                        self.temporal_refractory_seconds.get()
                    ),
                    rebuild=self.rebuild_pipeline.get(),
                ),
            )
        steps = build_pipeline_steps(
            self.dataset_root.get(),
            self.experiment_output.get(),
            rebuild=self.rebuild_pipeline.get(),
        )
        selected = self.pipeline_stage.get()
        if selected == "1. Base feature atlas":
            return (steps[0],)
        if selected == "2. Extended feature audit":
            return (steps[1],)
        if selected == "3. Controlled ablation":
            return (steps[2],)
        return steps

    def _run_pipeline(self) -> None:
        try:
            if self.experiment_workflow.get().startswith("V7-H"):
                summary = inspect_mind_dataset(self.mind_root.get())
                if not summary.adapter_ready:
                    raise ValueError(
                        "MIND requires paired EEG motor/rest and HbO/HbR derivatives."
                    )
                steps = self._selected_pipeline_steps()
                dataset_path = (
                    f"MIND={summary.root}; paired subjects="
                    f"{len(summary.paired_subjects)}; modality=EEG + HbO + HbR"
                )
            elif self.experiment_workflow.get().startswith(
                ("Translational v7", "Translational v6", "Translational v5")
            ):
                physionet = inspect_eegmmidb_folder(self.dataset_root.get())
                openbmi = inspect_openbmi_folder(self.openbmi_root.get())
                if not physionet.looks_like_eegmmidb:
                    raise ValueError("PhysioNet root is incomplete.")
                if not openbmi.adapter_ready:
                    raise ValueError("OpenBMI root has no complete subjects.")
                if not Path(self.liu_root.get()).expanduser().is_dir():
                    raise ValueError("Liu stroke root is missing.")
                bakeoff = (
                    Path(self.experiment_output.get()).expanduser().resolve()
                    / "eeg_decoder_model_bakeoff"
                )
                missing_models = [
                    str(bakeoff / "seeds" / f"seed_{seed}" / f"{architecture}.pt")
                    for seed in (42, 43, 44, 45, 46)
                    for architecture in ("mrcp_fbcnet", "shallow_convnet")
                    if not (
                        bakeoff / "seeds" / f"seed_{seed}" / f"{architecture}.pt"
                    ).is_file()
                ]
                if missing_models:
                    raise ValueError(
                        "Run the GPU decoder model bakeoff first; missing model(s): "
                        + ", ".join(missing_models)
                    )
                steps = self._selected_pipeline_steps()
                dataset_path = (
                    f"PhysioNet={physionet.path}; OpenBMI={openbmi.path}; "
                    f"Liu={Path(self.liu_root.get()).expanduser().resolve()}; "
                    "neural=ShallowConvNet + MRCP/FBCNet seeds 42-46"
                )
            elif self.experiment_workflow.get().startswith("EEG decoder model bakeoff"):
                required = (
                    PROJECT_ROOT.parent
                    / "artifacts"
                    / "gui_experiments"
                    / "translational_v3"
                    / "translational_windows_v3.npz",
                    PROJECT_ROOT
                    / "artifacts"
                    / "emg_supervised_transfer"
                    / "pure_eeg_target_features.npz",
                    PROJECT_ROOT
                    / "artifacts"
                    / "multisource_privileged_ensemble"
                    / "multisource_eeg_intent_ensemble.joblib",
                )
                missing = [str(path) for path in required if not path.is_file()]
                if missing:
                    raise ValueError(
                        "EEG decoder bakeoff inputs are incomplete: "
                        + ", ".join(missing)
                    )
                steps = self._selected_pipeline_steps()
                dataset_path = "PhysioNet/OpenBMI/Liu target EEG caches"
            elif self.experiment_workflow.get().startswith("WAY EEG-only"):
                root = resolve_way_eeg_gal_root(self.way_eeg_gal_root.get())
                series = discover_way_eeg_series(root)
                required = (
                    PROJECT_ROOT.parent
                    / "artifacts"
                    / "gui_experiments"
                    / "translational_v3"
                    / "translational_windows_v3.npz",
                    PROJECT_ROOT
                    / "artifacts"
                    / "emg_supervised_transfer"
                    / "pure_eeg_target_features.npz",
                    PROJECT_ROOT
                    / "artifacts"
                    / "multisource_privileged_ensemble"
                    / "multisource_eeg_intent_ensemble.joblib",
                )
                missing = [str(path) for path in required if not path.is_file()]
                if not series or missing:
                    raise ValueError(
                        "WAY EEG-only SSL inputs are incomplete: "
                        + ", ".join(missing or ["no continuous WAY EEG series"])
                    )
                steps = self._selected_pipeline_steps()
                dataset_path = (
                    f"WAY unlabeled EEG={root}; target=PhysioNet/OpenBMI/Liu raw branches"
                )
            elif self.experiment_workflow.get().startswith("Multi-source"):
                caches = (
                    PROJECT_ROOT
                    / "artifacts"
                    / "emg_supervised_transfer"
                    / "mrcp_eeg_emg_source_features.npz",
                    PROJECT_ROOT
                    / "artifacts"
                    / "way_eeg_gal_transfer"
                    / "way_eeg_gal_source_features.npz",
                    PROJECT_ROOT
                    / "artifacts"
                    / "emg_supervised_transfer"
                    / "pure_eeg_target_features.npz",
                )
                missing = [str(path) for path in caches if not path.is_file()]
                if missing:
                    raise ValueError(
                        "Missing standardized multi-source cache(s): "
                        + ", ".join(missing)
                    )
                steps = self._selected_pipeline_steps()
                dataset_path = "MRCP + WAY privileged sources; PhysioNet/OpenBMI/Liu EEG targets"
            elif self.experiment_workflow.get().startswith("WAY-EEG-GAL"):
                root = resolve_way_eeg_gal_root(self.way_eeg_gal_root.get())
                series = discover_way_series(root)
                if not series:
                    raise ValueError("No continuous WAY-EEG-GAL series were found.")
                target_cache = (
                    PROJECT_ROOT
                    / "artifacts"
                    / "emg_supervised_transfer"
                    / "pure_eeg_target_features.npz"
                )
                if not target_cache.is_file():
                    raise ValueError(
                        "The shared pure-EEG target cache is missing. Run the "
                        "EMG-supervised EEG transfer workflow once first."
                    )
                steps = self._selected_pipeline_steps()
                dataset_path = (
                    f"WAY-EEG-GAL={root}; paired series={len(series)}; "
                    "target=existing PhysioNet/OpenBMI/Liu cache"
                )
            elif self.experiment_workflow.get().startswith("EMG-supervised"):
                recordings, qc = discover_synchronized_recordings(
                    self.mrcp_eeg_emg_root.get()
                )
                if not recordings:
                    raise ValueError("No synchronized EEG-EMG recordings were found.")
                steps = self._selected_pipeline_steps()
                dataset_path = (
                    f"MRCP EEG-EMG={qc['dataset_root']}; "
                    "target=existing PhysioNet/OpenBMI/Liu cache"
                )
            elif self.experiment_workflow.get().startswith("Translational v4"):
                physionet = inspect_eegmmidb_folder(self.dataset_root.get())
                openbmi = inspect_openbmi_folder(self.openbmi_root.get())
                if not physionet.looks_like_eegmmidb:
                    raise ValueError("PhysioNet root is incomplete.")
                if not openbmi.adapter_ready:
                    raise ValueError("OpenBMI root has no complete subjects.")
                if not Path(self.liu_root.get()).expanduser().is_dir():
                    raise ValueError("Liu stroke root is missing.")
                bakeoff = PROJECT_ROOT / "artifacts" / "eeg_decoder_model_bakeoff"
                missing_models = [
                    str(bakeoff / "seeds" / f"seed_{seed}" / "mrcp_fbcnet.pt")
                    for seed in (42, 43, 44, 45, 46)
                    if not (
                        bakeoff / "seeds" / f"seed_{seed}" / "mrcp_fbcnet.pt"
                    ).is_file()
                ]
                if missing_models:
                    raise ValueError(
                        "Run the EEG decoder model bakeoff first; missing model(s): "
                        + ", ".join(missing_models)
                    )
                steps = self._selected_pipeline_steps()
                dataset_path = (
                    f"PhysioNet={physionet.path}; OpenBMI={openbmi.path}; "
                    f"Liu={Path(self.liu_root.get()).expanduser().resolve()}; "
                    "neural=MRCP + FBCNet seeds 42-46"
                )
            elif self.experiment_workflow.get().startswith("Translational v3"):
                physionet = inspect_eegmmidb_folder(self.dataset_root.get())
                openbmi = inspect_openbmi_folder(self.openbmi_root.get())
                if not physionet.looks_like_eegmmidb:
                    raise ValueError("PhysioNet root is incomplete.")
                if not openbmi.adapter_ready:
                    raise ValueError("OpenBMI root has no complete subjects.")
                if not Path(self.liu_root.get()).expanduser().is_dir():
                    raise ValueError("Liu stroke root is missing.")
                steps = self._selected_pipeline_steps()
                dataset_path = (
                    f"PhysioNet={physionet.path}; OpenBMI={openbmi.path}; "
                    f"Liu={Path(self.liu_root.get()).expanduser().resolve()}"
                )
            else:
                folder = inspect_eegmmidb_folder(self.dataset_root.get())
                if folder.looks_like_eegmmidb:
                    steps = self._selected_pipeline_steps()
                    dataset_path = folder.path
                else:
                    openbmi = inspect_openbmi_folder(self.dataset_root.get())
                    if not openbmi.adapter_ready:
                        raise ValueError(
                            "The OpenBMI root has no subjects with all four BDF recordings."
                        )
                    steps = (
                        build_openbmi_adapter_step(
                            self.dataset_root.get(),
                            self.experiment_output.get(),
                            rebuild=self.rebuild_pipeline.get(),
                        ),
                    )
                    dataset_path = openbmi.path
        except Exception as error:
            messagebox.showerror("Cannot start experiment", str(error))
            return
        self.pipeline_stop.clear()
        self.pipeline_run_button.configure(state="disabled")
        self.pipeline_stop_button.configure(state="normal")
        self._set_text(
            self.pipeline_log,
            "=== Motor Intent Research Console ===\n"
            f"Dataset: {dataset_path}\n"
            f"Output: {Path(self.experiment_output.get()).expanduser().resolve()}\n"
            f"Stages: {', '.join(step.name for step in steps)}\n\n",
        )
        self.status_text.set("Experiment running...")

        def worker() -> None:
            for step in steps:
                if self.pipeline_stop.is_set():
                    break
                self.events.put(("pipeline_log", f"\n[START] {step.name}\n"))
                try:
                    self.pipeline_process = subprocess.Popen(
                        step.command,
                        cwd=PROJECT_ROOT,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        env={**os.environ, "PYTHONUNBUFFERED": "1"},
                    )
                    assert self.pipeline_process.stdout is not None
                    for line in self.pipeline_process.stdout:
                        self.events.put(("pipeline_log", line))
                    return_code = self.pipeline_process.wait()
                except Exception as error:
                    self.events.put(("pipeline_done", (False, str(error))))
                    return
                finally:
                    self.pipeline_process = None
                if return_code != 0:
                    self.events.put(
                        (
                            "pipeline_done",
                            (False, f"{step.name} exited with code {return_code}."),
                        )
                    )
                    return
                self.events.put(
                    (
                        "pipeline_log",
                        f"[COMPLETE] {step.name}\nOutput: {step.expected_output}\n",
                    )
                )
            stopped = self.pipeline_stop.is_set()
            self.events.put(
                (
                    "pipeline_done",
                    (
                        not stopped,
                        "Experiment stopped by user."
                        if stopped
                        else "Selected experiment stages completed.",
                    ),
                )
            )

        threading.Thread(target=worker, daemon=True).start()

    def _export_protocol_preparation(self) -> None:
        try:
            instruction = tuple(
                float(value.strip())
                for value in self.protocol_instruction_delay.get().split(",")
            )
            go = tuple(
                float(value.strip())
                for value in self.protocol_go_delay.get().split(",")
            )
            if len(instruction) != 2 or len(go) != 2:
                raise ValueError("Delay ranges require exactly two values.")
            template = ProspectiveAcquisitionTemplate(
                enabled=False,
                mode="preparation_only",
                warning_seconds=float(self.protocol_warning.get()),
                instruction_delay_range_seconds=instruction,
                go_delay_range_seconds=go,
                imagery_seconds=float(self.protocol_imagery.get()),
                rest_seconds=float(self.protocol_rest.get()),
                no_go_fraction=float(self.protocol_no_go.get()),
            )
            json_path, markdown_path = write_prospective_preparation(
                self.protocol_output.get(),
                template,
            )
        except Exception as error:
            messagebox.showerror("Protocol export failed", str(error))
            return
        self.status_text.set(
            "Frozen prospective package exported: "
            f"{json_path.name}, {markdown_path.name}, controller protocol and model hashes"
        )

    def _stop_pipeline(self) -> None:
        self.pipeline_stop.set()
        process = self.pipeline_process
        if process is not None and process.poll() is None:
            process.terminate()
        self.status_text.set("Stopping experiment process...")

    def _discover_lsl(self) -> None:
        self.status_text.set("Discovering LSL streams...")

        def worker() -> None:
            try:
                streams = discover_lsl_streams()
                self.events.put(("lsl_discovered", streams))
            except Exception as error:
                self.events.put(("error", ("LSL discovery failed", str(error))))

        threading.Thread(target=worker, daemon=True).start()

    def _start_live(self) -> None:
        if self.live_source is not None:
            return
        kind = self.live_kind.get()
        if kind == "File replay" and self.raw is None:
            messagebox.showerror(
                "Recording required",
                "Load a recording in the Recording Workspace before starting file replay.",
            )
            return
        self.live_start_button.configure(state="disabled")
        self.status_text.set(f"Connecting {kind} source...")

        def worker() -> None:
            try:
                if kind == "File replay":
                    source = FileReplaySource(self.raw, loop=False)
                elif kind == "LSL":
                    source = LSLSource.connect(
                        stream_name=self.lsl_name.get().strip(),
                        stream_type=self.lsl_type.get().strip(),
                    )
                else:
                    source = BrainFlowSource.connect(
                        int(self.brainflow_board.get()),
                        serial_port=self.brainflow_port.get().strip(),
                    )
                self.events.put(("live_connected", (kind, source)))
            except Exception as error:
                self.events.put(("error", (f"{kind} connection failed", str(error))))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_live(self) -> None:
        if self.live_source is None:
            return
        try:
            chunk = self.live_source.pull_chunk(timeout=0.0)
            if chunk.data.shape[1]:
                self.live_chunks.append(chunk.data)
                self.live_sample_count += chunk.data.shape[1]
                maximum = int(round(self.live_source.sampling_hz * 600.0))
                while self.live_chunks and self.live_sample_count > maximum:
                    removed = self.live_chunks.pop(0)
                    self.live_sample_count -= removed.shape[1]
                self._draw_live_monitor(chunk.data)
            if getattr(self.live_source, "finished", False):
                self.status_text.set("File replay reached the end of the recording.")
            else:
                self.live_poll_job = self.after(120, self._poll_live)
        except Exception as error:
            self._stop_live(run_audit=False)
            messagebox.showerror("Live capture failed", str(error))

    def _draw_live_monitor(self, latest: np.ndarray | None = None) -> None:
        canvas = self.live_monitor
        canvas.delete("all")
        width = max(canvas.winfo_width(), 400)
        height = max(canvas.winfo_height(), 300)
        canvas.create_rectangle(0, 0, width, height, fill="#fbfcfd", outline="")
        if self.live_source is None:
            canvas.create_text(
                width / 2,
                height / 2,
                text="NO ACTIVE EEG SOURCE",
                fill=PALETTE["muted"],
                font=("Avenir Next", 16, "bold"),
            )
            return
        duration = self.live_sample_count / max(self.live_source.sampling_hz, 1e-9)
        rms = float(np.sqrt(np.mean(latest**2)) * 1e6) if latest is not None else 0.0
        canvas.create_text(
            32,
            38,
            anchor="w",
            text="CAPTURE ACTIVE",
            fill=PALETTE["teal"],
            font=("Avenir Next", 16, "bold"),
        )
        metrics = (
            f"{len(self.live_source.channel_names)} EEG channels\n"
            f"{self.live_source.sampling_hz:.1f} Hz\n"
            f"{self.live_sample_count:,} buffered samples\n"
            f"{duration:.1f} seconds retained\n"
            f"Latest RMS: {rms:.2f} uV"
        )
        canvas.create_text(
            32,
            78,
            anchor="nw",
            text=metrics,
            fill=PALETTE["ink"],
            font=("Menlo", 13),
            spacing2=7,
        )
        if latest is not None and latest.shape[1] > 1:
            signal = latest[0] - np.mean(latest[0])
            maximum = max(float(np.max(np.abs(signal))), 1e-9)
            x = np.linspace(32, width - 32, signal.size)
            y = height * 0.76 - signal / maximum * height * 0.13
            points = [coordinate for pair in zip(x, y) for coordinate in pair]
            canvas.create_line(*points, fill=PALETTE["blue"], width=2)
            canvas.create_text(
                32,
                height * 0.58,
                anchor="w",
                text=f"Live preview: {self.live_source.channel_names[0]}",
                fill=PALETTE["muted"],
                font=("Avenir Next", 10, "bold"),
            )

    def _stop_live(self, *, run_audit: bool) -> None:
        if self.live_poll_job is not None:
            self.after_cancel(self.live_poll_job)
            self.live_poll_job = None
        source = self.live_source
        self.live_source = None
        if source is not None:
            try:
                source.close()
            except Exception:
                pass
        self.live_start_button.configure(state="normal")
        self.live_stop_button.configure(state="disabled")
        self._draw_live_monitor()
        if not run_audit or not self.live_chunks or source is None:
            self.live_chunks = []
            self.live_sample_count = 0
            self.status_text.set("Live capture stopped.")
            return
        try:
            captured = raw_from_live_chunks(
                self.live_chunks,
                source.sampling_hz,
                source.channel_names,
            )
        except Exception as error:
            messagebox.showerror("Capture conversion failed", str(error))
            return
        self.live_chunks = []
        self.live_sample_count = 0
        self.raw = captured
        self.recording_path = Path("live_capture")
        self.recording_file.set("live_capture")
        self.audit_button.configure(state="normal")
        self.notebook.select(self.recording_tab)
        self._set_text(
            self.recording_summary,
            summarize_recording(captured, "live_capture").to_text()
            + "\n\nLive capture loaded. Running its signal audit now.",
        )
        self._run_recording_audit()

    def _select_dataset(self, event: Any = None) -> None:
        del event
        selection = self.library_tree.selection()
        if not selection:
            return
        key = selection[0]
        self.catalog_selection = next(
            dataset for dataset in dataset_catalog() if dataset.key == key
        )
        dataset = self.catalog_selection
        text = (
            f"{dataset.name}\n\n"
            f"Priority tier: {dataset.priority_tier}\n"
            f"Cohort: {dataset.cohort}\n"
            f"Subjects: {dataset.subjects}\n"
            f"Sessions: {dataset.sessions}\n"
            f"EEG channels: {dataset.eeg_channels}\n"
            f"Modalities: {', '.join(dataset.modalities)}\n"
            f"Sampling: {dataset.sampling_hz} Hz\n"
            f"Tasks: {', '.join(dataset.tasks)}\n"
            f"Rest support: {dataset.rest_support}\n"
            f"Formats: {', '.join(dataset.file_formats)}\n"
            f"License: {dataset.license}\n\n"
            f"Movement ground truth\n{dataset.movement_ground_truth}\n\n"
            f"Recommended role\n{dataset.recommended_role}\n\n"
            f"Integration status\n{dataset.integration_status}\n\n"
            f"Important caveat\n{dataset.caveat}"
        )
        self._set_text(self.library_details, text)

    def _open_catalog_url(self, kind: str) -> None:
        if self.catalog_selection is None:
            return
        url = (
            self.catalog_selection.access_url
            if kind == "access"
            else self.catalog_selection.publication_url
        )
        webbrowser.open(url)

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "recording_loaded":
                    raw, path, summary = payload
                    self.raw = raw
                    self.recording_path = path
                    self._set_text(self.recording_summary, summary.to_text())
                    self.load_button.configure(state="normal")
                    self.audit_button.configure(state="normal")
                    self.status_text.set("Recording loaded. Signal audit is ready.")
                elif kind == "audit_done":
                    self._show_audit(payload)
                    self.audit_button.configure(state="normal")
                    self.status_text.set("Signal audit completed.")
                elif kind == "pipeline_log":
                    self._append_text(self.pipeline_log, payload)
                elif kind == "pipeline_done":
                    success, message = payload
                    self.pipeline_run_button.configure(state="normal")
                    self.pipeline_stop_button.configure(state="disabled")
                    self._append_text(self.pipeline_log, f"\n{message}\n")
                    self.status_text.set(message)
                    if not success and not self.pipeline_stop.is_set():
                        messagebox.showerror("Experiment failed", message)
                elif kind == "lsl_discovered":
                    streams = payload
                    text = "No LSL streams found."
                    if streams:
                        text = "\n".join(
                            f"{stream.name} | {stream.stream_type} | "
                            f"{stream.channels} ch | {stream.sampling_hz:g} Hz"
                            for stream in streams
                        )
                        self.lsl_name.set(streams[0].name)
                        self.lsl_type.set(streams[0].stream_type)
                    self._set_text(self.live_status, text)
                    self.status_text.set(f"LSL discovery found {len(streams)} stream(s).")
                elif kind == "live_connected":
                    source_name, source = payload
                    self.live_source = source
                    self.live_chunks = []
                    self.live_sample_count = 0
                    self.live_stop_button.configure(state="normal")
                    self.status_text.set(f"{source_name} capture active.")
                    self._set_text(
                        self.live_status,
                        f"Connected: {source_name}\n"
                        f"Channels: {len(source.channel_names)}\n"
                        f"Sampling: {source.sampling_hz:.2f} Hz\n\n"
                        "Up to ten minutes are retained in memory for a post-capture audit.",
                    )
                    self._draw_live_monitor()
                    self._poll_live()
                elif kind == "error":
                    title, message = payload
                    self.load_button.configure(state="normal")
                    self.audit_button.configure(
                        state="normal" if self.raw is not None else "disabled"
                    )
                    self.live_start_button.configure(state="normal")
                    self.pipeline_run_button.configure(state="normal")
                    self.pipeline_stop_button.configure(state="disabled")
                    self.status_text.set(message)
                    messagebox.showerror(title, message)
        except queue.Empty:
            pass
        self.after(100, self._drain_events)

    def _close(self) -> None:
        self._stop_pipeline()
        self._stop_live(run_audit=False)
        self.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print backend readiness without opening a desktop window.",
    )
    parser.add_argument(
        "--export-catalog",
        type=Path,
        help="Write the curated external dataset catalog to JSON and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.export_catalog:
        args.export_catalog.parent.mkdir(parents=True, exist_ok=True)
        args.export_catalog.write_text(
            json.dumps(catalog_as_dicts(), indent=2),
            encoding="ascii",
        )
        print(f"Saved dataset catalog: {args.export_catalog.resolve()}")
        return
    if args.check:
        print("Motor Intent EEG Research Console")
        print(f"Project: {PROJECT_ROOT}")
        print(f"Datasets cataloged: {len(dataset_catalog())}")
        for adapter, available in optional_adapter_status().items():
            print(f"{adapter}: {'ready' if available else 'optional dependency missing'}")
        return
    app = MotorIntentConsole()
    app.mainloop()


if __name__ == "__main__":
    main()
