#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"

if (( $# < 3 || $# > 4 )); then
  print "Usage:"
  print "  $0 PHYSIONET_ROOT OPENBMI_ROOT LIU_ROOT [V3_CACHE_FOLDER]"
  exit 2
fi

PHYSIONET_ROOT="${1:A}"
OPENBMI_ROOT="${2:A}"
LIU_ROOT="${3:A}"

for source in "$PHYSIONET_ROOT" "$OPENBMI_ROOT" "$LIU_ROOT"; do
  if [[ ! -d "$source" ]]; then
    print "Dataset folder not found: $source"
    exit 1
  fi
done

replace_link() {
  local source="$1"
  local destination="$2"
  if [[ -e "$destination" && ! -L "$destination" ]]; then
    print "Refusing to replace non-link path: $destination"
    exit 1
  fi
  ln -sfn "$source" "$destination"
}

mkdir -p "$PROJECT_DIR/files/eegmmidb"
mkdir -p "$PROJECT_DIR/files/external/openbmi"
mkdir -p "$PROJECT_DIR/data_links"

replace_link "$PHYSIONET_ROOT" "$PROJECT_DIR/files/eegmmidb/1.0.0"
replace_link "$OPENBMI_ROOT" "$PROJECT_DIR/files/external/openbmi/nm000338"
replace_link "$LIU_ROOT" "$PROJECT_DIR/Liu acute-stroke dataset"
replace_link \
  "$PHYSIONET_ROOT" \
  "$PROJECT_DIR/data_links/EEG_BCI_Project_PhysioNet_EEGMMIDB"
replace_link \
  "$OPENBMI_ROOT" \
  "$PROJECT_DIR/data_links/EEG_BCI_Project_OpenBMI"
replace_link \
  "$LIU_ROOT" \
  "$PROJECT_DIR/data_links/EEG_BCI_Project_Liu_Stroke"

if (( $# == 4 )); then
  CACHE_FOLDER="${4:A}"
  if [[ ! -f "$CACHE_FOLDER/translational_windows_v3.npz" ]]; then
    print "Cache folder does not contain translational_windows_v3.npz."
    exit 1
  fi
  mkdir -p "$PROJECT_DIR/artifacts/gui_experiments"
  replace_link \
    "$CACHE_FOLDER" \
    "$PROJECT_DIR/artifacts/gui_experiments/translational_v3"
fi

print "EEG_BCI_Project dataset links updated."

