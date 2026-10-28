"""Global configuration for the MI-EEG synthesis filtering pipeline.

Central place for paths, acquisition parameters, channel selection, and the
BCI Competition IV 2a event mapping used across the pipeline.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent          # C:\BMI
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"                                 # extracted GDF files
SYNTHETIC_DIR = DATA_DIR / "synthetic"                     # synthetic data (placeholder)
ZIP_PATH = ROOT_DIR / "BCICIV_2a_gdf.zip"                  # source archive

# ---------------------------------------------------------------------------
# Acquisition / preprocessing parameters
# ---------------------------------------------------------------------------
FS = 250          # target sampling rate after (optional) down-sampling [Hz]
TMIN = -1.0       # epoch start relative to cue onset [s] (includes pre-cue baseline)
TMAX = 3.5        # epoch end relative to cue onset [s]
FMIN = 8.0        # band-pass lower edge [Hz] (mu band)
FMAX = 30.0       # band-pass upper edge [Hz] (beta band)

# ERD analysis windows (seconds, relative to cue onset).
BASELINE_TMIN = -1.0   # baseline window start
BASELINE_TMAX = 0.0    # baseline window end
TASK_TMIN = 0.5        # MI task window start
TASK_TMAX = 3.5        # MI task window end

# ERD frequency bands (Hz).
MU_BAND = (8.0, 13.0)    # mu rhythm
BETA_BAND = (13.0, 30.0)  # beta rhythm

# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------
# Sensorimotor channels of interest (C3, Cz, C4 over the hand areas).
SENSORIMOTOR_CHANNELS = ["C3", "Cz", "C4"]

# Full 22-channel EEG montage of BCI Competition IV 2a (in file order).
EEG_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]

# EOG channels as stored in the GDF.
EOG_CHANNELS = ["EOG-left", "EOG-central", "EOG-right"]

# Full montage in GDF file order: 22 EEG channels followed by the 3 EOG channels.
BCICIV_2A_CHANNELS = EEG_CHANNELS + EOG_CHANNELS

# ---------------------------------------------------------------------------
# Source reconstruction (eLORETA)
# ---------------------------------------------------------------------------
FSAVERAGE_SUBJECT = "fsaverage"      # template MRI/surface subject
SOURCE_SPACING = "ico4"              # surface source-space density (~5124 vertices total)
MONTAGE_NAME = "colin27_1020"       # template montage to align the 10-20 electrodes
SNR = 3.0                            # assumed SNR (lambda2 = 1 / SNR**2)
N_ROI_VERTICES = 100                 # target number of sensory-motor ROI source vertices

# Electrodes over the hand sensory-motor cortex; the ROI is formed by the source
# vertices nearest to these channels.
SENSORIMOTOR_ROI_CHANNELS = [
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
]

# ---------------------------------------------------------------------------
# BCI Competition IV 2a event mapping
# ---------------------------------------------------------------------------
# GDF annotations store the stimulus code as a string (e.g. "769" = left hand).
# Map them to compact integer ids used downstream.
ANNOTATION_TO_EVENT_ID = {
    "769": 1,   # left-hand MI
    "770": 2,   # right-hand MI
}

# Human-readable labels for the compact event ids (used by mne.Epochs).
EVENT_ID = {
    "left_hand": 1,
    "right_hand": 2,
}
