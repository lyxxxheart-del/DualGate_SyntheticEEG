"""GDF loading and MNE Epochs construction for BCI Competition IV 2a.

Responsible for extracting the source archive, reading a subject GDF with MNE,
mapping the dataset's stimulus-code annotations to compact event ids, and
returning band-pass filtered, down-sampled MNE Epochs over the MI task window.
"""

import zipfile
from pathlib import Path

import mne

from .config import (
    ANNOTATION_TO_EVENT_ID,
    BCICIV_2A_CHANNELS,
    EOG_CHANNELS,
    EVENT_ID,
    FMAX,
    FMIN,
    FS,
    RAW_DIR,
    TMAX,
    TMIN,
    ZIP_PATH,
)


def extract_dataset(zip_path: Path | None = None, raw_dir: Path | None = None) -> Path:
    """Extract ``BCICIV_2a_gdf.zip`` into ``data/raw`` (idempotent).

    Only members that are not already present on disk are written, so repeated
    calls are cheap. Each member is confined inside ``raw_dir`` (guards against
    archive path traversal).
    """
    zip_path = Path(zip_path) if zip_path is not None else ZIP_PATH
    raw_dir = Path(raw_dir) if raw_dir is not None else RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_root = raw_dir.resolve()

    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            target = (raw_dir / name).resolve()
            if not target.is_relative_to(raw_root):
                raise ValueError(f"Unsafe zip member escapes raw_dir: {name!r}")
            if target.exists():
                continue
            zf.extract(name, raw_dir)
    return raw_dir


def load_epochs(
    subject: str = "A01",
    run: str = "T",
    event_id: dict | None = None,
    picks=None,
    fmin: float | None = None,
    fmax: float | None = None,
    fs: float | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
    raw_dir: Path | None = None,
) -> mne.Epochs:
    """Load a subject's GDF and return band-pass filtered, down-sampled Epochs.

    Reads ``<subject><run>.gdf`` (e.g. ``A01T.gdf``), maps the dataset's
    stimulus-code annotations ``"769"``/``"770"`` to event ids ``1``/``2``,
    band-pass filters the EEG, then epochs the MI task window ``[tmin, tmax]``
    and resamples to ``fs``.

    Parameters
    ----------
    subject : str
        Subject id, e.g. ``"A01"``.
    run : str
        ``"T"`` for the training session, ``"E"`` for the evaluation session.
    event_id : dict, optional
        Annotation-description -> integer-id mapping. Defaults to
        :data:`ANNOTATION_TO_EVENT_ID` (``"769"`` -> ``1``, ``"770"`` -> ``2``).
    picks : str | list, optional
        Channels to keep. Defaults to ``"eeg"`` (the 22 EEG channels; EOG dropped).
    fmin, fmax, fs, tmin, tmax : float, optional
        Preprocessing parameters; default to :data:`FMIN`, :data:`FMAX`,
        :data:`FS`, :data:`TMIN`, :data:`TMAX`.
    raw_dir : Path, optional
        Directory holding the extracted GDF files; defaults to :data:`RAW_DIR`.

    Returns
    -------
    epochs : mne.Epochs
        EEG-only Epochs over the MI window, events labeled ``{1: left-hand,
        2: right-hand}``, band-pass filtered ``[fmin, fmax]`` and resampled to
        ``fs``.
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else RAW_DIR
    gdf_path = raw_dir / f"{subject}{run}.gdf"
    if not gdf_path.exists():
        extract_dataset(raw_dir=raw_dir)

    event_id = ANNOTATION_TO_EVENT_ID if event_id is None else event_id
    picks = "eeg" if picks is None else picks
    fmin = FMIN if fmin is None else fmin
    fmax = FMAX if fmax is None else fmax
    fs = FS if fs is None else fs
    tmin = TMIN if tmin is None else tmin
    tmax = TMAX if tmax is None else tmax

    raw = mne.io.read_raw_gdf(gdf_path, preload=True)

    # The GDF header leaves most channel labels blank, so read_raw_gdf invents
    # names such as "EEG-0". Restore the known BCI IV-2a montage in file order
    # and mark the EOG channels so picks="eeg" keeps only the 22 EEG channels.
    if len(raw.ch_names) != len(BCICIV_2A_CHANNELS):
        raise ValueError(
            f"Expected {len(BCICIV_2A_CHANNELS)} channels in {gdf_path.name}, "
            f"got {len(raw.ch_names)}"
        )
    raw.rename_channels(dict(zip(raw.ch_names, BCICIV_2A_CHANNELS)))
    raw.set_channel_types({ch: "eog" for ch in EOG_CHANNELS})

    # Map stimulus-code annotations to compact ids, keeping only left/right hand.
    events, _ = mne.events_from_annotations(raw, event_id=event_id)

    raw = raw.filter(fmin, fmax, picks=picks)

    epochs = mne.Epochs(
        raw,
        events,
        event_id=EVENT_ID,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        picks=picks,
        preload=True,
    )

    # Down-sample to the target rate (no-op when the source is already at ``fs``).
    if epochs.info["sfreq"] != fs:
        epochs = epochs.resample(fs)

    return epochs
