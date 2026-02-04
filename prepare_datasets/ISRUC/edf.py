from pathlib import Path
import os, tempfile, shutil
from mne.io import read_raw_edf as _mne_read_raw_edf
from mne.io import read_raw_bdf as _mne_read_raw_bdf
from mne.io import read_raw_gdf as _mne_read_raw_gdf
def _as_edf_path(path):
    p = Path(path)
    if p.suffix.lower() != '.rec':
        return str(p)
    tmpdir = tempfile.mkdtemp(prefix='mne_rec2edf_')
    edf_path = Path(tmpdir) / (p.stem + '.edf')
    try:
        os.symlink(p, edf_path)
    except Exception:
        try:
            os.link(p, edf_path)
        except Exception:
            shutil.copy2(p, edf_path)
    return str(edf_path)
def read_raw_edf(input_fname, *args, **kwargs):
    return _mne_read_raw_edf(_as_edf_path(input_fname), *args, **kwargs)
def read_raw_bdf(input_fname, *args, **kwargs):
    return _mne_read_raw_bdf(input_fname, *args, **kwargs)
def read_raw_gdf(input_fname, *args, **kwargs):
    return _mne_read_raw_gdf(input_fname, *args, **kwargs)
