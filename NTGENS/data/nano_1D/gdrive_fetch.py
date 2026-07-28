"""Fetch the Alexandria 1D-nanotube pickle(s) from Google Drive into this folder.

The raw ASE pickles live in Google Drive (several files, ~48 MB each). This module
downloads any that are missing locally, so build_templates.py
can turn them into the compact template cache the model pins during denoising.

This is a *preprocess-stage* helper only: it is never imported at generation time
(runtime just mmaps nanotube_templates.npz, no Google/ase deps needed). It degrades
gracefully when the download libraries are absent -- it prints an install hint and
exits without touching anything.

Files to pull are declared in gdrive_manifest.json (see that file for the schema):
each entry is a local `name` plus a Drive `id` (the long token in a share link
https://drive.google.com/file/d/<id>/view). A whole shared folder can be given via
`folder_url` instead of per-file ids.

Two backends, tried in order:
  1. gdown            -- simplest; works with "anyone with the link" files/folders.
  2. google-api-python-client + a service-account JSON (env NTU_IQM_GCP_CREDS) --
     for private files shared with the service-account email.

Usage:
    python data/nano_1D/gdrive_fetch.py                 # fetch all missing files
    python data/nano_1D/gdrive_fetch.py --force         # re-download even if cached
    python data/nano_1D/gdrive_fetch.py --list          # show manifest vs. local state
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / 'gdrive_manifest.json'


def _load_manifest(path=MANIFEST):
    if not path.exists():
        raise SystemExit(
            f'No manifest at {path}. Create it (see the schema in gdrive_manifest.json) '
            'with the Drive file ids of your pkl files.')
    with open(path) as fh:
        man = json.load(fh)
    man.setdefault('files', [])
    man.setdefault('folder_url', None)
    # Explicit null (or missing) dest_dir -> this data/nano_1D folder.
    if not man.get('dest_dir'):
        man['dest_dir'] = str(HERE)
    return man


def _md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def _ok_local(dest, entry):
    """A local file counts as present if it exists and (when the manifest gives an
    md5) matches it. A mismatching md5 means a stale/partial download -> re-fetch."""
    if not dest.exists():
        return False
    want = entry.get('md5')
    if want and _md5(dest) != want:
        print(f'  {dest.name}: md5 mismatch, will re-download')
        return False
    return True


# --- Backend 1: gdown (per-file id or whole folder) ---------------------------

def _gdown_available():
    try:
        import gdown  # noqa: F401
        return True
    except ModuleNotFoundError:
        return False


def _fetch_with_gdown(man, dest_dir, force):
    import gdown
    fetched = []
    # Whole-folder mode: let gdown mirror the shared folder into dest_dir.
    if man.get('folder_url'):
        print(f'Fetching folder into {dest_dir} via gdown...')
        gdown.download_folder(url=man['folder_url'], output=str(dest_dir),
                              quiet=False, use_cookies=False)
    for entry in man['files']:
        dest = dest_dir / entry['name']
        if not force and _ok_local(dest, entry):
            print(f'  cached: {dest.name}')
            fetched.append(dest)
            continue
        fid = entry.get('id')
        if not fid:
            print(f'  skip {entry["name"]}: no "id" in manifest')
            continue
        print(f'  downloading {dest.name} ...')
        gdown.download(id=fid, output=str(dest), quiet=False)
        fetched.append(dest)
    return fetched


# --- Backend 2: Drive API + service account -----------------------------------

def _drive_api_available():
    try:
        import googleapiclient  # noqa: F401
        import google.oauth2.service_account  # noqa: F401
        return os.getenv('NTU_IQM_GCP_CREDS') is not None
    except ModuleNotFoundError:
        return False


def _fetch_with_api(man, dest_dir, force):
    import io
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    creds = Credentials.from_service_account_file(
        os.getenv('NTU_IQM_GCP_CREDS'),
        scopes=['https://www.googleapis.com/auth/drive.readonly'])
    drive = build('drive', 'v3', credentials=creds)

    fetched = []
    for entry in man['files']:
        dest = dest_dir / entry['name']
        if not force and _ok_local(dest, entry):
            print(f'  cached: {dest.name}')
            fetched.append(dest)
            continue
        fid = entry.get('id')
        if not fid:  # no id: resolve by name via a Drive search
            q = f"name = '{entry['name']}' and trashed = false"
            res = drive.files().list(q=q, spaces='drive', pageSize=1,
                                     fields='files(id)').execute()
            hits = res.get('files', [])
            if not hits:
                print(f'  skip {entry["name"]}: not found in Drive')
                continue
            fid = hits[0]['id']
        print(f'  downloading {dest.name} ...')
        req = drive.files().get_media(fileId=fid)
        with open(dest, 'wb') as fh:
            dl = MediaIoBaseDownload(fh, req, chunksize=16 << 20)
            done = False
            while not done:
                status, done = dl.next_chunk()
                if status:
                    print(f'    {status.progress() * 100:5.1f}%', end='\r')
        print()
        fetched.append(dest)
    return fetched


def fetch_all(manifest_path=MANIFEST, force=False):
    """Download every manifest file missing from dest_dir. Returns list of Paths.

    Picks the first available backend (gdown, then Drive API). Raises SystemExit
    with an install/config hint if neither is usable."""
    man = _load_manifest(manifest_path)
    dest_dir = Path(man['dest_dir'])
    dest_dir.mkdir(parents=True, exist_ok=True)

    if _gdown_available():
        return _fetch_with_gdown(man, dest_dir, force)
    if _drive_api_available():
        return _fetch_with_api(man, dest_dir, force)
    raise SystemExit(
        'No Google Drive backend available. Install one of:\n'
        '  pip install gdown            # simplest; needs "anyone with link" files\n'
        '  pip install google-api-python-client google-auth   # + set NTU_IQM_GCP_CREDS\n'
        'then set the file ids in gdrive_manifest.json and re-run.')


def _print_status(manifest_path=MANIFEST):
    man = _load_manifest(manifest_path)
    dest_dir = Path(man['dest_dir'])
    print(f'dest_dir: {dest_dir}')
    print(f'backend : {"gdown" if _gdown_available() else ("drive-api" if _drive_api_available() else "NONE")}')
    for entry in man['files']:
        dest = dest_dir / entry['name']
        state = 'present' if _ok_local(dest, entry) else 'MISSING'
        fid = entry.get('id', '<by-name>')
        print(f'  [{state:>7}] {entry["name"]}  (id={fid})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--force', action='store_true',
                    help='re-download even if a valid local copy exists')
    ap.add_argument('--list', action='store_true',
                    help='show manifest vs. local state, download nothing')
    ap.add_argument('--manifest', default=str(MANIFEST))
    args = ap.parse_args()

    if args.list:
        _print_status(Path(args.manifest))
        sys.exit(0)

    got = fetch_all(Path(args.manifest), force=args.force)
    print(f'\nready: {len(got)} file(s) in place')
    for p in got:
        print(f'  {p}')
