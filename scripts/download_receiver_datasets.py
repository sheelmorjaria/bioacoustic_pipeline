#!/usr/bin/env python3
"""Download pinned BirdPark resources; import elephant Dryad ZIP when available.

Small files by default. --pair-data adds 1.2 GB; --group-data adds 12.3 GB.
Dryad HTTP failures are recorded, never saved as dataset CSVs. A downloaded
Dryad ZIP in ~/Downloads is verified file-by-file against current API digests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bioacoustic_pipeline.ingestion.receiver_grounding import verified_extract


def checksum(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fetch(url, path, expected=None, size=None):
    if path.exists() and (expected is None or checksum(path, expected.split(':')[0]) == expected.split(':')[1]):
        return {'path': str(path.relative_to(ROOT)), 'status': 'available', 'sha256': checksum(path)}
    part = path.with_suffix(path.suffix + '.part')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, part.open('wb') as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        if size is not None and part.stat().st_size != size:
            raise ValueError(f'Unexpected byte count: {path.name}')
        if expected and checksum(part, expected.split(':')[0]) != expected.split(':')[1]:
            raise ValueError(f'Checksum mismatch: {path.name}')
        part.replace(path)
        return {'path': str(path.relative_to(ROOT)), 'status': 'downloaded', 'sha256': checksum(path)}
    except (OSError, ValueError, urllib.error.URLError) as error:
        return {'path': str(path.relative_to(ROOT)), 'status': 'unavailable', 'reason': str(error), 'url': url}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pair-data', action='store_true')
    ap.add_argument('--group-data', action='store_true')
    ap.add_argument('--local-only', action='store_true')
    ap.add_argument('--downloads', type=Path, default=Path('/home/sheel/Downloads'))
    args = ap.parse_args()
    rows = []
    for folder, record in [('birdpark_pairs', '7105196'), ('birdpark_groups', '13144875')]:
        base = ROOT / 'data' / folder
        metadata = base / 'record.json'
        if not metadata.exists() and not args.local_only:
            rows.append(fetch(f'https://zenodo.org/api/records/{record}', metadata))
        if not metadata.exists():
            continue
        for f in json.loads(metadata.read_text())['files']:
            if f['key'] == 'Data.zip' and not (args.pair_data if folder == 'birdpark_pairs' else args.group_data):
                if not (base / f['key']).exists():
                    rows.append({'path': str((base / f['key']).relative_to(ROOT)), 'status': 'deferred_large_archive'})
                    continue
            path = base / f['key']
            part = path.with_suffix(path.suffix + '.part')
            if part.exists() and part.stat().st_size == f['size'] and checksum(part, 'md5') == f['checksum'].split(':')[1]:
                part.replace(path)
            if args.local_only and not path.exists():
                rows.append({'path': str(path.relative_to(ROOT)), 'status': 'not_downloaded'})
                continue
            if args.local_only:
                assert checksum(path, 'md5') == f['checksum'].split(':')[1], path
                row = {'path': str(path.relative_to(ROOT)), 'status': 'verified', 'sha256': checksum(path)}
            else:
                row = fetch(f'https://zenodo.org/records/{record}/files/{f["key"]}?download=1',
                            path, f['checksum'], f['size'])
            rows.append(row)
            if path.suffix == '.zip' and row['status'] != 'unavailable':
                verified_extract(path, base, f['checksum'])
    base = ROOT / 'data/elephant_playback'
    metadata = base / 'files.json'
    if metadata.exists():
        files = json.loads(metadata.read_text())['_embedded']['stash:files']
        # Only use the latest version; original playback CSV remains archived, unused.
        wanted = [f for f in files if f['path'] != '20230409_playback_data_for_upload.csv']
        for archive in sorted(args.downloads.glob('*hmgqnk9nj*.zip')):
            with zipfile.ZipFile(archive) as z:
                for f in wanted:
                    matches = [n for n in z.namelist() if Path(n).name == f['path']]
                    if len(matches) == 1:
                        data = z.read(matches[0])
                        if hashlib.sha256(data).hexdigest() != f['digest']:
                            raise ValueError(f'Dryad archive member checksum mismatch: {f["path"]}')
                        (base / f['path']).write_bytes(data)
        for f in wanted:
            path = base / f['path']
            expected = f'sha256:{f["digest"]}'
            if path.exists():
                if checksum(path) != f['digest']:
                    raise ValueError(f'Dryad file checksum mismatch: {path.name}')
                rows.append({'path': str(path.relative_to(ROOT)), 'status': 'verified', 'sha256': checksum(path)})
            elif not args.local_only:
                url = 'https://datadryad.org' + f['_links']['stash:download']['href']
                rows.append(fetch(url, path, expected, f['size']))
            else:
                rows.append({'path': str(path.relative_to(ROOT)), 'status': 'not_downloaded'})
    out = ROOT / 'outputs/receiver_dataset_downloads.json'
    out.write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
