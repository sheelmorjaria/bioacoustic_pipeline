#!/usr/bin/env python3
"""Inspect deposited source artifacts for provenance without executing their code."""
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path, algorithm='sha256'):
    with path.open('rb') as source:
        return hashlib.file_digest(source, algorithm).hexdigest()


def main():
    path = ROOT / 'data/marmaudio/Code_Usage.zip'
    repository = json.loads((path.parent / 'record.json').read_text())
    info = next(f for f in repository['files'] if f['key'] == path.name)
    algorithm, expected = info['checksum'].split(':')
    if path.stat().st_size != info['size'] or sha(path, algorithm) != expected:
        raise ValueError('Repository checksum mismatch')
    members, tables, references = [], [], []
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            members.append({'name': item.filename, 'bytes': item.file_size})
            if item.filename.endswith(('.tsv', '.csv')):
                with archive.open(item) as source:
                    header = source.readline().decode().strip('\r\n').split('\t' if item.filename.endswith('.tsv') else ',')
                tables.append({'name': item.filename, 'columns': header,
                               'raw_recording_fields': [c for c in header if c.startswith('raw_recording')]})
            if item.filename.endswith('.py'):
                for i, line in enumerate(archive.read(item).decode().splitlines(), 1):
                    if any(key in line for key in ('read_csv', 'Annotations.tsv', 'raw_recording')):
                        references.append({'member': item.filename, 'line': i, 'source': line.strip()})
    ceb_path = ROOT / 'data/ceb/train_xenocanto.tar.gz'
    non_audio = []
    with tarfile.open(ceb_path, 'r|gz') as archive:
        for member in archive:
            if member.isfile() and not member.name.lower().endswith('.flac'):
                item = {'member': member.name, 'bytes': member.size}
                if member.name.lower().endswith(('.csv', '.tsv', '.json', '.txt', '.md')) and member.size <= 10_000_000:
                    raw = archive.extractfile(member).read()
                    item['sha256'] = hashlib.sha256(raw).hexdigest()
                    item['first_line'] = raw.decode(errors='replace').splitlines()[:1]
                non_audio.append(item)
    result = {'code_sha256': sha(Path(__file__)),
              'MarmAudio': {'archive_sha256': sha(path), 'repository_checksum': info['checksum'],
                           'members': members, 'table_schemas': tables, 'code_metadata_references': references,
                           'parent_session_crosswalk_recovered': False,
                           'finding': 'Deposited code bundle contains a label table with archive-folder/file/type columns, not the required raw-recording/offset/session crosswalk.'},
              'CEB': {'archive_sha256': sha(ceb_path), 'non_flac_files': non_audio,
                      'parent_session_metadata_recovered': False,
                      'finding': 'No source/session metadata recovered from focal-audio archive inspection; acquired source CSV lacks dates/session IDs. HTTP401 is an access failure, not proof original provenance does not exist.'},
              'source_certification_ledger_sha256': sha(ROOT / 'outputs/macde_source_certification_ledger.json'),
              'certified_independent_groups': 0, 'partitions_assigned': False, 'models_fitted': False, 'stage2_ready': False,
              'next_evidence': 'Authoritative clip-to-parent/offset/session crosswalk for MarmAudio; original XC metadata or deposited source manifest for CEB. No weaker grouping claim substituted.'}
    out = ROOT / 'outputs/macde_provenance_recovery_audit.json'
    if out.exists():
        raise RuntimeError('Refusing to overwrite recovery audit')
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'MarmAudio_crosswalk_recovered': False, 'CEB_non_flac_files': non_audio,
                      'certified_groups': 0, 'output': str(out)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
