import stat
from zipfile import ZipInfo
import pytest
from bioacoustic_pipeline.diagnostics.rclat_vctk_source_contract import (
    validate_archive_members,
    recording_identity,
)


def test_archive_rejects_escape_duplicate_and_symlink():
    for names in [
        ["../escape"],
        ["/absolute"],
        ["C:/escape"],
        ["root/file", "root/file"],
        ["root\\escape"],
    ]:
        with pytest.raises(ValueError):
            validate_archive_members([ZipInfo(n) for n in names])
    link = ZipInfo("root/link")
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(ValueError):
        validate_archive_members([link])
    assert validate_archive_members([ZipInfo("root/mic/p225_001_mic2.flac")])["members"] == 1


def test_microphone_variants_share_one_event_with_no_fallback():
    a = recording_identity("p225/p225_001_mic1.flac")
    b = recording_identity("p225/p225_001_mic2.flac")
    assert a["event"] == b["event"] == "p225_001"
    assert not a["selected_microphone"] and b["selected_microphone"]
    with pytest.raises(ValueError):
        recording_identity("p226/p225_001_mic2.flac")
