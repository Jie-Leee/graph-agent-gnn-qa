"""Offline tests for the raw-data downloader's integrity guarantees."""
from __future__ import annotations

import hashlib
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import download_data as downloader  # noqa: E402


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class DownloadTests(unittest.TestCase):
    def test_bad_download_preserves_existing_file_and_cleans_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "train.txt"
            target.write_bytes(b"existing")
            with self.assertRaises(downloader.DownloadError):
                downloader._atomic_copy(io.BytesIO(b"corrupt"), target, digest(b"expected"))
            self.assertEqual(target.read_bytes(), b"existing")
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_valid_download_atomically_replaces_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "train.txt"
            downloader._atomic_copy(io.BytesIO(b"expected"), target, digest(b"expected"))
            self.assertEqual(target.read_bytes(), b"expected")

    def test_modified_raw_is_rejected_before_network_or_other_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            (raw / "entity2text.txt").write_bytes(b"modified")
            with patch.object(downloader, "_download") as network:
                with self.assertRaisesRegex(downloader.DownloadError, "未覆盖"):
                    downloader.download_data(raw)
                network.assert_not_called()
            self.assertEqual((raw / "entity2text.txt").read_bytes(), b"modified")
            self.assertFalse((raw / "FB15K-237").exists())

    def test_complete_raw_directory_is_reused_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            (raw / "FB15K-237").mkdir()
            graph = {name: digest(name.encode()) for name in downloader.GRAPH_SHA256}
            text = {name: digest(name.encode()) for name in downloader.TEXT_SHA256}
            for name in graph:
                (raw / "FB15K-237" / name).write_bytes(name.encode())
            for name in text:
                (raw / name).write_bytes(name.encode())
            with (
                patch.dict(downloader.GRAPH_SHA256, graph, clear=True),
                patch.dict(downloader.TEXT_SHA256, text, clear=True),
                patch.object(downloader, "_download") as network,
            ):
                result = downloader.download_data(raw)
                network.assert_not_called()
            self.assertEqual(len(result["files"]), 5)
            self.assertTrue((raw / "download_sources.json").exists())

    def test_extraction_allows_only_requested_graph_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            archive = raw / "archive.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("Release/train.txt", b"train")
                handle.writestr("Release/unrequested.txt", b"extra")
            with patch.dict(downloader.GRAPH_SHA256, {"train.txt": digest(b"train")}):
                downloader._extract_graph(archive, raw, ["train.txt"])
            self.assertEqual((raw / "FB15K-237" / "train.txt").read_bytes(), b"train")
            self.assertFalse((raw / "Release").exists())
            self.assertFalse((raw / "FB15K-237" / "unrequested.txt").exists())

    def test_unsafe_zip_members_are_rejected(self) -> None:
        for member in ["../escape", "/absolute", "C:/escape", "Release\\escape"]:
            with self.subTest(member=member), tempfile.TemporaryDirectory() as directory:
                raw = Path(directory)
                archive = raw / "archive.zip"
                with zipfile.ZipFile(archive, "w") as handle:
                    # ZipInfo normally normalizes backslashes on Windows;
                    # assign after construction to exercise hostile archives.
                    info = zipfile.ZipInfo("placeholder")
                    info.filename = member
                    handle.writestr(info, b"bad")
                with self.assertRaisesRegex(downloader.DownloadError, "不安全路径"):
                    downloader._extract_graph(archive, raw, [])


if __name__ == "__main__":
    unittest.main()
