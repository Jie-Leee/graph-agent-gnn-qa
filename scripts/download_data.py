"""Download and checksum-verify A's raw data without third-party packages.

Existing verified files are reused offline. Modified files require --force.
Only the three graph split files are extracted from the Microsoft archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
KG_BERT_COMMIT = "f3f260309c5e2986467a66e0e68fa7f53b1a150e"
ARCHIVE_URL = (
    "https://download.microsoft.com/download/8/7/0/"
    "8700516a-ab3d-4850-b4bb-805c515aece1/FB15K-237.2.zip"
)
ARCHIVE_NAME = "FB15K-237.2.zip"
ARCHIVE_SHA256 = "40bd8dca0e5a93f8c25b077b8f173f9a6660818f235bb5a6bb67c87d45f2da40"
GRAPH_SHA256 = {
    "train.txt": "6e4c2782169af21e9743f3b1d200886f5d595bf6bc504ec1351720949c5cdfae",
    "valid.txt": "cf6309010852f6a8d47a45df830a426415d1ee6f7a3970a8376ff1fb81db4a5c",
    "test.txt": "5711cf41623ceb4eacc50eb6108a3ca6565c7492e3caaf82a3e355cc660d1574",
}
TEXT_SHA256 = {
    "entity2text.txt": "16beedcfad3d32696e8b0b98cfa0f5eeb144f5142225f227a64fd9ce37447da5",
    "relation2text.txt": "4bf7cb6ea0ac09b17ad210889f723be5e079669f6ae5d507787e1320bbaa0f72",
}


class DownloadError(RuntimeError):
    """A download, archive or checksum did not satisfy the fixed data contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches(path: Path, expected: str) -> bool:
    return path.is_file() and sha256(path) == expected


def _atomic_copy(source, destination: Path, expected: str) -> None:
    """Read a stream into a temporary sibling; expose it only after verification."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=".download-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            digest = hashlib.sha256()
            for chunk in iter(lambda: source.read(1 << 20), b""):
                handle.write(chunk)
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise DownloadError(
                f"SHA256 校验失败：{destination.name}，预期 {expected}，实际 {actual}"
            )
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _download(url: str, destination: Path, expected: str) -> None:
    print(f"Downloading {destination.name} ...")
    request = urllib.request.Request(url, headers={"User-Agent": "graph-agent-data/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            _atomic_copy(response, destination, expected)
    except (OSError, urllib.error.URLError) as error:
        raise DownloadError(f"下载失败：{destination.name}：{error}") from error


def _extract_graph(archive: Path, raw_dir: Path, needed: list[str]) -> None:
    try:
        with zipfile.ZipFile(archive) as handle:
            # Reject unsafe archive entries even though extraction below uses
            # explicit allowlisted member names rather than extractall().
            for member in handle.infolist():
                original_name = member.orig_filename
                path = PurePosixPath(original_name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or "\\" in original_name
                    or ":" in original_name
                    or stat.S_ISLNK(member.external_attr >> 16)
                ):
                    raise DownloadError(f"ZIP 包含不安全路径：{member.filename}")
            for filename in needed:
                member_name = f"Release/{filename}"
                matches = [m for m in handle.infolist() if m.filename == member_name]
                if len(matches) != 1:
                    raise DownloadError(f"ZIP 中必须有且仅有一个 {member_name}")
                with handle.open(matches[0]) as source:
                    _atomic_copy(source, raw_dir / "FB15K-237" / filename, GRAPH_SHA256[filename])
    except (zipfile.BadZipFile, KeyError) as error:
        raise DownloadError(f"无效的官方 ZIP：{error}") from error


def download_data(raw_dir: Path, *, force: bool = False) -> dict:
    raw_dir = Path(raw_dir)
    expected = {
        **{raw_dir / "FB15K-237" / name: digest for name, digest in GRAPH_SHA256.items()},
        **{raw_dir / name: digest for name, digest in TEXT_SHA256.items()},
    }
    archive = raw_dir / ARCHIVE_NAME
    if archive.exists():
        expected[archive] = ARCHIVE_SHA256
    # Preflight all existing destinations before making any changes.
    matches = {path: _matches(path, digest) for path, digest in expected.items()}
    modified = [path for path, ok in matches.items() if path.exists() and not ok]
    if modified and not force:
        raise DownloadError(
            "已有文件与固定版本校验值不符，未覆盖："
            + ", ".join(str(path) for path in modified)
            + "。确认需要替换后使用 --force。"
        )
    raw_dir.mkdir(parents=True, exist_ok=True)
    needed = [name for name in GRAPH_SHA256 if not matches[raw_dir / "FB15K-237" / name]]
    if needed:
        if not _matches(archive, ARCHIVE_SHA256):
            _download(ARCHIVE_URL, archive, ARCHIVE_SHA256)
        _extract_graph(archive, raw_dir, needed)
    elif archive in modified:
        _download(ARCHIVE_URL, archive, ARCHIVE_SHA256)
    for filename, digest in TEXT_SHA256.items():
        if not matches[raw_dir / filename]:
            url = (
                f"https://raw.githubusercontent.com/yao8839836/kg-bert/{KG_BERT_COMMIT}"
                f"/data/FB15k-237/{filename}"
            )
            _download(url, raw_dir / filename, digest)
    provenance = {
        "graph_url": ARCHIVE_URL,
        "archive_sha256": ARCHIVE_SHA256,
        "text_repository": "https://github.com/yao8839836/kg-bert",
        "text_commit": KG_BERT_COMMIT,
        "files": {
            **{f"FB15K-237/{name}": digest for name, digest in GRAPH_SHA256.items()},
            **TEXT_SHA256,
        },
    }
    (raw_dir / "download_sources.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Verified {len(GRAPH_SHA256) + len(TEXT_SHA256)} raw files: {raw_dir}")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="下载并校验固定版本的 FB15k-237 与文本映射")
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--force", action="store_true", help="替换与固定校验值不符的已有原始文件")
    args = parser.parse_args()
    try:
        download_data(args.raw, force=args.force)
    except (DownloadError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
