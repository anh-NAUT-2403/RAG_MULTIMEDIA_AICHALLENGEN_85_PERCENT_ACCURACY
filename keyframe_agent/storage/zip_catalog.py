from __future__ import annotations

import shutil
import re
import zipfile
from pathlib import Path


class ZipCatalog:
    """Locate and extract exact members, never unpack a complete archive."""

    def __init__(self, zip_paths: list[Path]) -> None:
        self.zip_paths = sorted(Path(path) for path in zip_paths)
        self._resolved: dict[str, Path] = {}
        self._archives: dict[Path, zipfile.ZipFile] = {}

    def _archive(self, zip_path: Path) -> zipfile.ZipFile:
        archive = self._archives.get(zip_path)
        if archive is None:
            archive = zipfile.ZipFile(zip_path)
            self._archives[zip_path] = archive
        return archive

    def close(self) -> None:
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()

    def __del__(self) -> None:
        self.close()

    def _ordered_paths(self, member_or_suffix: str) -> list[Path]:
        match = re.search(r"(L\d+)_V\d+", member_or_suffix)
        if match is None:
            return self.zip_paths
        collection = match.group(1)
        preferred = [
            path for path in self.zip_paths if f"_{collection}" in path.stem
        ]
        return preferred or self.zip_paths

    def locate(self, member: str) -> Path:
        if member in self._resolved:
            return self._resolved[member]
        for zip_path in self._ordered_paths(member):
            archive = self._archive(zip_path)
            try:
                archive.getinfo(member)
            except KeyError:
                continue
            self._resolved[member] = zip_path
            return zip_path
        raise FileNotFoundError(f"ZIP member not found: {member}")

    def locate_by_suffix(self, suffix: str) -> tuple[Path, str]:
        suffix = suffix.replace("\\", "/")
        for zip_path in self._ordered_paths(suffix):
            archive = self._archive(zip_path)
            for member in archive.namelist():
                if member.endswith(suffix):
                    return zip_path, member
        raise FileNotFoundError(f"ZIP member suffix not found: {suffix}")

    def extract(self, member: str, output_path: Path) -> Path:
        output_path = Path(output_path)
        if output_path.is_file() and output_path.stat().st_size > 0:
            return output_path
        zip_path = self.locate(member)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        archive = self._archive(zip_path)
        with archive.open(member) as source:
            with output_path.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
        return output_path

    def extract_suffix(self, suffix: str, output_path: Path) -> Path:
        output_path = Path(output_path)
        if output_path.is_file() and output_path.stat().st_size > 0:
            return output_path
        zip_path, member = self.locate_by_suffix(suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        archive = self._archive(zip_path)
        with archive.open(member) as source:
            with output_path.open("wb") as target:
                shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
        return output_path
