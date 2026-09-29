"""Acquire one UTC day of official OKX BTC-USDT trades and depth-400 data."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile


SYMBOL = "BTC-USDT"
SOURCE_PAGE = "https://www.okx.com/historical-data"
DATA_ROOT = Path(r"C:\DaxAlgoData")
TRADE_HEADER = [
    "instrument_name",
    "trade_id",
    "side",
    "price",
    "size",
    "created_time",
]
L2_KEYS = {"instId", "action", "ts", "asks", "bids"}
MINIMUM_DAY = date(2023, 3, 23)
CHUNK_SIZE = 4 * 1024 * 1024
MAX_NDJSON_LINE = 4 * 1024 * 1024
MANIFEST_SCHEMA = "okx-btc-usdt-utc-day-v1"
MANIFEST_NAME = re.compile(
    rf"okx-{re.escape(SYMBOL)}-(\d{{4}}-\d{{2}}-\d{{2}})-manifest\.json"
)
USER_AGENT = "CHMT-Q research acquisition/1.0"


@dataclass(frozen=True)
class RemoteMetadata:
    content_length: int
    content_type: str
    etag: str | None
    last_modified: str | None


@dataclass(frozen=True)
class SourceFile:
    kind: str
    label_day: date
    filename: str
    member_name: str
    url: str


def parse_day(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("day must be YYYY-MM-DD") from exc


def validate_destination(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("--destination must be an absolute path under C:\\DaxAlgoData")
    destination = path.resolve(strict=False)
    root = DATA_ROOT.resolve(strict=False)
    try:
        relative = destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("--destination must remain under C:\\DaxAlgoData") from exc
    if not relative.parts:
        raise ValueError("choose an explicit subdirectory beneath C:\\DaxAlgoData")
    return destination


def build_sources(day: date) -> list[SourceFile]:
    def trade_source(label_day: date) -> SourceFile:
        iso = label_day.isoformat()
        compact = label_day.strftime("%Y%m%d")
        filename = f"{SYMBOL}-trades-{iso}.zip"
        return SourceFile(
            kind="trades",
            label_day=label_day,
            filename=filename,
            member_name=f"{SYMBOL}-trades-{iso}.csv",
            url=(
                "https://static.okx.com/cdn/okex/traderecords/trades/daily/"
                f"{compact}/{filename}?v=999"
            ),
        )

    iso = day.isoformat()
    compact = day.strftime("%Y%m%d")
    l2_filename = f"{SYMBOL}-L2orderbook-400lv-{iso}.tar.gz"
    return [
        trade_source(day),
        trade_source(day + timedelta(days=1)),
        SourceFile(
            kind="depth400",
            label_day=day,
            filename=l2_filename,
            member_name=f"{SYMBOL}-L2orderbook-400lv-{iso}.data",
            url=(
                "https://static.okx.com/cdn/okx/match/orderbook/pro/L2/400lv/"
                f"daily/{compact}/{l2_filename}"
            ),
        ),
    ]


def utc_bounds_ms(day: date) -> tuple[int, int]:
    start = datetime.combine(day, time.min, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def normalize_content_type(value: str | None) -> str:
    return (value or "").split(";", 1)[0].strip().lower()


def remote_metadata(source: SourceFile) -> RemoteMetadata:
    request = Request(source.url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=60) as response:  # noqa: S310 - fixed HTTPS hosts
            if response.status != 200:
                raise RuntimeError(
                    f"HEAD returned HTTP {response.status}: {source.url}"
                )
            length_text = response.headers.get("Content-Length")
            if not length_text or not length_text.isdigit():
                raise RuntimeError(f"missing Content-Length: {source.url}")
            metadata = RemoteMetadata(
                content_length=int(length_text),
                content_type=normalize_content_type(
                    response.headers.get("Content-Type")
                ),
                etag=response.headers.get("ETag"),
                last_modified=response.headers.get("Last-Modified"),
            )
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"HEAD failed for {source.url}: {exc}") from exc

    expected_type = "application/zip" if source.kind == "trades" else "application/gzip"
    if metadata.content_type != expected_type:
        raise RuntimeError(
            f"unexpected Content-Type for {source.filename}: "
            f"{metadata.content_type!r} != {expected_type!r}"
        )
    return metadata


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def metadata_matches(left: dict[str, Any], right: RemoteMetadata) -> bool:
    return left == asdict(right)


def safe_member_name(name: str) -> bool:
    member = PurePosixPath(name.replace("\\", "/"))
    return not member.is_absolute() and ".." not in member.parts


def validate_trade_archive(path: Path, source: SourceFile) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) != 1:
                raise ValueError(f"expected one ZIP member, found {len(infos)}")
            info = infos[0]
            if info.filename != source.member_name or not safe_member_name(
                info.filename
            ):
                raise ValueError(f"unexpected ZIP member: {info.filename!r}")
            if info.is_dir() or info.flag_bits & 0x1:
                raise ValueError("trade CSV must be a regular, unencrypted member")
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"ZIP CRC failed for {bad_member}")
            with archive.open(info) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
                    reader = csv.reader(text)
                    header = next(reader, None)
                    first = next(reader, None)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"invalid trade ZIP {path.name}: {exc}") from exc

    if header != TRADE_HEADER:
        raise ValueError(f"unexpected trade CSV header: {header!r}")
    if first is None or len(first) != len(TRADE_HEADER):
        raise ValueError("trade CSV has no valid first data row")
    if first[0] != SYMBOL or first[2] not in {"buy", "sell"}:
        raise ValueError(f"unexpected first trade row: {first!r}")
    try:
        int(first[1])
        Decimal(first[3])
        Decimal(first[4])
        timestamp = int(first[5])
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(
            f"invalid numeric value in first trade row: {first!r}"
        ) from exc

    # OKX trade archive labels are Asia/Shanghai calendar days (UTC+08:00).
    shanghai = timezone(timedelta(hours=8))
    label_start = datetime.combine(source.label_day, time.min, tzinfo=shanghai)
    label_start_ms = int(label_start.timestamp() * 1000)
    if not label_start_ms <= timestamp < label_start_ms + 86_400_000:
        raise ValueError("first trade timestamp is outside its labelled UTC+08:00 day")


def validate_level(level: Any) -> None:
    if not isinstance(level, list) or len(level) != 3:
        raise ValueError(f"invalid L2 level: {level!r}")
    try:
        price = Decimal(level[0])
        size = Decimal(level[1])
        order_count = int(level[2])
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"invalid L2 level values: {level!r}") from exc
    if price <= 0 or size < 0 or order_count < 0:
        raise ValueError(f"out-of-range L2 level values: {level!r}")


def validate_depth_archive(path: Path, source: SourceFile, day: date) -> None:
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            members = archive.getmembers()
            if len(members) != 1:
                raise ValueError(f"expected one tar member, found {len(members)}")
            member = members[0]
            if (
                member.name != source.member_name
                or not safe_member_name(member.name)
                or not member.isfile()
            ):
                raise ValueError(f"unexpected tar member: {member.name!r}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError("unable to open L2 data member")
            first_line = extracted.readline(MAX_NDJSON_LINE + 1)
    except (EOFError, OSError, tarfile.TarError) as exc:
        raise ValueError(f"invalid depth archive {path.name}: {exc}") from exc

    if (
        not first_line
        or len(first_line) > MAX_NDJSON_LINE
        or not first_line.endswith(b"\n")
    ):
        raise ValueError("invalid or oversized first L2 NDJSON record")
    try:
        record = json.loads(first_line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("first L2 record is not valid UTF-8 JSON") from exc
    if set(record) != L2_KEYS:
        raise ValueError(f"unexpected L2 keys: {sorted(record)}")
    if record["instId"] != SYMBOL or record["action"] != "snapshot":
        raise ValueError("first L2 record must be a BTC-USDT snapshot")
    if not isinstance(record["asks"], list) or not isinstance(record["bids"], list):
        raise ValueError("L2 asks and bids must be arrays")
    if not 1 <= len(record["asks"]) <= 400 or not 1 <= len(record["bids"]) <= 400:
        raise ValueError("initial L2 snapshot must contain 1 to 400 levels per side")
    for level in record["asks"] + record["bids"]:
        validate_level(level)
    start_ms, end_ms = utc_bounds_ms(day)
    try:
        timestamp = int(record["ts"])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid first L2 timestamp") from exc
    if not start_ms <= timestamp < end_ms:
        raise ValueError("first L2 timestamp is outside the requested UTC day")


def load_manifest(path: Path, day: date, destination: Path) -> dict[str, Any]:
    start_ms, end_ms = utc_bounds_ms(day)
    expected = {
        "schema": MANIFEST_SCHEMA,
        "symbol": SYMBOL,
        "utc_day": day.isoformat(),
        "utc_interval_ms": {"start_inclusive": start_ms, "end_exclusive": end_ms},
        "destination": str(destination),
        "source_page": SOURCE_PAGE,
    }
    if not path.exists():
        now = datetime.now(timezone.utc).isoformat()
        return {**expected, "created_at_utc": now, "updated_at_utc": now, "files": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid manifest {path}: {exc}") from exc
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"manifest mismatch for {key}: {payload.get(key)!r} != {value!r}"
            )
    if not isinstance(payload.get("files"), list):
        raise ValueError("manifest files must be a list")
    return payload


def completed_entry(
    manifest: dict[str, Any], source: SourceFile, metadata: RemoteMetadata, path: Path
) -> dict[str, Any] | None:
    entries = [
        item for item in manifest["files"] if item.get("filename") == source.filename
    ]
    if not entries:
        return None
    if len(entries) != 1:
        raise ValueError(f"duplicate manifest entries for {source.filename}")
    entry = entries[0]
    expected_identity = {
        "kind": source.kind,
        "label_day": source.label_day.isoformat(),
        "filename": source.filename,
        "member_name": source.member_name,
        "url": source.url,
    }
    for key, value in expected_identity.items():
        if entry.get(key) != value:
            raise ValueError(f"manifest identity mismatch for {source.filename}: {key}")
    if not metadata_matches(entry.get("remote", {}), metadata):
        raise RuntimeError(
            f"remote metadata changed for {source.filename}; refusing overwrite"
        )
    if not path.is_file() or path.stat().st_size != metadata.content_length:
        raise RuntimeError(f"manifested file is missing or has the wrong size: {path}")
    digest = sha256_path(path)
    if digest != entry.get("sha256"):
        raise RuntimeError(f"SHA-256 mismatch for manifested file: {path}")
    return entry


def reuse_from_trusted_manifest(
    *,
    current_manifest_path: Path,
    destination: Path,
    source: SourceFile,
    metadata: RemoteMetadata,
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"existing shared path is not a regular file: {path}")

    references: list[tuple[Path, dict[str, Any]]] = []
    allowed_manifest_days = {source.label_day}
    if source.kind == "trades":
        allowed_manifest_days.add(source.label_day - timedelta(days=1))
    for candidate_path in sorted(destination.glob(f"okx-{SYMBOL}-*-manifest.json")):
        if candidate_path == current_manifest_path:
            continue
        match = MANIFEST_NAME.fullmatch(candidate_path.name)
        if match is None:
            raise ValueError(f"unrecognized manifest filename: {candidate_path.name}")
        try:
            candidate_day = date.fromisoformat(match.group(1))
        except ValueError as exc:
            raise ValueError(f"invalid manifest date: {candidate_path.name}") from exc
        candidate = load_manifest(candidate_path, candidate_day, destination)
        matches = [
            item
            for item in candidate["files"]
            if item.get("filename") == source.filename
        ]
        if len(matches) > 1:
            raise ValueError(
                f"duplicate references to {source.filename} in {candidate_path.name}"
            )
        if matches:
            if candidate_day not in allowed_manifest_days:
                raise ValueError(
                    f"nonadjacent manifest reference to {source.filename}: "
                    f"{candidate_path.name}"
                )
            references.append((candidate_path, matches[0]))

    if not references:
        raise RuntimeError(
            f"refusing unmanifested existing file with no trusted reference: {path}"
        )
    if len(references) != 1:
        names = ", ".join(candidate.name for candidate, _ in references)
        raise RuntimeError(
            f"ambiguous trusted references for {source.filename}: {names}"
        )

    reference_path, entry = references[0]
    expected_identity = {
        "kind": source.kind,
        "label_day": source.label_day.isoformat(),
        "filename": source.filename,
        "member_name": source.member_name,
        "url": source.url,
    }
    for key, value in expected_identity.items():
        if entry.get(key) != value:
            raise ValueError(
                f"shared-file identity mismatch in {reference_path.name}: {key}"
            )
    if not metadata_matches(entry.get("remote", {}), metadata):
        raise RuntimeError(
            f"shared-file remote metadata mismatch in {reference_path.name}"
        )
    recorded_digest = entry.get("sha256")
    if (
        not isinstance(recorded_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", recorded_digest) is None
    ):
        raise ValueError(f"invalid shared-file SHA-256 in {reference_path.name}")
    if path.stat().st_size != metadata.content_length:
        raise RuntimeError(f"shared file has the wrong size: {path}")
    actual_digest = sha256_path(path)
    if actual_digest != recorded_digest:
        raise RuntimeError(f"shared-file SHA-256 mismatch: {path}")

    reused = {
        key: value
        for key, value in entry.items()
        if key not in {"reused_at_utc", "reused_from_manifest"}
    }
    reused["validated_at_utc"] = datetime.now(timezone.utc).isoformat()
    reused["reused_at_utc"] = reused["validated_at_utc"]
    reused["reused_from_manifest"] = reference_path.name
    return reused


def partial_paths(path: Path) -> tuple[Path, Path]:
    part = path.with_name(path.name + ".part")
    return part, part.with_name(part.name + ".json")


def download(source: SourceFile, metadata: RemoteMetadata, path: Path) -> str:
    part, state_path = partial_paths(path)
    expected_state = {
        "schema": MANIFEST_SCHEMA,
        "url": source.url,
        "filename": source.filename,
        "remote": asdict(metadata),
    }
    offset = 0
    if state_path.exists():
        if not state_path.is_file():
            raise RuntimeError(f"partial state is not a file: {state_path}")
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid partial-download state: {state_path}") from exc
        if state != expected_state:
            raise RuntimeError(f"partial metadata changed; refusing append: {part}")
        if part.exists():
            if not part.is_file():
                raise RuntimeError(f"partial path is not a file: {part}")
            offset = part.stat().st_size
            if offset > metadata.content_length:
                raise RuntimeError(f"partial file exceeds remote length: {part}")
    elif part.exists():
        raise RuntimeError(f"partial file has no trusted metadata sidecar: {part}")
    else:
        write_json_atomic(state_path, expected_state)

    digest = hashlib.sha256()
    if offset:
        with part.open("rb") as existing:
            for chunk in iter(lambda: existing.read(CHUNK_SIZE), b""):
                digest.update(chunk)
    if offset == metadata.content_length:
        return digest.hexdigest()

    headers = {"User-Agent": USER_AGENT}
    mode = "wb"
    expected_status = 200
    if offset:
        headers["Range"] = f"bytes={offset}-"
        if metadata.etag:
            headers["If-Range"] = metadata.etag
        elif metadata.last_modified:
            headers["If-Range"] = metadata.last_modified
        mode = "ab"
        expected_status = 206
    request = Request(source.url, headers=headers)
    try:
        with urlopen(request, timeout=120) as response:  # noqa: S310 - fixed HTTPS hosts
            if response.status != expected_status:
                raise RuntimeError(
                    f"unsafe resume response HTTP {response.status}; expected {expected_status}"
                )
            if offset:
                content_range = response.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                if not match or int(match.group(1)) != offset:
                    raise RuntimeError(f"unexpected Content-Range: {content_range!r}")
                if int(match.group(3)) != metadata.content_length:
                    raise RuntimeError(
                        f"remote size changed during resume: {content_range!r}"
                    )
            with part.open(mode) as output:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                output.flush()
                os.fsync(output.fileno())
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"download failed for {source.url}: {exc}") from exc

    if part.stat().st_size != metadata.content_length:
        raise RuntimeError(
            f"incomplete download for {source.filename}: "
            f"{part.stat().st_size} != {metadata.content_length}"
        )
    return digest.hexdigest()


def finalize_download(
    source: SourceFile,
    metadata: RemoteMetadata,
    destination: Path,
    day: date,
) -> tuple[Path, str]:
    path = destination / source.filename
    digest = download(source, metadata, path)
    part, state_path = partial_paths(path)
    if source.kind == "trades":
        validate_trade_archive(part, source)
    else:
        validate_depth_archive(part, source, day)
    os.replace(part, path)
    state_path.unlink()
    return path, digest


def acquire(day: date, destination: Path) -> None:
    today = datetime.now(timezone.utc).date()
    latest_safe_day = today - timedelta(days=3)
    if day < MINIMUM_DAY:
        raise ValueError(f"depth-400 starts on {MINIMUM_DAY.isoformat()}")
    if day > latest_safe_day:
        raise ValueError(
            f"{day} is inside OKX's publication lag; latest safe UTC day is {latest_safe_day}"
        )

    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / f"okx-{SYMBOL}-{day.isoformat()}-manifest.json"
    manifest = load_manifest(manifest_path, day, destination)
    for source in build_sources(day):
        print(f"Checking {source.filename}")
        metadata = remote_metadata(source)
        path = destination / source.filename
        if completed_entry(manifest, source, metadata, path) is not None:
            print(f"Verified existing {path}")
            continue
        if path.exists():
            reused = reuse_from_trusted_manifest(
                current_manifest_path=manifest_path,
                destination=destination,
                source=source,
                metadata=metadata,
                path=path,
            )
            manifest["files"].append(reused)
            manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json_atomic(manifest_path, manifest)
            print(
                f"Reused verified {path} from {reused['reused_from_manifest']} "
                f"sha256={reused['sha256']}"
            )
            continue
        path, digest = finalize_download(source, metadata, destination, day)
        entry = {
            "kind": source.kind,
            "label_day": source.label_day.isoformat(),
            "filename": source.filename,
            "member_name": source.member_name,
            "url": source.url,
            "remote": asdict(metadata),
            "sha256": digest,
            "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest["files"].append(entry)
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(manifest_path, manifest)
        print(f"Validated {path} sha256={digest}")


def plan(day: date, destination: Path) -> None:
    start_ms, end_ms = utc_bounds_ms(day)
    payload = {
        "destination": str(destination),
        "source_page": SOURCE_PAGE,
        "symbol": SYMBOL,
        "utc_day": day.isoformat(),
        "utc_interval_ms": {"start_inclusive": start_ms, "end_exclusive": end_ms},
        "note": "Trade labels use UTC+08:00, so UTC day D requires D and D+1.",
        "sources": [
            {
                **asdict(source),
                "label_day": source.label_day.isoformat(),
            }
            for source in build_sources(day)
        ],
    }
    print(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download official OKX BTC-USDT trades and depth-400 archives needed "
            "for one UTC day."
        )
    )
    parser.add_argument("day", type=parse_day, help="UTC day in YYYY-MM-DD format")
    parser.add_argument(
        "--destination",
        type=Path,
        required=True,
        help=r"explicit output directory beneath C:\DaxAlgoData",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="print URLs and UTC bounds without network access or filesystem writes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        destination = validate_destination(args.destination)
        if args.plan:
            plan(args.day, destination)
        else:
            acquire(args.day, destination)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
