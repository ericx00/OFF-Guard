"""
Real OFAC SDN sanctions-list screening.

Downloads and caches the public OFAC Specially Designated Nationals (SDN)
list, then does fuzzy name matching against it. This is real, public
government sanctions data — not synthetic test data.

Data source: U.S. Treasury Office of Foreign Assets Control (OFAC)
https://ofac.treasury.gov/sanctions-list-service

IMPORTANT: this is a demo-grade screening helper, not a compliance system.
Fuzzy name matching produces false positives and false negatives. Real KYC
programs use OFAC's own Sanctions List Search / SLS tooling, review matches
manually, and match on more than just name (DOB, address, program, aliases).
Treat any "match" here as a prompt to review, never as a verdict.
"""

import csv
import time
from pathlib import Path
from typing import NamedTuple

import requests
from rapidfuzz import fuzz, process, utils

# Treasury has used both hosts for this file over the years; try in order.
SDN_CSV_URLS = [
    "https://ofac.treasury.gov/downloads/sdn.csv",
    "https://www.treasury.gov/ofac/downloads/sdn.csv",
]

CACHE_PATH = Path(__file__).parent / "data" / "sdn_cache.csv"
CACHE_MAX_AGE_SECONDS = 24 * 60 * 60  # refresh at most once a day


class SanctionsMatch(NamedTuple):
    name: str
    score: float
    program: str
    entity_number: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "score": round(self.score, 1),
            "program": self.program,
            "entity_number": self.entity_number,
        }


def _cache_is_fresh() -> bool:
    if not CACHE_PATH.exists():
        return False
    age = time.time() - CACHE_PATH.stat().st_mtime
    return age < CACHE_MAX_AGE_SECONDS


def ensure_sdn_list(force: bool = False) -> Path:
    """Download the SDN list if we don't already have a fresh local copy."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not force and _cache_is_fresh():
        return CACHE_PATH

    last_error = None
    for url in SDN_CSV_URLS:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            CACHE_PATH.write_bytes(response.content)
            return CACHE_PATH
        except Exception as exc:  # noqa: BLE001 - we want to try every URL
            last_error = exc
            continue

    if CACHE_PATH.exists():
        # Downloads failed but we have a (possibly stale) cached copy — use it
        # rather than hard-failing the whole app.
        return CACHE_PATH

    raise RuntimeError(
        "Could not download the OFAC SDN list from any known URL, and no "
        "cached copy exists. Check your internet connection, or manually "
        f"download the CSV and save it to {CACHE_PATH}. Last error: {last_error}"
    )


_SDN_RECORDS: list[dict] | None = None


def _load_records(force_refresh: bool = False) -> list[dict]:
    global _SDN_RECORDS
    if _SDN_RECORDS is not None and not force_refresh:
        return _SDN_RECORDS

    path = ensure_sdn_list(force=force_refresh)
    records = []
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            # Classic sdn.csv has no header row; columns are:
            # ent_num, SDN_Name, SDN_Type, Program, Title, Call_Sign, ...
            if len(row) < 4:
                continue
            records.append({
                "entity_number": row[0].strip(),
                "name": row[1].strip(),
                "type": row[2].strip(),
                "program": row[3].strip(),
            })

    if not records:
        raise RuntimeError(
            f"Parsed 0 records from {path} — the file may be empty, "
            "corrupted, or in a different format than expected. Try "
            "deleting the cache file and re-running to force a fresh download."
        )

    _SDN_RECORDS = records
    return records


def check_name(candidate_name: str, threshold: float = 82.0, limit: int = 5) -> list[SanctionsMatch]:
    """
    Fuzzy-match a candidate name against the real OFAC SDN list.
    Returns matches scoring at or above `threshold` (0-100 scale), highest first.
    """
    candidate_name = (candidate_name or "").strip()
    if not candidate_name:
        return []

    records = _load_records()
    names = [r["name"] for r in records]

    results = process.extract(
        candidate_name,
        names,
        scorer=fuzz.token_sort_ratio,
        processor=utils.default_process,
        limit=limit,
    )

    matches = []
    for matched_name, score, idx in results:
        if score >= threshold:
            record = records[idx]
            matches.append(SanctionsMatch(
                name=record["name"],
                score=score,
                program=record["program"],
                entity_number=record["entity_number"],
            ))
    return matches