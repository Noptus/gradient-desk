"""Build a reproducible supervised panel from Kenneth French's 12 industry portfolios.

This is a small, fully open numerical ML experiment: predict each industry's
*next observed trading-day* value-weighted return from features computable with
information available at the decision date's close. Nothing here embeds the raw
data; the source archive is downloaded by the caller, pinned by SHA-256, and
decoded only after that hash is verified.

Source
    Kenneth R. French data library, "12 Industry Portfolios" daily, the
    value-weighted (not equal-weighted) daily return table that leads the CSV.
    Archive URL and hash are pinned in ``ARCHIVE_URL`` / ``ARCHIVE_SHA256``.

Units
    The French CSV reports returns in **percentage points** (e.g. ``0.53`` means
    +0.53%). We keep percentage points throughout: ``features`` and ``targets``
    are in percent. Any later standardization is a *fit-only* step the caller
    applies downstream; this module emits raw natural-unit percent so the panel
    stays inspectable and leakage-free.

Missing data
    The French tables use sentinels (``-99.99`` and ``-999``) for unobserved
    cells. We parse those as ``nan`` and keep calendar rows intact (so trading
    days are never silently dropped during parsing), then, when building the
    panel, we *skip* any decision day whose lookback window, current row, or
    next-day target contains a non-finite value. No interpolation or bridging
    across gaps is performed.

Timing / labels
    ``Day.date`` is the decision date, evaluated at its close. Its ``features``
    use only rows with date ``<= t``. ``Day.label_date`` is the next *observed*
    trading date strictly after ``t`` (the immediately following panel row), and
    ``Day.targets`` are that row's realized per-industry returns in percent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import io
import math
import re
import zipfile
from pathlib import Path

# --------------------------------------------------------------------------- #
# Pinned source (data is never embedded; only the locator and digest are)
# --------------------------------------------------------------------------- #
ARCHIVE_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp_202412/"
    "12_Industry_Portfolios_daily_CSV.zip"
)
ARCHIVE_SHA256 = "d1571474ee0f6e1ff463f7fccc6c3cca74d6d18d0385b6a24c9c51498b714ea6"

#: Ordered feature names, all in percentage points (see module docstring).
FEATURES: tuple[str, ...] = (
    "own_return_t",          # own value-weighted return on date t
    "own_mean_5",            # own rolling mean over the last 5 rows inclusive of t
    "own_mean_20",           # own rolling mean over the last 20 rows inclusive of t
    "own_std_20",            # own rolling std over the last 20 rows inclusive, ddof=0
    "basket_return_t",       # equal-weight mean across the 12 industries on t
    "own_minus_basket_t",    # own_return_t - basket_return_t
)

N_INDUSTRIES = 12
_MEAN_5_WINDOW = 5
_MEAN_20_WINDOW = 20
_STD_20_WINDOW = 20

# Guard rails for a well-formed French daily table.
_MIN_ROWS = _STD_20_WINDOW + 1          # enough history for one usable Day
_MAX_ROWS = 100_000                     # ~400 trading years; rejects runaway input
_SENTINELS = {"-99.99", "-999"}
_DATE_LINE = re.compile(r"^\s*(\d{8})\s*,")
_VINTAGE = re.compile(r"ftp_(\d{6})/")


# --------------------------------------------------------------------------- #
# Public dataclass
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Day:
    """One supervised example keyed by the decision date.

    ``features`` is a tuple of 12 per-industry feature tuples (industry order
    matches the returned industry codes); each inner tuple follows ``FEATURES``.
    ``targets`` holds the 12 next-trading-day returns in percentage points.
    """

    date: date
    label_date: date
    features: tuple[tuple[float, ...], ...]
    targets: tuple[float, ...]


# --------------------------------------------------------------------------- #
# Archive integrity + decoding
# --------------------------------------------------------------------------- #
def _verify_hash(payload: bytes) -> None:
    digest = hashlib.sha256(payload).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError(
            "Archive SHA-256 mismatch: refusing to decode untrusted data.\n"
            f"  expected {ARCHIVE_SHA256}\n  actual   {digest}"
        )


def _read_csv_text(path: Path) -> str:
    """Verify the pinned hash, then extract the single CSV member as text."""
    with path.open("rb") as stream:
        payload = stream.read(2_000_001)
    if len(payload) > 2_000_000:
        raise ValueError("French ZIP exceeds the 2 MB archive limit")
    _verify_hash(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Expected exactly one CSV in archive, found {members!r}")
        member = archive.getinfo(members[0])
        if member.file_size > 8_000_000:
            raise ValueError("French CSV exceeds the 8 MB text limit")
        return archive.read(member).decode("latin-1")


def _vintage() -> str:
    """Best-effort database vintage, e.g. '202412', taken from the URL pattern."""
    match = _VINTAGE.search(ARCHIVE_URL)
    return match[1] if match else "unknown"


def _parse_header_codes(lines: list[str]) -> tuple[list[str], int]:
    """Find the value-weighted table header and return (codes, first_data_index).

    The French daily file leads with the value-weighted average daily returns
    table; its header is the first line that has blank first cell followed by the
    industry codes. We take the first such header (the value-weighted block).
    """
    for index, line in enumerate(lines):
        cells = [cell.strip() for cell in line.split(",")]
        if cells and cells[0] == "" and len(cells) == N_INDUSTRIES + 1:
            codes = cells[1:]
            if all(code and not code.isdigit() for code in codes):
                if len(set(codes)) != N_INDUSTRIES:
                    raise ValueError(f"Industry codes are not unique: {codes!r}")
                return codes, index + 1
    raise ValueError("Could not locate the 12-industry value-weighted header row")


def _parse_cell(token: str) -> float:
    token = token.strip()
    if token in _SENTINELS or token == "":
        return math.nan
    try:
        return float(token)
    except ValueError as exc:
        raise ValueError(f"Non-numeric return cell: {token!r}") from exc


def parse_archive(path: Path) -> tuple[str, list[str], list[tuple[date, tuple[float, ...]]]]:
    """Verify, decode, and parse the pinned French archive.

    Returns ``(vintage, industry_codes, rows)`` where ``rows`` are the daily
    value-weighted returns (percentage points) for the leading table, sorted by
    date and deduplicated. Sentinel cells are kept as ``nan`` so no trading date
    is dropped here; ``build_panel`` enforces completeness per decision day.
    Reading stops at the first blank line after the value-weighted block, which
    separates it from the equal-weighted and firm-count tables.
    """
    text = _read_csv_text(path)
    lines = text.splitlines()
    codes, start = _parse_header_codes(lines)

    rows: list[tuple[date, tuple[float, ...]]] = []
    seen: set[date] = set()
    for line in lines[start:]:
        if not line.strip():
            break  # end of the value-weighted daily block
        match = _DATE_LINE.match(line)
        if not match:
            break  # a non-date line marks a new section / annotation
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) != N_INDUSTRIES + 1:
            raise ValueError(f"Expected {N_INDUSTRIES + 1} columns, got {len(cells)}: {line!r}")
        stamp = match[1]
        day = date(int(stamp[0:4]), int(stamp[4:6]), int(stamp[6:8]))
        if day in seen:
            raise ValueError(f"Duplicate date in archive: {day.isoformat()}")
        seen.add(day)
        rows.append((day, tuple(_parse_cell(cell) for cell in cells[1:])))
        if len(rows) > _MAX_ROWS:
            raise ValueError(f"Archive exceeds bounded length of {_MAX_ROWS} rows")

    if len(rows) < _MIN_ROWS:
        raise ValueError(f"Too few daily rows ({len(rows)}); need at least {_MIN_ROWS}")
    rows.sort(key=lambda item: item[0])
    return _vintage(), codes, rows


# --------------------------------------------------------------------------- #
# Panel construction (leakage-safe windows, no gap bridging)
# --------------------------------------------------------------------------- #
def _finite_window(values: list[float]) -> bool:
    return all(math.isfinite(v) for v in values)


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _pstd(values: list[float]) -> float:
    mean = _mean(values)
    var = math.fsum((v - mean) ** 2 for v in values) / len(values)  # ddof=0
    return math.sqrt(var)


def build_panel(
    raw_rows: list[tuple[date, tuple[float, ...]]],
) -> list[Day]:
    """Turn sorted daily rows into leakage-safe supervised ``Day`` examples.

    For each decision index ``t`` (needing a full 20-row lookback and a next
    row ``t+1``) we build 12 per-industry feature tuples from rows ``<= t`` and
    label with the realized returns of the immediately following row. A ``Day``
    is skipped entirely if any value in its lookback window, its current row, or
    its next-day targets is non-finite (missing sentinel); windows are never
    bridged across such gaps.
    """
    if len(raw_rows) < _MIN_ROWS:
        raise ValueError(f"Need at least {_MIN_ROWS} rows to build a panel")

    columns = [[row[1][i] for row in raw_rows] for i in range(N_INDUSTRIES)]
    panel: list[Day] = []

    for t in range(_STD_20_WINDOW - 1, len(raw_rows) - 1):
        decision_day, _ = raw_rows[t]
        label_day, next_values = raw_rows[t + 1]

        if not _finite_window(list(next_values)):
            continue

        per_industry: list[tuple[float, ...]] = []
        complete = True

        # Current-row returns (needed for the basket) must be finite for all 12.
        current = [columns[i][t] for i in range(N_INDUSTRIES)]
        if not _finite_window(current):
            continue
        basket = _mean(current)

        for i in range(N_INDUSTRIES):
            win5 = columns[i][t - _MEAN_5_WINDOW + 1 : t + 1]
            win20 = columns[i][t - _STD_20_WINDOW + 1 : t + 1]
            if not (_finite_window(win5) and _finite_window(win20)):
                complete = False
                break
            own_t = columns[i][t]
            per_industry.append((
                own_t,
                _mean(win5),
                _mean(win20),
                _pstd(win20),
                basket,
                own_t - basket,
            ))

        if not complete:
            continue

        panel.append(
            Day(
                date=decision_day,
                label_date=label_day,
                features=tuple(per_industry),
                targets=tuple(next_values),
            )
        )

    return panel
