"""Every paid call goes through `call`: budget check, request logged to CASE/fal_calls.jsonl with
an estimated cost, retries, and a hard failure if the endpoint fails (a stage never substitutes a
stand-in for a failed call). FAL_KEY is read from the environment by fal_client and never printed.

Endpoints (fal.ai):
  fal-ai/nano-banana-2/edit   the photoreal target of the fixed view (grey render in, photo out)
  fal-ai/marigold-v2          albedo and camera-space normals of the target
  fal-ai/patina               metalness / height (its roughness follows lighting on whole scenes)
  microsoft/moge-3            metric depth and normals of the target, with the known field of view
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

from . import common

NANO_BANANA_2_EDIT = "fal-ai/nano-banana-2/edit"
MARIGOLD_V2 = "fal-ai/marigold-v2"
PATINA = "fal-ai/patina"
MOGE_3 = "microsoft/moge-3"

# (billed unit, estimated USD per unit) for the budget guard; check the fal dashboard for exact billing
PRICES_USD = {
    NANO_BANANA_2_EDIT: ("image", 0.08),
    MARIGOLD_V2: ("image", 0.03),
    PATINA: ("megapixel", 0.01),
    MOGE_3: ("megapixel", 0.015),
}
DEFAULT_BUDGET_USD = 4.0

_upload_cache: dict[str, str] = {}


class BudgetError(RuntimeError):
    pass


def have_key() -> bool:
    return bool(os.environ.get("FAL_KEY") or os.environ.get("FAL_KEY_ID"))


def _client():
    try:
        import fal_client
    except ImportError as e:
        raise SystemExit("fal-client is not installed: pip install fal-client") from e
    if not have_key():
        raise SystemExit("FAL_KEY is not set: export it (it is never printed or written to files)")
    return fal_client


def _log_path(case) -> Path:
    return common.case_dir(case) / "fal_calls.jsonl"


def spent(case) -> float:
    """Estimated USD of the successful calls logged for this case."""
    p = _log_path(case)
    if not p.exists():
        return 0.0
    total = 0.0
    for line in p.read_text().splitlines():
        e = json.loads(line)
        if e.get("ok"):
            total += float(e.get("est_cost_usd", 0.0))
    return round(total, 4)


def summary(case) -> dict:
    """calls and estimated USD per endpoint"""
    out: dict[str, dict] = {}
    p = _log_path(case)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        e = json.loads(line)
        if not e.get("ok"):
            continue
        s = out.setdefault(e["endpoint"], {"calls": 0, "est_cost_usd": 0.0})
        s["calls"] += 1
        s["est_cost_usd"] = round(s["est_cost_usd"] + e.get("est_cost_usd", 0.0), 4)
    return out


def upload(path) -> str:
    """Upload a local file to fal storage and return its URL (cached per path + mtime)."""
    path = Path(path)
    key = f"{path.resolve()}:{path.stat().st_mtime_ns}"
    if key not in _upload_cache:
        _upload_cache[key] = _client().upload_file(path)
    return _upload_cache[key]


def download(url: str, dest, tries: int = 3) -> Path:
    """Fetch a result file; only https links are followed."""
    if urllib.parse.urlparse(str(url)).scheme != "https":
        raise ValueError(f"refusing to download a non-https link: {str(url)[:80]}")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "blockout-to-real"})
            with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
                f.write(r.read())
            return dest
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))
    return dest


def call(case, endpoint: str, arguments: dict, note: str = "", units: float = 1.0, retries: int = 2) -> dict:
    """Run a fal endpoint and return its JSON result.

    `units` is the billed quantity for the estimate (images, or megapixels for per-megapixel
    endpoints). Raises BudgetError before spending past case.json settings.budget_usd, and
    RuntimeError after `retries` failed attempts.
    """
    client = _client()
    unit, price = PRICES_USD.get(endpoint, ("call", 0.05))
    est = round(price * units, 4)
    budget = float(common.load_case(case).get("settings", {}).get("budget_usd", DEFAULT_BUDGET_USD))
    already = spent(case)
    if already + est > budget + 1e-9:
        raise BudgetError(f"fal budget ${budget:.2f} would be exceeded by {note or endpoint} (${already:.2f} spent, "
                          f"est. ${est:.2f}). Raise settings.budget_usd in case.json if intended.")
    log = _log_path(case)
    last_err = None
    for attempt in range(retries + 1):
        t = time.time()
        entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "endpoint": endpoint, "note": note,
                 "attempt": attempt, "arguments": arguments}
        try:
            result = client.subscribe(endpoint, arguments=arguments, with_logs=False)
            entry.update(ok=True, seconds=round(time.time() - t, 2), billed_unit=unit, units=units, est_cost_usd=est)
            with open(log, "a") as f:
                f.write(json.dumps(entry) + "\n")
            return result
        except Exception as e:  # noqa: BLE001 - logged, then re-raised below
            last_err = e
            entry.update(ok=False, seconds=round(time.time() - t, 2), error=f"{type(e).__name__}: {e}"[:500])
            with open(log, "a") as f:
                f.write(json.dumps(entry) + "\n")
            if attempt < retries:
                time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"fal call to {endpoint} failed after {retries + 1} attempts: {last_err}")
