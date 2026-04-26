"""
Module: verify_sources.py
=========================
CLI that live-checks every entry in ``constants.OFFICIAL_MIRRORS`` and
reports which alternative download sources are reachable.

Run:
    python -m dataset_sorter.verify_sources                  # all archs
    python -m dataset_sorter.verify_sources flux flux2 sd35  # selected
    python -m dataset_sorter.verify_sources --json           # machine-readable

Each mirror is HEAD-checked (no body downloaded) so the whole sweep takes
~30 seconds for the full 16-arch table on a normal connection. Mirrors
returning 401/403 are reported as "auth required" — this is normal for
HF gated repos (Flux dev, SD3.x) and isn't a failure.

Output columns:
    arch          architecture id
    source        huggingface | civitai | modelscope | github | url
    id            source-specific identifier
    status        OK / AUTH / 404 / unreachable / blocked
    notes         per-mirror caveats from the registry
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from dataset_sorter.constants import OFFICIAL_MIRRORS
from dataset_sorter.model_sources import (
    resolve_mirror_to_url,
    verify_source_reachable,
)

log = logging.getLogger(__name__)


def _check_one(arch: str, entry: dict) -> dict[str, Any]:
    """Probe a single mirror and return a result row."""
    url = resolve_mirror_to_url(entry)
    result = {
        "arch": arch,
        "source": entry.get("source", "?"),
        "id": entry.get("id", ""),
        "variant": entry.get("variant"),
        "gated": bool(entry.get("gated", False)),
        "note": entry.get("note", ""),
        "url": url,
        "status": "skip",
        "code": None,
    }
    if url is None:
        result["status"] = "no-url"
        return result
    reachable, code = verify_source_reachable(url, timeout=10.0)
    result["code"] = code
    if reachable and code is not None and code < 400:
        result["status"] = "OK"
    elif code in (401, 403):
        result["status"] = "AUTH" if entry.get("gated") else "AUTH (unexpected)"
    elif code == 404:
        result["status"] = "404"
    elif code is None:
        result["status"] = "unreachable"
    else:
        result["status"] = f"HTTP {code}"
    return result


def _format_row(r: dict[str, Any], col_widths: dict[str, int]) -> str:
    status = r["status"]
    # Lightweight status colouring via ANSI when stdout is a tty.
    if sys.stdout.isatty():
        if status == "OK":
            status = f"\033[32m{status:<14}\033[0m"
        elif status.startswith("AUTH"):
            status = f"\033[33m{status:<14}\033[0m"
        elif status in ("404", "unreachable"):
            status = f"\033[31m{status:<14}\033[0m"
        else:
            status = f"{status:<14}"
    else:
        status = f"{status:<14}"
    return (
        f"  {r['arch']:<{col_widths['arch']}}  "
        f"{r['source']:<{col_widths['source']}}  "
        f"{(r['id'] or '-'):<{col_widths['id']}}  "
        f"{status}  "
        f"{(r['note'] or '')}"
    )


def run(targets: list[str] | None = None, *, json_output: bool = False,
        max_workers: int = 8) -> int:
    """Run the verification sweep. Returns process exit code."""
    targets = targets or list(OFFICIAL_MIRRORS.keys())
    unknown = [a for a in targets if a not in OFFICIAL_MIRRORS]
    if unknown:
        print(f"Unknown architectures: {', '.join(unknown)}", file=sys.stderr)
        print(f"Known: {', '.join(sorted(OFFICIAL_MIRRORS))}", file=sys.stderr)
        return 2

    jobs: list[tuple[str, dict]] = []
    for arch in targets:
        for entry in OFFICIAL_MIRRORS[arch]:
            jobs.append((arch, entry))

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_check_one, a, e): (a, e) for a, e in jobs}
        for fut in as_completed(futures):
            results.append(fut.result())

    # Stable order: arch then preserve registry order within each arch.
    arch_order = {a: i for i, a in enumerate(targets)}
    results.sort(key=lambda r: (
        arch_order.get(r["arch"], 999),
        # preserve the original within-arch order via the registry index
        next((i for i, e in enumerate(OFFICIAL_MIRRORS[r["arch"]])
              if e.get("id") == r["id"] and e.get("source") == r["source"]),
             0),
    ))

    if json_output:
        json.dump(results, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        col_widths = {
            "arch": max(len("arch"), max((len(r["arch"]) for r in results), default=4)),
            "source": max(len("source"), max((len(r["source"]) for r in results), default=6)),
            "id": min(60, max(len("id"), max((len(r["id"]) for r in results), default=2))),
        }
        header = (
            f"  {'arch':<{col_widths['arch']}}  "
            f"{'source':<{col_widths['source']}}  "
            f"{'id':<{col_widths['id']}}  "
            f"{'status':<14}  notes"
        )
        print(header)
        print("  " + "─" * (len(header) - 2))
        for r in results:
            print(_format_row(r, col_widths))

        # Summary
        total = len(results)
        ok = sum(1 for r in results if r["status"] == "OK")
        auth = sum(1 for r in results if r["status"].startswith("AUTH"))
        gone = sum(1 for r in results if r["status"] in ("404", "unreachable"))
        print()
        print(f"  {ok}/{total} reachable, {auth} auth-required, {gone} dead")

    # Exit code: 0 if no unexpected failures, 1 if any mirror is dead OR
    # a non-gated mirror returned auth-required.
    bad = [r for r in results if r["status"] in (
        "404", "unreachable") or "unexpected" in r["status"]]
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dataset_sorter.verify_sources",
        description="Live-check every mirror in OFFICIAL_MIRRORS.",
    )
    parser.add_argument(
        "archs", nargs="*",
        help="Architectures to check (default: all). E.g. flux sd35 zimage.",
    )
    parser.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Emit machine-readable JSON instead of the table view.",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Parallel HEAD requests (default: 8).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return run(args.archs or None,
               json_output=args.json_output,
               max_workers=args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
