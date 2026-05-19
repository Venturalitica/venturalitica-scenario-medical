"""Cross-model comparison report for the VertebraSeg AI scenario.

Given two `run_compliance_suite()` outputs — typically MONAI's
`wholeBody_ct_segmentation` (baseline) and TotalSegmentator v2
(challenger) — emits a side-by-side markdown report so a reviewer can
see at a glance which controls each model passes / fails on the same
TCIA cohort under the same OSCAL policy.

The report is consumed by:
  * the `--scenario compare` orchestrator (main.py)
  * (future) the Streamlit dashboard's Phase 4 Technical Report tab
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple


CompareTuple = Optional[Tuple[list, list, object]]  # (data_results, model_results, system_desc)


def _index_results(results: Iterable) -> dict:
    """Index a flat list of ComplianceResult by control_id for diffing."""
    return {r.control_id: r for r in (results or [])}


def _verdict_icon(r) -> str:
    return "✅" if r.passed else "❌"


def _format_value(r) -> str:
    try:
        return f"{r.actual_value:.4f}"
    except (TypeError, ValueError):
        return str(r.actual_value)


def emit_comparison_report(
    *,
    monai: CompareTuple,
    totalseg: CompareTuple,
    output_path: Path,
) -> Path:
    """Write a markdown side-by-side comparison of two audit runs.

    Either input may be `None` (the corresponding audit failed); in that
    case the column is filled with `—` and the report calls out the gap.
    """
    monai_data, monai_model, monai_sd = monai if monai else ([], [], None)
    ts_data, ts_model, ts_sd = totalseg if totalseg else ([], [], None)

    monai_all = (monai_data or []) + (monai_model or [])
    ts_all = (ts_data or []) + (ts_model or [])

    monai_by_id = _index_results(monai_all)
    ts_by_id = _index_results(ts_all)

    all_ids = sorted(set(monai_by_id) | set(ts_by_id))

    monai_pass = sum(1 for r in monai_all if r.passed)
    monai_fail = sum(1 for r in monai_all if not r.passed)
    ts_pass = sum(1 for r in ts_all if r.passed)
    ts_fail = sum(1 for r in ts_all if not r.passed)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Cross-Model Compliance Comparison\n\n")
        f.write(
            "Same TCIA Spine-Mets-CT-SEG cohort, same Venturalítica OSCAL\n"
            "policy, two different vertebra-segmentation models scored\n"
            "back-to-back through `vl.enforce()`.\n\n"
        )

        # ── Executive summary ────────────────────────────────────────────
        f.write("## Executive summary\n\n")
        f.write("| Model | Passed | Failed | Verdict |\n")
        f.write("|---|---:|---:|---|\n")
        f.write(
            f"| MONAI `wholeBody_ct_segmentation` (baseline) | {monai_pass} | "
            f"{monai_fail} | {'✅ COMPLIANT' if monai_fail == 0 else '❌ NON-COMPLIANT'} |\n"
        )
        f.write(
            f"| TotalSegmentator v2 (challenger) | {ts_pass} | {ts_fail} | "
            f"{'✅ COMPLIANT' if ts_fail == 0 else '❌ NON-COMPLIANT'} |\n\n"
        )

        # ── Side-by-side table ───────────────────────────────────────────
        f.write("## Per-control comparison\n\n")
        f.write(
            "| Control | Threshold | MONAI value | MONAI | TotalSeg value | TotalSeg | "
            "Δ (TS - MONAI) |\n"
        )
        f.write("|---|---|---:|---|---:|---|---:|\n")

        flips = []
        for cid in all_ids:
            m = monai_by_id.get(cid)
            t = ts_by_id.get(cid)

            # threshold + operator are policy-level — same on both sides
            ref = m or t
            threshold_cell = f"`{ref.operator} {ref.threshold}`" if ref else "—"

            m_val = _format_value(m) if m else "—"
            t_val = _format_value(t) if t else "—"
            m_icon = _verdict_icon(m) if m else "—"
            t_icon = _verdict_icon(t) if t else "—"

            delta = "—"
            if m and t:
                try:
                    d = float(t.actual_value) - float(m.actual_value)
                    delta = f"{d:+.4f}"
                except (TypeError, ValueError):
                    delta = "—"

            if m and t and m.passed != t.passed:
                flips.append((cid, m.passed, t.passed))

            f.write(
                f"| `{cid}` | {threshold_cell} | {m_val} | {m_icon} | "
                f"{t_val} | {t_icon} | {delta} |\n"
            )
        f.write("\n")

        # ── Flips highlighted ────────────────────────────────────────────
        if flips:
            f.write("## Verdict flips (model A passes where model B fails)\n\n")
            for cid, m_passed, t_passed in flips:
                arrow = "MONAI ❌ → TotalSeg ✅" if t_passed else "MONAI ✅ → TotalSeg ❌"
                f.write(f"- `{cid}` — {arrow}\n")
            f.write("\n")
        else:
            f.write("_No verdict flips: both models pass / fail the same controls._\n\n")

        # ── Notes ─────────────────────────────────────────────────────────
        f.write("## Notes\n\n")
        f.write(
            "- Both models are scored against the **same** OSCAL\n"
            "  `component-definition` policy (`shared_data/policies/`).\n"
            "  The Venturalítica SDK does not see which model produced the\n"
            "  Dice numbers; controls are evaluated identically.\n"
        )
        f.write(
            "- TotalSegmentator v2 reports a `Confidence` proxy "
            "(predicted-spine fraction) rather than a softmax-derived\n"
            "  per-voxel probability, so the `safety-calibration` Pearson\n"
            "  correlation is less informative than for MONAI.\n"
        )
        f.write(
            "- The MONAI ↔ TotalSegmentator vertebra-label mapping lives in\n"
            "  `base_medical/totalseg_evaluation.py::_MONAI_TO_TS`. The TS\n"
            "  prediction is restricted to the levels actually annotated in\n"
            "  the TCIA SEG before Dice is computed — fair-comparison\n"
            "  prerequisite.\n"
        )

    print(f"\n  ✓ Comparison report written to {output_path}")
    return output_path
