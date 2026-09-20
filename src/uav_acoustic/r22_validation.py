"""Guided R2.2 evidence-rich, external-phone hardware validation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from .coordinates import az_el_to_unit
from .doa.srp_phat import estimate_srp_phat
from .io.capture_file import load_capture
from .preflight import run_preflight
from .round2 import TestCase, _capture_and_analyze, _resolve_stimulus, _validate_stimulus, _write_json


BANDS = [(1000, 1500), (1000, 2000), (1200, 2000), (2000, 3000), (2000, 5000), (1000, 5000)]
PLAYBACK_MODE = "EXTERNAL_PHONE"
SOURCE_ORIENTATION = "FACING_ARRAY_CENTER"


@dataclass
class R22Case:
    test_id: str
    group: str
    repeat: int
    intended_az_deg: float | None
    intended_el_deg: float | None
    nominal_distance_m: float | None
    question: str
    background: bool = False
    optional: bool = False
    orientation: str = SOURCE_ORIENTATION


def evidence_rich_matrix(include_phone_directivity: bool = False) -> list[R22Case]:
    cases = [R22Case("background_start", "background_start", 1, None, None, None,
                     "Q7: establish starting background/no-target baseline", background=True)]
    for az in (-60, -45, -30, 0, 30, 45, 60):
        for repeat in (1, 2):
            cases.append(R22Case(f"az_2m_{az:+d}_r{repeat}", "azimuth_2m", repeat,
                                 float(az), 0.0, 2.0,
                                 "Q1/Q2/Q6/Q8: transition, symmetry, fix replication and band dependence"))
    for az in (-60, -30, 0, 30, 60):
        cases.append(R22Case(f"az_far_{az:+d}_r1", "azimuth_far", 1, float(az), 0.0, 4.0,
                             "Q3: controlled distance intervention for edge compression"))
    for el in (-15, 0, 15, 30):
        for repeat in (1, 2):
            cases.append(R22Case(f"el_2m_{el:+d}_r{repeat}", "elevation_2m", repeat,
                                 0.0, float(el), 2.0,
                                 "Q4/Q6/Q8: elevation sign, scale, repeatability and band dependence"))
    for el in (-15, 0, 15):
        cases.append(R22Case(f"el_far_{el:+d}_r1", "elevation_far", 1, 0.0, float(el), 3.0,
                             "Q5: practical elevation distance intervention"))
    if include_phone_directivity:
        cases.extend([
            R22Case("phone_directivity_facing_array", "phone_directivity", 1, 60.0, 0.0, 2.0,
                    "Q6: isolate phone loudspeaker directivity", optional=True),
            R22Case("phone_directivity_fixed_body", "phone_directivity", 2, 60.0, 0.0, 2.0,
                    "Q6: isolate phone loudspeaker directivity", optional=True,
                    orientation="FIXED_AS_ZERO_DEG_BODY_ORIENTATION"),
        ])
    cases.append(R22Case("background_end", "background_end", 1, None, None, None,
                         "Q7: detect background drift after the full round", background=True))
    return cases


def placement_coordinates(case: R22Case, distance_m: float, array_height_m: float) -> dict:
    az = float(case.intended_az_deg or 0.0)
    el = float(case.intended_el_deg or 0.0)
    if case.group.startswith("azimuth") or case.group == "phone_directivity":
        lateral = distance_m * math.sin(math.radians(az))
        forward = distance_m * math.cos(math.radians(az))
        delta_h = 0.0
    else:
        lateral = 0.0
        forward = distance_m * math.cos(math.radians(el))
        delta_h = distance_m * math.sin(math.radians(el))
    return {
        "expected_lateral_offset_m": lateral,
        "expected_forward_offset_m": forward,
        "relative_height_m": delta_h,
        "expected_source_height_m": array_height_m + delta_h,
    }


def placement_text(case: R22Case, ordinal: int, total: int, distance_m: float,
                   array_height_m: float) -> str:
    if case.background:
        return (f"\n{'='*72}\n[{ordinal}/{total}] {case.test_id}\n"
                "停止手机/音箱播放；本项不询问角度、距离或高度。\n"
                "确认测试声源静音后按 Enter。\n" + "="*72)
    p = placement_coordinates(case, distance_m, array_height_m)
    orientation = ("扬声器始终正对阵列中心" if case.orientation == SOURCE_ORIENTATION
                   else "手机机身保持与 0° 测试相同朝向（OPTIONAL 对照）")
    return (f"\n{'='*72}\n[{ordinal}/{total}] {case.test_id}\n"
            f"假设：{case.question}\n"
            f"目标 az={case.intended_az_deg:+g}°, el={case.intended_el_deg:+g}°, R={distance_m:.3f} m\n"
            f"左右偏移 X={p['expected_lateral_offset_m']:+.3f} m（左负右正）\n"
            f"正前距离 Z={p['expected_forward_offset_m']:.3f} m\n"
            f"相对高度 ΔH={p['relative_height_m']:+.3f} m\n"
            f"声源中心绝对高度={p['expected_source_height_m']:.3f} m\n"
            f"声源方向：{orientation}\n"
            "播放 broadband_1k_5k_60s.wav，声音稳定后再采集。\n"
            "DO NOT ADJUST PHYSICAL POSITION TO MATCH ESTIMATED ANGLE\n"
            "严禁根据屏幕 DOA 改变物理摆位。\n" + "="*72)


def quick_quality_check(result: dict, *, background: bool, expected_samples: int = 160000) -> dict:
    shape = result.get("audio_shape") or [0, 0]
    channel = result.get("channel_metrics") or {}
    frequency = np.asarray((result.get("spectrum") or {}).get("frequency_hz", []), float)
    magnitude = np.asarray((result.get("spectrum") or {}).get("mean_magnitude", []), float)
    in_band = (frequency >= 1000) & (frequency <= 5000)
    band_fraction = float(np.sum(magnitude[in_band]) / max(np.sum(magnitude), 1e-30)) if magnitude.size else 0.0
    median_rms = float(np.median(channel.get("rms") or [0.0]))
    maximum_clip = float(np.max(channel.get("clipping_fraction") or [1.0]))
    checks = {
        "channels_128": int(shape[0]) == 128,
        "correct_sample_count": int(shape[1]) == expected_samples,
        "not_clipping": maximum_clip <= 0.01,
        "not_silent": median_rms >= 1e-5,
        "broadband_energy_present": True if background else band_fraction >= 0.40,
    }
    return {"status": "PASS" if all(checks.values()) else "REPEAT_THIS_CAPTURE",
            "checks": checks, "median_rms": median_rms,
            "maximum_clipping_fraction": maximum_clip, "band_1k_5k_fraction": band_fraction}


def _ask_float(prompt: str, input_fn: Callable[[str], str], default: float | None = None) -> float:
    while True:
        raw = input_fn(prompt).strip()
        if not raw and default is not None:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("请输入数字。")


def _photo_prompt(group: str) -> None:
    labels = {
        "azimuth_2m": "2m 方位组", "azimuth_far": "远距离方位组",
        "elevation_2m": "2m 俯仰组", "elevation_far": "远距离俯仰组",
    }
    if group in labels:
        print(f"\n【{labels[group]}开始】如果方便，请拍一张能看到阵列、声源和周围墙面/桌面的照片。")
        print("照片不是算法输入；最后与结果 ZIP 一起返回。")


def _write_session_files(session: Path, rows: list[dict], metadata: dict) -> None:
    import csv
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (session / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    _write_json(session / "session_metadata.json", metadata)
    (session / "error.log").touch(exist_ok=True)
    lines = ["# R2.2 Evidence-Rich Validation", "", f"Captures planned: {len(metadata['validation_plan'])}",
             f"Captures recorded: {sum(bool(r.get('capture')) for r in rows)}", "",
             "Physical placement is ground truth; estimated DOA must never be used to move the source.", ""]
    for row in rows:
        lines.append(f"- {row['test_id']}: {row['outcome']} — {row['reason']}")
    (session / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _band_ablation(session: Path, rows: list[dict]) -> list[dict]:
    output = []
    targets = [r for r in rows if r.get("capture") and not r["background"]]
    for index, row in enumerate(targets, 1):
        frame, _, _ = load_capture(session / row["capture"])
        per_capture = []
        for lo, hi in BANDS:
            result = estimate_srp_phat(frame.audio, frame.fs, frame.mic_xyz, freq_range=(lo, hi),
                                       azimuth_grid_deg=np.arange(-90, 91, 2),
                                       elevation_grid_deg=np.arange(-60, 61, 2), max_pairs=512)
            item = {"test_id": row["test_id"], "band_hz": f"{lo}-{hi}",
                    "estimated_az_deg": result["azimuth_deg"], "estimated_el_deg": result["elevation_deg"],
                    "psr_db": result["diagnostics"]["peak_to_sidelobe_db"],
                    "confidence": result["confidence"]["score"],
                    "top_k": result["diagnostics"]["top_candidates"]}
            output.append(item); per_capture.append(item)
        _write_json(session / "analysis" / f"{row['test_id']}.band_ablation.json",
                     {"bands": per_capture, "source_capture": row["capture"]})
        print(f"Band analysis {index}/{len(targets)}: {row['test_id']}", flush=True)
    _write_json(session / "band_ablation_all.json", {"records": output})
    return output


def _aggregate(session: Path, rows: list[dict], bands: list[dict]) -> dict:
    truth = {r["test_id"]: r for r in rows}
    by_band = {}
    for band in sorted({r["band_hz"] for r in bands}):
        rr = [r for r in bands if r["band_hz"] == band]
        az_err = [abs(r["estimated_az_deg"] - truth[r["test_id"]]["intended_az_deg"])
                  for r in rr if truth[r["test_id"]]["group"] in {"azimuth_2m", "azimuth_far"}]
        el_err = [abs(r["estimated_el_deg"] - truth[r["test_id"]]["intended_el_deg"])
                  for r in rr if truth[r["test_id"]]["group"] in {"elevation_2m", "elevation_far"}]
        by_band[band] = {"azimuth_mae_deg": float(np.mean(az_err)),
                         "elevation_mae_deg": float(np.mean(el_err)),
                         "median_psr_db": float(np.median([r["psr_db"] for r in rr]))}
    best = min(by_band, key=lambda b: by_band[b]["azimuth_mae_deg"] + by_band[b]["elevation_mae_deg"])
    selected = {r["test_id"]: r for r in bands if r["band_hz"] == best}
    def group(name): return [r for r in rows if r["group"] == name and r["test_id"] in selected]
    az2, azfar, el2, elfar = group("azimuth_2m"), group("azimuth_far"), group("elevation_2m"), group("elevation_far")
    def errors(items, axis):
        key_est = f"estimated_{axis}_deg"; key_truth = f"intended_{axis}_deg"
        return [abs(selected[r["test_id"]][key_est] - r[key_truth]) for r in items]
    repeat_std = {}
    for axis, items in (("az", az2), ("el", el2)):
        key = f"intended_{axis}_deg"
        for value in sorted({r[key] for r in items}):
            estimates = [selected[r["test_id"]][f"estimated_{axis}_deg"] for r in items if r[key] == value]
            repeat_std[f"{axis}_{value:+g}"] = float(np.std(estimates))
    az_sign = all(r["intended_az_deg"] == 0 or np.sign(selected[r["test_id"]]["estimated_az_deg"])
                  == np.sign(r["intended_az_deg"]) for r in az2 if abs(r["intended_az_deg"]) in {30, 60})
    el_sign = all(r["intended_el_deg"] == 0 or np.sign(selected[r["test_id"]]["estimated_el_deg"])
                  == np.sign(r["intended_el_deg"]) for r in el2 if abs(r["intended_el_deg"]) in {15, 30})
    angle_transition = {}
    left_right = {}
    for angle in (30.0, 45.0, 60.0):
        negative = [selected[r["test_id"]]["estimated_az_deg"] for r in az2 if r["intended_az_deg"] == -angle]
        positive = [selected[r["test_id"]]["estimated_az_deg"] for r in az2 if r["intended_az_deg"] == angle]
        if negative and positive:
            neg_med, pos_med = float(np.median(negative)), float(np.median(positive))
            angle_transition[f"abs_{angle:g}"] = {
                "negative_median_estimate_deg": neg_med, "positive_median_estimate_deg": pos_med,
                "median_absolute_error_deg": float(np.median(
                    [abs(v + angle) for v in negative] + [abs(v - angle) for v in positive])),
            }
            left_right[f"abs_{angle:g}"] = {
                "absolute_magnitude_difference_deg": abs(abs(neg_med) - abs(pos_med)),
                "signed_center_offset_deg": (neg_med + pos_med) / 2.0,
            }
    el_negative = [selected[r["test_id"]]["estimated_el_deg"] for r in el2 if r["intended_el_deg"] == -15]
    el_positive = [selected[r["test_id"]]["estimated_el_deg"] for r in el2 if r["intended_el_deg"] == 15]
    elevation_symmetry = ({"minus15_median_deg": float(np.median(el_negative)),
                           "plus15_median_deg": float(np.median(el_positive)),
                           "absolute_magnitude_difference_deg": abs(abs(float(np.median(el_negative)))
                                                                    - abs(float(np.median(el_positive))))}
                          if el_negative and el_positive else {})
    background_records = []
    algorithm_records = []
    theoretical_observed = []
    for row in rows:
        if not row.get("analysis"):
            continue
        analysis = json.loads((session / row["analysis"]).read_text(encoding="utf-8"))
        srp = analysis.get("broadband_srp_phat") or {}
        vendor = analysis.get("vendor_compatible_das") or {}
        tdoa = analysis.get("independent_wavefront_fit") or {}
        if row["background"]:
            background_records.append({"test_id": row["test_id"], "status": analysis.get("status"),
                "srp_confidence": (srp.get("confidence") or {}).get("score"),
                "srp_psr_db": (srp.get("diagnostics") or {}).get("peak_to_sidelobe_db"),
                "band_1k_5k_fraction": (analysis.get("quick_quality_check") or {}).get("band_1k_5k_fraction")})
            continue
        algorithm_records.append({"test_id": row["test_id"],
            "srp_1k_5k": [srp.get("azimuth_deg"), srp.get("elevation_deg")],
            "vendor_das_1k_5k": [vendor.get("azimuth_deg"), vendor.get("elevation_deg")],
            "robust_tdoa_1k_2k": [tdoa.get("azimuth_deg"), tdoa.get("elevation_deg")],
            "selected_band_srp": [selected[row["test_id"]]["estimated_az_deg"],
                                  selected[row["test_id"]]["estimated_el_deg"]],
            "selected_band_psr_db": selected[row["test_id"]]["psr_db"],
            "selected_band_confidence": selected[row["test_id"]]["confidence"],
            "selected_band_top_k": selected[row["test_id"]]["top_k"]})
        pairs = tdoa.get("pairs") or []
        if pairs:
            frame, _, _ = load_capture(session / row["capture"])
            source = float(row["actual_distance_m"]) * az_el_to_unit(
                float(row["intended_az_deg"]), float(row["intended_el_deg"]))
            expected, observed = [], []
            for pair in pairs:
                i, j = int(pair["mic_i"]), int(pair["mic_j"])
                expected.append((np.linalg.norm(source-frame.mic_xyz[i])
                                 - np.linalg.norm(source-frame.mic_xyz[j])) / 343.0)
                observed.append(float(pair["tdoa_us"]) * 1e-6)
            expected, observed = np.asarray(expected), np.asarray(observed)
            identifiable = float(np.sqrt(np.mean(expected**2)) * 1e6) >= 1.0
            theoretical_observed.append({"test_id": row["test_id"],
                "observed_vs_theoretical_slope": (float(np.dot(expected, observed) /
                    max(np.dot(expected, expected), 1e-30)) if identifiable else None),
                "correlation": (float(np.corrcoef(expected, observed)[0, 1]) if identifiable else None),
                "rmse_us": float(np.sqrt(np.mean((expected-observed)**2)) * 1e6),
                "tdoa_fit_rms_residual_us": tdoa.get("rms_residual_us")})
    background_ok = (len(background_records) == 2 and all(
        r["status"] in {"NO_TARGET", "LOW_CONFIDENCE"} for r in background_records))
    directivity = [r for r in rows if r["group"] == "phone_directivity" and r["test_id"] in selected]
    directivity_comparison = ({r["test_id"]: selected[r["test_id"]] for r in directivity}
                              if directivity else {"status": "NOT_RUN_OPTIONAL"})
    az_gate_points = [r for r in az2 if abs(float(r["intended_az_deg"])) in {0.0, 30.0, 60.0}]
    el_gate_points = [r for r in el2 if float(r["intended_el_deg"]) in {-15.0, 0.0, 15.0, 30.0}]
    gate = {"best_band_hz": best, "band_summary": by_band,
            "gate_definition": {"azimuth_truth_deg": [-60, -30, 0, 30, 60],
                                "elevation_truth_deg": [-15, 0, 15, 30],
                                "median_absolute_error_limit_deg": 10,
                                "repeat_std_limit_deg": 5},
            "azimuth_median_absolute_error_deg": float(np.median(errors(az_gate_points, "az"))),
            "elevation_median_absolute_error_deg": float(np.median(errors(el_gate_points, "el"))),
            "azimuth_sign_correct": az_sign, "elevation_sign_correct": el_sign,
            "repeat_std_deg": repeat_std,
            "repeat_std_all_le_5deg": all(v <= 5 for v in repeat_std.values()),
            "angle_transition_30_45_60": angle_transition,
            "negative_positive_symmetry": left_right,
            "elevation_minus15_plus15_symmetry": elevation_symmetry,
            "background_start_end": background_records,
            "background_no_stable_high_quality_target": background_ok,
            "phone_directivity_control": directivity_comparison,
            "algorithm_consistency": algorithm_records,
            "theoretical_vs_observed_tdoa": theoretical_observed,
            "distance_comparison": {
                "azimuth_2m_mae_deg": float(np.mean(errors(az2, "az"))),
                "azimuth_far_mae_deg": float(np.mean(errors(azfar, "az"))),
                "elevation_2m_mae_deg": float(np.mean(errors(el2, "el"))),
                "elevation_far_mae_deg": float(np.mean(errors(elfar, "el"))),
            }}
    gate["pass"] = bool(az_sign and el_sign and background_ok and gate["azimuth_median_absolute_error_deg"] <= 10
                        and gate["elevation_median_absolute_error_deg"] <= 10
                        and gate["repeat_std_all_le_5deg"])
    _write_json(session / "session_comparisons.json", gate)
    _write_json(session / "validation_gate.json", gate)
    return gate


def run_r22_validation(session_root: Path, *, config_path: Path, dry_run: bool = False,
                       input_fn: Callable[[str], str] = input) -> Path:
    session = (Path(session_root) / f"R2.2_Evidence_Rich_Validation_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    for name in ("captures", "analysis", "failed_attempts", "screenshots"):
        (session / name).mkdir(parents=True, exist_ok=True)
    preflight = run_preflight(config_path, bind_udp=not dry_run)
    _write_json(session / "preflight.json", preflight)
    stimulus = _validate_stimulus(_resolve_stimulus(config_path, "broadband_1k_5k_60s.wav"))
    if dry_run:
        array_height, playback_device, room_note, volume = 1.2, "DRY_RUN_PHONE", "DRY_RUN", "FIXED"
        include_directivity = False
    else:
        print("\nR2.2 Evidence-Rich Validation — PLAYBACK_MODE=EXTERNAL_PHONE")
        print("电脑只采集/分析，不播放声音；阵列整轮不得移动、旋转或倾斜。")
        array_height = _ask_float("阵列中心离地高度 array_center_height_m: ", input_fn)
        playback_device = input_fn("播放设备（手机型号或手机+外部音箱）: ").strip() or "UNRECORDED"
        volume = input_fn("固定音量/EQ设置: ").strip() or "FIXED_UNRECORDED"
        room_note = input_fn("房间与墙面/桌面简要备注: ").strip() or "UNRECORDED"
        include_directivity = ("phone" in playback_device.lower() or "手机" in playback_device) and (
            input_fn("OPTIONAL 手机方向性 +60° 对照？Y/N [N]: ").strip().upper() == "Y")
    cases = evidence_rich_matrix(include_directivity)
    metadata = {"schema_version": 4, "mode": "R2.2_EVIDENCE_RICH_VALIDATION",
                "created_local": datetime.now().isoformat(), "playback_mode": PLAYBACK_MODE,
                "computer_audio_used": False, "playback_device": playback_device,
                "fixed_volume_eq": volume, "room_note": room_note,
                "array_center_height_m": array_height, "source_orientation": SOURCE_ORIENTATION,
                "stimulus": stimulus, "preflight_status": preflight["status"],
                "validation_plan": [asdict(c) for c in cases], "dry_run": dry_run,
                "coordinate_convention": "+Z front; left negative az; right positive az; up positive el"}
    rows: list[dict] = []
    group_distances: dict[str, float] = {}
    last_group = None
    for ordinal, case in enumerate(cases, 1):
        if case.group != last_group:
            _photo_prompt(case.group)
            last_group = case.group
        if case.background:
            distance = 0.0
        elif case.group not in group_distances:
            default = float(case.nominal_distance_m)
            distance = default if dry_run else _ask_float(
                f"{case.group} 实际直线距离 actual_distance_m [{default:.1f}]: ", input_fn, default)
            group_distances[case.group] = distance
        else:
            distance = group_distances[case.group]
        coords = ({"expected_lateral_offset_m": None, "expected_forward_offset_m": None,
                   "relative_height_m": None, "expected_source_height_m": None}
                  if case.background else placement_coordinates(case, distance, array_height))
        if dry_run:
            actual_height = coords["expected_source_height_m"]
        elif case.background:
            actual_height = None
            print(placement_text(case, ordinal, len(cases), 0.0, array_height))
            input_fn("确认测试声源静音后按 Enter 采集: ")
        else:
            print(placement_text(case, ordinal, len(cases), distance, array_height))
            actual_height = _ask_float(
                f"实测声源中心高度 actual_source_height_m [{coords['expected_source_height_m']:.3f}]: ",
                input_fn, coords["expected_source_height_m"])
            input_fn("确认物理标记与声音稳定后按 Enter 采集: ")
        rich = {"test_id": case.test_id, "repeat": case.repeat,
                "intended_az_deg": case.intended_az_deg, "intended_el_deg": case.intended_el_deg,
                "actual_distance_m": None if case.background else distance,
                "array_center_height_m": array_height,
                "expected_source_height_m": coords["expected_source_height_m"],
                "actual_source_height_m": actual_height,
                "expected_lateral_offset_m": coords["expected_lateral_offset_m"],
                "expected_forward_offset_m": coords["expected_forward_offset_m"],
                "playback_device": playback_device, "source_orientation": case.orientation,
                "operator_note": "physical marks confirmed; DOA not used to adjust placement",
                "room_note": room_note, "timestamp": datetime.now().isoformat(),
                "azimuth_deg": case.intended_az_deg, "elevation_deg": case.intended_el_deg,
                "distance_m": None if case.background else distance,
                "distance_quality": "NOT_APPLICABLE" if case.background else "MEASURED",
                "truth_quality": "NOT_APPLICABLE" if case.background else "MEASURED"}
        base = {**asdict(case), **rich, "capture": "", "analysis": ""}
        if dry_run:
            rows.append({**base, "outcome": "INCONCLUSIVE", "reason": "dry-run; no capture",
                         "quick_quality": "NOT_RUN"})
            continue
        tc = TestCase(case.test_id, "BACKGROUND" if case.background else "BROADBAND_1K_5K",
                      case.question, case.intended_az_deg, case.intended_el_deg,
                      None if case.background else distance, "MEASURED",
                      stimulus_wav="" if case.background else "broadband_1k_5k_60s.wav")
        attempt = 1
        while True:
            try:
                result, capture_path, analysis_path = _capture_and_analyze(tc, session, config_path, rich, 2.0)
                quality = quick_quality_check(result, background=case.background)
                result["quick_quality_check"] = quality
                _write_json(analysis_path, result)
                if quality["status"] == "PASS":
                    outcome, reason = "PASS", "quick quality checks passed"
                    break
                print("\nREPEAT_THIS_CAPTURE")
                print(json.dumps(quality, ensure_ascii=False, indent=2))
                choice = input_fn("按 Enter 重采；输入 S 保留失败并继续: ").strip().upper()
                if choice == "S":
                    outcome, reason = "INCONCLUSIVE", "operator continued after failed quick quality"
                    break
                failed_capture = session / "failed_attempts" / f"{case.test_id}_attempt{attempt}.npz"
                failed_analysis = session / "failed_attempts" / f"{case.test_id}_attempt{attempt}.analysis.json"
                capture_path.replace(failed_capture); analysis_path.replace(failed_analysis)
                attempt += 1
            except Exception as exc:
                with (session / "error.log").open("a", encoding="utf-8") as log:
                    log.write(f"{case.test_id}: {exc}\n")
                outcome, reason, quality = "INCONCLUSIVE", f"capture/analyze error: {exc}", {"status": "ERROR"}
                capture_path = analysis_path = None
                break
        rows.append({**base, "outcome": outcome, "reason": reason,
                     "quick_quality": quality["status"],
                     "capture": str(capture_path.relative_to(session)) if capture_path else "",
                     "analysis": str(analysis_path.relative_to(session)) if analysis_path else ""})
        _write_session_files(session, rows, metadata)
    if dry_run:
        _write_json(session / "external_playback_dry_run.json",
                     {"playback_mode": PLAYBACK_MODE, "computer_audio_used": False,
                      "planned_capture_count": len(cases)})
        _write_json(session / "validation_gate.json", {"status": "NOT_RUN_DRY_RUN"})
    else:
        print("\n采集完成，正在自动执行六频带与跨距离/对称/重复比较……")
        bands = _band_ablation(session, rows)
        metadata["validation_gate"] = _aggregate(session, rows, bands)
    _write_session_files(session, rows, metadata)
    return session
