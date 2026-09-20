"""Round-2 field workflow with a formal NO_DRONE_MODE."""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from datetime import datetime
import csv
import json
from pathlib import Path
from typing import Callable
import wave
import numpy as np

from .analysis import analyze_frame
from .config import load_config, resolve_geometry_path
from .io.capture_file import save_capture
from .io.geometry import load_wavefrag_csv, transform_pcb_to_engineering
from .io.wavefrag_udp import WaveFragUDPConfig, WaveFragUDPSource, WaveFragUDPError
from .doa.wavefront import local_pair_wavefront_fit, robust_pair_wavefront_fit
from .tracking.doa_track import track_broadband_doa
from .preflight import run_preflight


EVIDENCE_SIMULATED = "SIMULATED_ACOUSTIC_SOURCE"
OUTCOMES = ("PASS", "FAIL", "INCONCLUSIVE", "NOT_TESTED", "REQUIRES_REAL_DRONE")


@dataclass
class TestCase:
    test_id: str
    source_type: str
    engineering_question: str
    true_azimuth_deg: float | None = None
    true_elevation_deg: float | None = None
    distance_m: float | None = None
    distance_quality: str = "UNKNOWN"
    evidence_type: str = "STANDARD_LOUDSPEAKER"
    optional: bool = False
    stimulus_wav: str = ""


def standard_matrix() -> list[TestCase]:
    cases = [
        TestCase("background_01", "BACKGROUND", "Does NO_TARGET suppress false directions?"),
        TestCase("background_02", "BACKGROUND", "Are channel health and background behavior repeatable?"),
    ]
    for hz in (100, 200, 500, 1000, 2000, 3000):
        cases.append(TestCase(f"tone_{hz}", f"TONE_{hz}_HZ",
            f"Is {hz} Hz detected and is its own frequency used for DOA?"))
    for az in (-60, -30, 0, 30, 60):
        cases.append(TestCase(f"broadband_az_{az:+d}", "BROADBAND_1K_5K",
            "Is broadband azimuth sign, monotonicity and absolute error acceptable?", true_azimuth_deg=az, true_elevation_deg=0))
    for el in (0, 15, 30):
        cases.append(TestCase(f"broadband_el_{el:+d}", "BROADBAND_1K_5K",
            "Does elevation follow known ground truth?", true_azimuth_deg=0, true_elevation_deg=el))
    for run in range(1, 6):
        cases.append(TestCase(f"repeat_az30_{run}", "BROADBAND_1K_5K",
            "Is fixed-position DOA repeatable across five independent captures?", true_azimuth_deg=30, true_elevation_deg=0))
    for az in (0, 30, -30):
        cases.append(TestCase(f"uav_like_{az:+d}", "UAV_COMB_BPF_200",
            "Do BPF, harmonics, periodicity and multi-harmonic DOA work end-to-end?",
            true_azimuth_deg=az, true_elevation_deg=0, evidence_type=EVIDENCE_SIMULATED))
    cases.append(TestCase("moving_left_to_right", "BROADBAND_OR_UAV_COMB",
        "Is the DOA track continuous from negative through zero to positive azimuth?"))
    return cases


def optional_matrix() -> list[TestCase]:
    return [
        TestCase("real_drone_hover", "REAL_DRONE", "Does the chain detect and localize a real drone?",
                 evidence_type="REAL_DRONE", optional=True),
        TestCase("vehicle_noise", "VEHICLE_NOISE", "How does uncalibrated vehicle noise affect false alarms?",
                 evidence_type="UNCALIBRATED_NOISE", optional=True),
        TestCase("dual_source", "DUAL_SOURCE", "Can interference/multiple peaks be exposed without hiding ambiguity?",
                 optional=True),
    ]


def calibration_matrix(repeats: int = 2) -> list[TestCase]:
    """Minimal R2.1 matrix: two backgrounds, five azimuth and three elevation points."""
    if repeats not in {2, 3}:
        raise ValueError("calibration repeats must be 2 or 3")
    cases = [
        TestCase("cal_background_01", "BACKGROUND", "Confirm no stable false target in background 01."),
        TestCase("cal_background_02", "BACKGROUND", "Confirm no stable false target in background 02."),
    ]
    for az in (-60, -30, 0, 30, 60):
        for repeat in range(1, repeats + 1):
            cases.append(TestCase(f"cal_az_{az:+d}_r{repeat}", "BROADBAND_1K_5K",
                "Calibrate horizontal sign, monotonicity, top-K peaks and edge-angle behavior.",
                true_azimuth_deg=float(az), true_elevation_deg=0.0,
                distance_quality="MEASURED", stimulus_wav="broadband_1k_5k_60s.wav"))
    for el in (0, 15, 30):
        for repeat in range(1, repeats + 1):
            cases.append(TestCase(f"cal_el_{el:+d}_r{repeat}", "BROADBAND_1K_5K",
                "Calibrate elevation trend, repeatability and PCB-Y physical sign.",
                true_azimuth_deg=0.0, true_elevation_deg=float(el),
                distance_quality="MEASURED", stimulus_wav="broadband_1k_5k_60s.wav"))
    return cases


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def _write_outputs(session: Path, rows: list[dict], metadata: dict) -> None:
    session.mkdir(parents=True, exist_ok=True)
    fields = ["test_id", "source_type", "engineering_question", "outcome", "reason",
              "true_azimuth_deg", "true_elevation_deg", "distance_m", "distance_quality",
              "stimulus_wav", "volume_setting", "environment", "note", "evidence_type", "capture", "analysis"]
    with (session/"summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    sections = {
        "VERIFIED_THIS_ROUND": [r for r in rows if r["outcome"] in {"PASS", "FAIL"}],
        "INCONCLUSIVE": [r for r in rows if r["outcome"] == "INCONCLUSIVE"],
        "NOT_TESTED": [r for r in rows if r["outcome"] == "NOT_TESTED"],
        "REQUIRES_REAL_DRONE": [r for r in rows if r["outcome"] == "REQUIRES_REAL_DRONE"],
    }
    lines = ["# Round 2 Summary", "", f"Mode: `{metadata['mode']}`", "",
             "A FAIL means a performed test did not meet its criterion. Missing equipment or unknown truth is never converted to FAIL.", ""]
    for title, values in sections.items():
        lines.extend([f"## {title}", ""])
        if not values: lines.append("None.")
        for row in values:
            lines.append(f"- **{row['test_id']} — {row['outcome']}**: {row['engineering_question']} — {row['reason']}")
        lines.append("")
    lines.extend(["## Evidence boundary", "",
        "UAV-like WAV tests are labelled `SIMULATED_ACOUSTIC_SOURCE`. They validate only the BPF/harmonic/periodicity/multi-harmonic-localization algorithm chain and are not real-drone validation.",
        "Tests without a calibrated sound-level meter must not be labelled 80–90 dB. Estimated distances lower evidence quality.", ""])
    (session/"summary.md").write_text("\n".join(lines), encoding="utf-8")
    _write_json(session/"session_metadata.json", metadata)
    (session/"error.log").touch(exist_ok=True)
    (session/"screenshots").mkdir(exist_ok=True)


def _aggregate_completed_tests(session: Path, rows: list[dict]) -> None:
    aggregate_ids={"broadband_azimuth_aggregate","elevation_aggregate","repeatability_aggregate"}
    rows[:]=[r for r in rows if r["test_id"] not in aggregate_ids]
    def analyses(prefix: str, count: int):
        selected=[r for r in rows if r["test_id"].startswith(prefix) and r.get("analysis")]
        if len(selected) != count: return []
        values=[]
        for row in selected:
            data=json.loads((session/row["analysis"]).read_text(encoding="utf-8"))
            srp=data.get("broadband_srp_phat") or {}
            if "azimuth_deg" not in srp: return []
            values.append((row, float(srp["azimuth_deg"]), float(srp["elevation_deg"])))
        return values
    base={"volume_setting":"","environment":"","note":"automatic aggregate",
          "evidence_type":"STANDARD_LOUDSPEAKER","capture":"","analysis":"",
          "true_azimuth_deg":None,"true_elevation_deg":None,"distance_m":None,"distance_quality":""}
    az_values=analyses("broadband_az_",5)
    if az_values:
        az_values.sort(key=lambda x: float(x[0]["true_azimuth_deg"]))
        errors=np.array([abs(az-float(row["true_azimuth_deg"])) for row,az,_ in az_values])
        estimates=np.array([az for _,az,_ in az_values])
        monotonic=bool(np.all(np.diff(estimates)>0)); signs=all(float(r["true_azimuth_deg"])==0 or np.sign(a)==np.sign(float(r["true_azimuth_deg"])) for r,a,_ in az_values)
        median=float(np.median(errors)); p90=float(np.percentile(errors,90)); ok=monotonic and signs and median<=10 and p90<=15
        rows.append({**base,"test_id":"broadband_azimuth_aggregate","source_type":"BROADBAND_1K_5K",
            "engineering_question":"Are sign, monotonicity, median and P90 azimuth errors acceptable?",
            "outcome":"PASS" if ok else "FAIL",
            "reason":f"median={median:.2f}°, p90={p90:.2f}°, monotonic={monotonic}, signs_correct={signs}"})
    el_values=analyses("broadband_el_",3)
    if el_values:
        el_values.sort(key=lambda x: float(x[0]["true_elevation_deg"]))
        errors=np.array([abs(el-float(row["true_elevation_deg"])) for row,_,el in el_values]); estimates=np.array([el for _,_,el in el_values])
        median=float(np.median(errors)); trend=bool(np.all(np.diff(estimates)>=0)); ok=median<=15 and trend
        rows.append({**base,"test_id":"elevation_aggregate","source_type":"BROADBAND_1K_5K",
            "engineering_question":"Is elevation trend correct with median error <=15°?",
            "outcome":"PASS" if ok else "FAIL","reason":f"median_error={median:.2f}°, monotonic_trend={trend}"})
    repeat=analyses("repeat_az30_",5)
    if repeat:
        az=np.array([a for _,a,_ in repeat]); el=np.array([e for _,_,e in repeat])
        az_std=float(np.std(az)); el_std=float(np.std(el)); ok=az_std<=5 and el_std<=8
        reason=(f"az mean/std/max-min={np.mean(az):.2f}/{az_std:.2f}/{np.ptp(az):.2f}°; "
                f"el mean/std/max-min={np.mean(el):.2f}/{el_std:.2f}/{np.ptp(el):.2f}°")
        rows.append({**base,"test_id":"repeatability_aggregate","source_type":"BROADBAND_1K_5K",
            "engineering_question":"Do five independent captures meet repeatability targets?",
            "outcome":"PASS" if ok else "FAIL","reason":reason})


def _aggregate_calibration(session: Path, rows: list[dict], repeats: int) -> dict:
    aggregate_ids = {"calibration_azimuth_aggregate", "calibration_elevation_aggregate",
                     "calibration_background_aggregate"}
    rows[:] = [r for r in rows if r["test_id"] not in aggregate_ids]
    completed = [r for r in rows if r.get("analysis")]
    records = []
    for row in completed:
        data = json.loads((session / row["analysis"]).read_text(encoding="utf-8"))
        srp = data.get("broadband_srp_phat") or {}
        wave = data.get("independent_wavefront_fit") or {}
        records.append({"row": row, "srp": srp, "wavefront": wave, "status": data.get("status"),
                        "coordinate_transform": data.get("coordinate_transform", {})})

    def grouped(prefix: str, truth_key: str, expected: list[float]):
        output = []
        for truth in expected:
            values = [x for x in records if x["row"]["test_id"].startswith(prefix)
                      and float(x["row"][truth_key]) == truth]
            if len(values) != repeats:
                return []
            az = np.asarray([float(x["srp"]["azimuth_deg"]) for x in values])
            el = np.asarray([float(x["srp"]["elevation_deg"]) for x in values])
            output.append({"truth": truth, "median_azimuth_deg": float(np.median(az)),
                           "median_elevation_deg": float(np.median(el)),
                           "repeat_azimuth_std_deg": float(np.std(az)),
                           "repeat_elevation_std_deg": float(np.std(el)),
                           "ambiguous_count": sum(bool(x["srp"]["diagnostics"]["ambiguous"]) for x in values),
                           "top_k": [x["srp"]["diagnostics"]["top_candidates"] for x in values],
                           "physics_residual_us": [x["wavefront"].get("rms_residual_us") for x in values]})
        return output

    azimuth = grouped("cal_az_", "true_azimuth_deg", [-60., -30., 0., 30., 60.])
    elevation = grouped("cal_el_", "true_elevation_deg", [0., 15., 30.])
    backgrounds = [x for x in records if x["row"]["test_id"].startswith("cal_background_")]
    elevation_records = [x for x in records if x["row"]["test_id"].startswith("cal_el_")]
    y_hypotheses = {}
    selected_y = None
    if len(elevation_records) == 3 * repeats:
        configured_signs = {int(x["coordinate_transform"].get("pcb_y_to_engineering_y_sign", 1))
                            for x in elevation_records}
        if len(configured_signs) == 1:
            configured_sign = configured_signs.pop()
            truths = np.asarray([float(x["row"]["true_elevation_deg"]) for x in elevation_records])
            configured_estimates = np.asarray([float(x["srp"]["elevation_deg"]) for x in elevation_records])
            pcb_estimates = configured_estimates * configured_sign
            for label, physical_sign in (("PCB_+Y_IS_PHYSICAL_UP", 1), ("PCB_-Y_IS_PHYSICAL_UP", -1)):
                estimates = pcb_estimates * physical_sign
                medians = [float(np.median(estimates[truths == truth])) for truth in (0., 15., 30.)]
                y_hypotheses[label] = {
                    "physical_y_sign": physical_sign,
                    "all_estimates_deg": estimates.tolist(),
                    "median_by_truth_deg": medians,
                    "mae_all_captures_deg": float(np.mean(np.abs(estimates - truths))),
                    "monotonic_0_15_30": bool(np.all(np.diff(medians) > 0)),
                    "positive_up": bool(medians[-1] > medians[0]),
                }
            ranked = sorted(y_hypotheses, key=lambda key: y_hypotheses[key]["mae_all_captures_deg"])
            best, other = ranked
            if (y_hypotheses[best]["monotonic_0_15_30"] and y_hypotheses[best]["positive_up"]
                    and y_hypotheses[other]["mae_all_captures_deg"]
                    - y_hypotheses[best]["mae_all_captures_deg"] > 1.0):
                selected_y = best
                selected_sign = y_hypotheses[best]["physical_y_sign"]
                for point in elevation:
                    point["median_elevation_deg"] *= selected_sign * configured_sign
                    point["reported_under_selected_y_hypothesis"] = best
    az_est = np.asarray([x["median_azimuth_deg"] for x in azimuth]) if azimuth else np.array([])
    el_est = np.asarray([x["median_elevation_deg"] for x in elevation]) if elevation and selected_y else np.array([])
    gate = {
        "schema_version": 2, "mode": "R2.1_CALIBRATION", "repeats_per_angle": repeats,
        "azimuth_points": azimuth, "elevation_points": elevation,
        "pcb_y_physical_up_hypotheses": y_hypotheses,
        "selected_pcb_y_physical_up_hypothesis": selected_y,
        "physical_y_sign_unique": selected_y is not None,
        "background_statuses": [x["status"] for x in backgrounds],
        "azimuth_complete": bool(azimuth), "elevation_complete": bool(elevation),
        "background_complete": len(backgrounds) == 2,
        "azimuth_monotonic": bool(az_est.size and np.all(np.diff(az_est) > 0)),
        "azimuth_sign_correct": bool(az_est.size and az_est[0] < 0 < az_est[-1]),
        "edge_60_within_25deg": bool(az_est.size and abs(az_est[0] + 60) <= 25 and abs(az_est[-1] - 60) <= 25),
        "elevation_monotonic": bool(el_est.size and np.all(np.diff(el_est) > 0)),
        "elevation_positive_up": bool(el_est.size and el_est[-1] > el_est[0]),
        "background_no_valid_target": bool(len(backgrounds) == 2 and
                                           all(x["status"] in {"NO_TARGET", "LOW_CONFIDENCE"} for x in backgrounds)),
    }
    gate["ready_for_next_uav_test"] = all(gate[k] for k in (
        "azimuth_complete", "elevation_complete", "background_complete", "azimuth_monotonic",
        "azimuth_sign_correct", "edge_60_within_25deg", "elevation_monotonic",
        "elevation_positive_up", "physical_y_sign_unique", "background_no_valid_target"))
    _write_json(session / "calibration_gate.json", gate)
    base = {"true_azimuth_deg": None, "true_elevation_deg": None, "distance_m": None,
            "distance_quality": "MEASURED", "stimulus_wav": "broadband_1k_5k_60s.wav",
            "volume_setting": "", "environment": "", "note": "automatic calibration gate",
            "evidence_type": "STANDARD_LOUDSPEAKER", "capture": "", "analysis": ""}
    rows.extend([
        {**base, "test_id": "calibration_azimuth_aggregate", "source_type": "CALIBRATION_AGGREGATE",
         "engineering_question": "Are five azimuth points monotonic, signed and valid at ±60°?",
         "outcome": "PASS" if all(gate[k] for k in ("azimuth_complete", "azimuth_monotonic", "azimuth_sign_correct", "edge_60_within_25deg")) else "INCONCLUSIVE",
         "reason": f"complete={gate['azimuth_complete']}, monotonic={gate['azimuth_monotonic']}, signs={gate['azimuth_sign_correct']}, edges={gate['edge_60_within_25deg']}"},
        {**base, "test_id": "calibration_elevation_aggregate", "source_type": "CALIBRATION_AGGREGATE",
         "engineering_question": "Are 0/+15/+30° elevation points monotonic with positive-up sign?",
         "outcome": "PASS" if gate["elevation_complete"] and gate["elevation_monotonic"] and gate["elevation_positive_up"] and gate["physical_y_sign_unique"] else "INCONCLUSIVE",
         "reason": f"complete={gate['elevation_complete']}, monotonic={gate['elevation_monotonic']}, positive_up={gate['elevation_positive_up']}, fixed_y={gate['selected_pcb_y_physical_up_hypothesis']}"},
        {**base, "test_id": "calibration_background_aggregate", "source_type": "CALIBRATION_AGGREGATE",
         "engineering_question": "Do both backgrounds avoid a trusted target?",
         "outcome": "PASS" if gate["background_no_valid_target"] else "INCONCLUSIVE",
         "reason": f"statuses={gate['background_statuses']}"},
    ])
    return gate


def _ask(prompt: str, input_fn: Callable[[str], str], default="") -> str:
    value = input_fn(prompt).strip()
    return value if value else default


def _resolve_stimulus(config_path: Path, filename: str) -> Path:
    config_path = Path(config_path).resolve()
    candidates = [Path.cwd() / "PHONE_AUDIO" / filename,
                  config_path.parents[1] / "PHONE_AUDIO" / filename,
                  config_path.parents[1] / "dev" / "legacy_release_files" / filename,
                  config_path.parents[2] / "PHONE_AUDIO" / filename]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"calibration stimulus not found: {filename}")


def _validate_stimulus(path: Path) -> dict:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() not in {1, 2} or wav.getsampwidth() != 2:
            raise ValueError("calibration WAV must be 16-bit mono/stereo PCM")
        return {"path": str(path), "channels": wav.getnchannels(),
                "sample_rate_hz": wav.getframerate(),
                "duration_s": wav.getnframes() / wav.getframerate()}


def _placement_text(case: TestCase, ordinal: int, total: int, distance_m: float | None) -> str:
    if case.source_type == "BACKGROUND":
        number = "01" if case.test_id.endswith("01") else "02"
        return (f"\n{'=' * 68}\n【{ordinal}/{total} 背景测试 {number}】\n"
                "请停止手机/音箱播放。\n"
                "确认环境中没有测试声源。\n\n"
                "准备好后按 Enter。\n"
                f"{'=' * 68}")
    distance = "待输入" if distance_m is None else f"{distance_m:.3f} m"
    if case.test_id.startswith("cal_az_"):
        heading = "水平方位校准"
    else:
        heading = "俯仰校准"
    az_note = "（正前方）" if case.true_azimuth_deg == 0 else ""
    el_note = "（与阵列中心同高）" if case.true_elevation_deg == 0 else ""
    return (f"\n{'=' * 68}\n【{ordinal}/{total} {heading}】\n\n"
            "请将手机/音箱摆到：\n"
            f"方位角：{case.true_azimuth_deg:+g}°{az_note}\n"
            f"俯仰角：{case.true_elevation_deg:+g}°{el_note}\n"
            f"距离：{distance}（手机/音箱中心到阵列中心的直线距离）\n"
            "手机/音箱扬声器朝向：阵列中心\n\n"
            "测试音频：\n"
            "broadband_1k_5k_60s.wav\n\n"
            "请在手机/外部音箱上开始或保持播放。\n"
            "确认声音和摆放已经稳定后，回到电脑按 Enter 开始采集。\n\n"
            "0°基准 = 从麦克风阵列中心，垂直阵列正面向外的方向\n"
            "整轮使用同一播放设备，手机媒体音量/外部音箱增益保持不变。\n"
            f"{'=' * 68}")


def _capture_and_analyze(case: TestCase, session: Path, config_path: Path,
                         ground_truth: dict, duration_s: float) -> tuple[dict, Path, Path]:
    config, resolved = load_config(config_path)
    hw = config["hardware"]; proc = config["processing"]
    xyz = load_wavefrag_csv(resolve_geometry_path(config, resolved))
    coord = config["coordinates"]
    xyz = transform_pcb_to_engineering(xyz,
        x_sign=int(coord["pcb_x_to_engineering_x_sign"]),
        y_sign=int(coord["pcb_y_to_engineering_y_sign"]))
    udp = WaveFragUDPConfig(host=str(hw["udp_host"]), port=int(hw["udp_port"]),
        channels=int(hw["channels"]), fs=int(hw["fs"]),
        reorder_vendor_channels=bool(hw["reorder_vendor_channels"]),
        recv_bytes=int(hw["receive_buffer_bytes"]), timeout_s=float(hw["udp_timeout_s"]),
        socket_buffer_bytes=int(hw["socket_buffer_bytes"]),
        expected_source_ip=hw.get("expected_source_ip"),
        expected_source_port=int(hw["expected_source_port"]) if hw.get("expected_source_port") else None,
        expected_datagram_bytes=int(hw["expected_datagram_bytes"]) if hw.get("expected_datagram_bytes") else None)
    with WaveFragUDPSource(xyz, udp) as source:
        frame = source.read_frame(int(round(duration_s*udp.fs)))
        frame.metadata.update({"test_id": case.test_id, "source_type": case.source_type,
            "evidence_type": case.evidence_type, "ground_truth": ground_truth,
            **ground_truth,
            "timestamp": datetime.now().isoformat(),
            "stimulus_wav": case.stimulus_wav,
            "real_drone_required": False,
            "coordinate_transform": {
                "pcb_x_to_engineering_x_sign": int(coord["pcb_x_to_engineering_x_sign"]),
                "pcb_y_to_engineering_y_sign": int(coord["pcb_y_to_engineering_y_sign"]),
                "pcb_y_physical_sign_evidence": coord["pcb_y_physical_sign_evidence"],
            }})
        capture_path = save_capture(session/"captures"/f"{case.test_id}.npz", frame,
                                    raw_int16=source.last_raw_int16, config_snapshot=config)
    tone = float(case.source_type.split("_")[1]) if case.source_type.startswith("TONE_") else None
    result = analyze_frame(frame, known_tone_hz=tone,
        doa_band_hz=tuple(proc["doa_band_hz"]), broadband_band_hz=tuple(proc["broadband_band_hz"]),
        clipping_threshold=float(proc["clipping_threshold"]),
        dead_channel_rms_ratio=float(proc["dead_channel_rms_ratio"]),
        healthy_channel_minimum=int(proc["healthy_channel_minimum"]),
        confidence_thresholds=dict(proc.get("confidence", {})),
        vendor_focal_distance_m=float(ground_truth.get("distance_m") or 2.0))
    result["test_id"] = case.test_id
    result["stimulus_wav"] = case.stimulus_wav
    result["ground_truth"] = ground_truth
    result["coordinate_transform"] = dict(coord)
    # The claim-bearing wavefront audit must use explicit band support and
    # sub-sample delays.  Keep the original integer-lag result for before/after
    # traceability, but never use its 12.5 us quantised angles as the final fit.
    result["independent_wavefront_fit"] = robust_pair_wavefront_fit(
        frame.audio, frame.fs, frame.mic_xyz, freq_range=(1000.0, 2000.0))
    result["legacy_integer_wavefront_fit"] = local_pair_wavefront_fit(
        frame.audio, frame.fs, frame.mic_xyz, freq_range=(500.0, 5000.0))
    if case.test_id.startswith("moving_"):
        result["tracking"] = track_broadband_doa(frame.audio, frame.fs, frame.mic_xyz,
            freq_range=tuple(proc["broadband_band_hz"]))
    result["test_case"] = asdict(case); result["ground_truth"] = ground_truth
    analysis_path = session/"analysis"/f"{case.test_id}.analysis.json"
    _write_json(analysis_path, result)
    return result, capture_path, analysis_path


def _evaluate(case: TestCase, result: dict, truth_accurate: bool) -> tuple[str, str]:
    if not result["channel_metrics"]["healthy_minimum_pass"]:
        return "FAIL", "fewer than configured minimum healthy channels"
    if case.source_type == "BACKGROUND":
        ok = result["status"] in {"NO_TARGET", "LOW_CONFIDENCE"}
        return ("PASS", f"background status={result['status']}") if ok else ("FAIL", "background produced a high-confidence VALID target")
    if case.source_type.startswith("TONE_"):
        tone = result["known_tone"]
        requested = tone["requested_hz"]
        freq_ok = abs(tone["estimated_hz"]-requested) <= max(8.0, requested*0.015)
        own = bool(tone["proof_requested_tone_drives_doa"])
        if freq_ok and own and result["status"] != "NO_TARGET":
            qualifier = result.get("interpretation") or result["status"]
            return "PASS", f"measured={tone['estimated_hz']:.2f} Hz; own-frequency DOA; {qualifier}"
        return "FAIL", f"frequency/own-frequency DOA criterion failed; measured={tone['estimated_hz']:.2f} Hz; status={result['status']}"
    if case.source_type == "UAV_COMB_BPF_200":
        h = result.get("harmonic_multiband_doa") or {}
        ok = (abs(result["uav_features"]["bpf_hz"]-200) <= 12 and
              result["uav_features"]["harmonic_comb_score"] > 0 and
              result["uav_features"]["periodicity_score"] > 0 and
              len(h.get("used_harmonic_orders", [])) >= 2)
        reason = (f"SIMULATED_ACOUSTIC_SOURCE only: BPF={result['uav_features']['bpf_hz']:.2f}, "
                  f"harmonics={h.get('used_harmonic_orders', [])}, fused_status={h.get('status')}")
        return ("PASS" if ok else "FAIL"), reason
    if case.test_id.startswith("moving_"):
        track = result.get("tracking", {})
        ok = track.get("negative_to_positive_crossing") and track.get("max_azimuth_jump_deg", 999) <= 30
        return ("PASS" if ok else "FAIL"), (f"negative-to-positive={track.get('negative_to_positive_crossing')}; "
            f"max smoothed jump={track.get('max_azimuth_jump_deg')}°")
    if not truth_accurate:
        return "INCONCLUSIVE", "capture analyzed, but angle/position was not accurately measured"
    gt_az = case.true_azimuth_deg; gt_el = case.true_elevation_deg
    estimate = result.get("broadband_srp_phat") or {}
    if gt_az is not None and "azimuth_deg" in estimate:
        az_error = abs(float(estimate["azimuth_deg"])-gt_az)
        el_error = abs(float(estimate["elevation_deg"])-(gt_el or 0))
        ok = az_error <= 15 and el_error <= 15
        return ("PASS" if ok else "FAIL"), f"SRP-PHAT az_error={az_error:.1f}°, el_error={el_error:.1f}°"
    return "INCONCLUSIVE", "automatic criterion requires a measured ground truth or trajectory annotation"


def run_round2(session_root: Path, *, config_path: Path | None = None, dry_run=False,
               calibration=False, calibration_repeats=2, input_fn=input) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = (Path(session_root)/f"Round2_{stamp}").resolve()
    for name in ("captures", "analysis", "screenshots"):
        (session/name).mkdir(parents=True, exist_ok=True)
    mode = "R2.1_CALIBRATION" if calibration else "NO_DRONE_MODE"
    metadata = {"schema_version": 3, "created_local": datetime.now().isoformat(),
                "mode": mode, "real_drone_required": False,
                "dry_run": bool(dry_run), "ground_truth_convention": "+Z front, +X positive azimuth, +Y positive elevation"}
    config_path = config_path or (Path(__file__).resolve().parents[2]/"03_configs"/"default.yaml")
    runtime_config, _ = load_config(config_path)
    playback_mode = str(runtime_config.get("playback", {}).get("mode", "external_manual"))
    if calibration and playback_mode != "external_manual":
        raise ValueError("R2.1 Calibration requires playback.mode: external_manual")
    preflight = run_preflight(config_path, bind_udp=not dry_run)
    _write_json(session/"preflight.json", preflight)
    metadata["preflight_status"] = preflight["status"]
    rows = []
    cases = calibration_matrix(calibration_repeats) if calibration else standard_matrix()+optional_matrix()
    if calibration:
        metadata["calibration_plan"] = [asdict(case) for case in cases]
        metadata["capture_ground_truth"] = []
        stimulus_path = _resolve_stimulus(config_path, "broadband_1k_5k_60s.wav")
        metadata["playback"] = {"mode": "EXTERNAL_PHONE", "computer_audio_used": False,
                                "phone_audio": _validate_stimulus(stimulus_path)}
    if dry_run:
        playback_events = []
        for ordinal, case in enumerate(cases, 1):
            if calibration:
                print(_placement_text(case, ordinal, len(cases), 2.5 if case.stimulus_wav else None))
                playback_events.append({"test_id": case.test_id,
                    "action": "STOP_EXTERNAL_PLAYBACK" if not case.stimulus_wav
                              else "EXTERNAL_PHONE_PLAY_STABLE_THEN_ENTER"})
            outcome = "REQUIRES_REAL_DRONE" if case.source_type == "REAL_DRONE" else ("NOT_TESTED" if case.optional else "INCONCLUSIVE")
            rows.append({**asdict(case), "outcome": outcome,
                         "reason": "dry-run validated workflow/files only; no measurement was made",
                         "volume_setting": "", "environment": "SYNTHETIC_DRY_RUN", "note": "",
                         "capture": "", "analysis": ""})
        if calibration:
            metadata["external_playback_dry_run_events"] = playback_events
            _write_json(session / "external_playback_dry_run.json", {
                "mode": "EXTERNAL_PHONE", "computer_audio_used": False, "events": playback_events})
        if calibration:
            _aggregate_calibration(session, rows, calibration_repeats)
        _write_outputs(session, rows, metadata)
        return session

    if calibration:
        print("\nR2.1 校准测试 / CALIBRATION (RECOMMENDED)")
        print("顺序：背景 -> 水平方位 -60/-30/0/+30/+60 -> 俯仰 0/+15/+30")
        print("每个角度重复采集；距离必须实测为 2–3 m。")
    else:
        print("\n标准测试（无需无人机） / STANDARD TEST (NO DRONE REQUIRED)")
    environment = _ask("Environment/location: ", input_fn, "UNRECORDED")
    if calibration:
        volume = _ask("记录当前手机媒体音量/外部音箱增益（整轮严禁调整）: ", input_fn, "FIXED_UNRECORDED")
        common_distance = None
        distance_quality = "UNKNOWN"
        print("\n注意：背景结束后只输入一次实测直线斜距 R；其余每步摆好后只按 Enter。")
    else:
        print("Press Enter when the source/position is ready; S skips unavailable conditions without failing the round.")
        volume = _ask("Loudspeaker volume setting (or UNKNOWN): ", input_fn, "UNKNOWN")
        distance_text = _ask("Common distance metres (blank if unavailable): ", input_fn)
        common_distance = float(distance_text) if distance_text else None
        distance_quality = "MEASURED" if distance_text else "UNKNOWN"
    for ordinal, case in enumerate(cases, 1):
        if calibration and case.stimulus_wav and common_distance is None:
            while True:
                distance_value = _ask("请输入本轮固定直线斜距 R（2–3 m，例如 2.50）: ", input_fn)
                try:
                    common_distance = float(distance_value)
                except ValueError:
                    print("请输入数字距离，例如 2.50")
                    continue
                if 2.0 <= common_distance <= 3.0:
                    distance_quality = "MEASURED"
                    metadata["fixed_distance_m"] = common_distance
                    metadata["fixed_volume_setting"] = volume
                    break
                print("R 必须为音箱中心到阵列中心的直线斜距，且在 2–3 m。")
        if calibration:
            print(_placement_text(case, ordinal, len(cases), common_distance))
            _ask("准备好后按 Enter 开始采集: ", input_fn)
            ready = ""
        else:
            print(f"\n[{case.test_id}] {case.engineering_question}\nSource: {case.source_type}")
            ready = _ask("Enter=CAPTURE, S=skip unavailable: ", input_fn).upper()
        if ready == "S":
            outcome = "REQUIRES_REAL_DRONE" if case.source_type == "REAL_DRONE" else "NOT_TESTED"
            reason = _ask("Reason condition is unavailable: ", input_fn, "condition/equipment unavailable")
            capture_name = analysis_name = ""
            truth_accurate = False
        else:
            if calibration:
                note = "operator confirmed external playback stable and fixed-radius placement"
                if case.source_type == "BACKGROUND":
                    ground_truth = {"azimuth_deg": None, "elevation_deg": None,
                        "distance_m": None, "distance_quality": "NOT_APPLICABLE",
                        "truth_quality": "NOT_APPLICABLE", "operator_note": note}
                else:
                    case = replace(case, distance_m=common_distance, distance_quality="MEASURED")
                    ground_truth = {"azimuth_deg": case.true_azimuth_deg,
                        "elevation_deg": case.true_elevation_deg, "distance_m": common_distance,
                        "distance_quality": "MEASURED", "truth_quality": "MEASURED",
                        "operator_note": note}
                truth_accurate = True
            else:
                truth_accurate = _ask("Is the stated angle accurately measured? Y/N: ", input_fn, "N").upper() == "Y"
                ground_truth = {"azimuth_deg": case.true_azimuth_deg, "elevation_deg": case.true_elevation_deg,
                    "distance_m": case.distance_m or common_distance, "distance_quality": distance_quality,
                    "truth_quality": "MEASURED" if truth_accurate else "ESTIMATED_OR_UNKNOWN"}
            try:
                result, capture_path, analysis_path = _capture_and_analyze(
                    case, session, config_path, ground_truth,
                    15.0 if case.test_id.startswith("moving_") else 2.0)
                outcome, reason = _evaluate(case, result, truth_accurate)
                capture_name = str(capture_path.relative_to(session)); analysis_name = str(analysis_path.relative_to(session))
            except (OSError, ValueError, RuntimeError, WaveFragUDPError) as exc:
                outcome, reason = "INCONCLUSIVE", f"capture/analyze error: {exc}"
                capture_name = analysis_name = ""
                with (session/"error.log").open("a", encoding="utf-8") as log:
                    log.write(f"{case.test_id}: {exc}\n")
        if not calibration or ready == "S":
            note = _ask("Note (optional): ", input_fn)
        rows.append({**asdict(case), "distance_m": case.distance_m or common_distance,
                     "distance_quality": case.distance_quality if case.distance_m else distance_quality,
                     "outcome": outcome, "reason": reason, "volume_setting": volume,
                     "environment": environment, "note": note, "capture": capture_name, "analysis": analysis_name})
        if calibration and ready != "S" and case.source_type != "BACKGROUND":
            metadata["capture_ground_truth"].append({"test_id": case.test_id, **ground_truth})
        _write_outputs(session, rows, metadata)
    if calibration:
        gate = _aggregate_calibration(session, rows, calibration_repeats)
        metadata["calibration_ready_for_next_uav_test"] = gate["ready_for_next_uav_test"]
    else:
        _aggregate_completed_tests(session, rows)
    _write_outputs(session, rows, metadata)
    return session
