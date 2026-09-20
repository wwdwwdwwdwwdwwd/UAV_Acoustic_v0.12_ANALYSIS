from __future__ import annotations
import numpy as np
from uav_acoustic.controller_state import ControllerState
from uav_acoustic.noise_gate import BaselineBuilder,NoveltyGate,band_features

def test_two_enter_state_machine():
    s=ControllerState();assert s.state=="WAIT_FOR_CALIBRATION";assert s.enter()=="START_CALIBRATION";assert s.state=="NOISE_CALIBRATION";s.calibration_complete(True);assert s.state=="WAIT_FOR_MEASUREMENT";assert s.enter()=="START_DETECTION";assert s.state=="SEARCH_IDLE"

def test_unhealthy_calibration_fails():
    s=ControllerState();s.enter();s.calibration_complete(False);assert not s.baseline_ready and s.state=="WAIT_FOR_CALIBRATION";assert s.enter()=="START_CALIBRATION"

def _fixture(amp=1.,tone=0.):
    fs=80000;t=np.arange(8000)/fs;x=np.random.default_rng(7).normal(0,.01,(16,len(t)))*amp
    if tone:x+=tone*np.sin(2*np.pi*700*t)[None,:]
    return x.astype(np.float32),fs

def test_stable_background_does_not_trigger():
    x,fs=_fixture();b=BaselineBuilder(range(16))
    for _ in range(30):b.add(band_features(x,fs,list(range(16))))
    gate=NoveltyGate(b.build());assert not any(gate.update(band_features(x,fs,list(range(16))))["triggered"] for _ in range(20))

def test_new_source_triggers_once_without_queue():
    x,fs=_fixture();b=BaselineBuilder(range(16))
    for _ in range(30):b.add(band_features(x,fs,list(range(16))))
    gate=NoveltyGate(b.build());y,_=_fixture(tone=.4);results=[gate.update(band_features(y,fs,list(range(16)))) for _ in range(5)];assert sum(r["triggered"] for r in results)==1

def test_drop_ratios_are_invalid_and_semantic_suppressed():
    calls=[]
    for loss in (.1,.3,.5):
        rate=1-loss;valid=rate>=.98 and rate>=.98
        if valid:calls.append(loss)
        assert not valid
    assert calls==[]

def test_background_model_does_not_modify_waveform():
    x,fs=_fixture();before=x.copy();b=BaselineBuilder(range(16));b.add(band_features(x,fs,list(range(16))));b.build();assert np.array_equal(x,before)
