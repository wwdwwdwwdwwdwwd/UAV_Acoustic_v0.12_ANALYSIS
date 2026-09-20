from __future__ import annotations
import json
import subprocess
import tempfile
import time
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from .common import CadencedMonoAdapter, INTERNAL_ROOT

class JoulesFeatureFusionAdapter(CadencedMonoAdapter):
    adapter_id="os04_joules_feature_fusion"; upstream_name="Joules feature_fusion"; hop_samples=16000
    def __init__(self,fs:int,config:dict):
        super().__init__(fs,config)
        exe=INTERNAL_ROOT/"adapters"/"joules_feature_fusion_bridge.exe"; state=INTERNAL_ROOT/"models"/"joules_feature_fusion_state.json"
        self.proc=subprocess.Popen([str(exe),"infer",str(state)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    def prepare_transport(self,raw_80k:np.ndarray)->np.ndarray:
        # Preserve the upstream Rust xeval resampler: no Python-side resampling.
        return np.asarray(raw_80k,np.float32)
    def infer_model_chunk(self,audio_16k:np.ndarray):
        t=time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as f: path=Path(f.name)
        try:
            wavfile.write(path,80000,np.clip(audio_16k,-1,1).astype(np.float32))
            assert self.proc.stdin is not None and self.proc.stdout is not None
            self.proc.stdin.write(str(path)+"\n"); self.proc.stdin.flush(); row=json.loads(self.proc.stdout.readline())
        finally: path.unlink(missing_ok=True)
        wall=(time.perf_counter()-t)*1000
        return float(row["score"]),max(0.0,wall-float(row["forward_ms"])),float(row["forward_ms"])
    def __del__(self):
        try:
            if self.proc.poll() is None: self.proc.stdin.write("QUIT\n"); self.proc.stdin.flush(); self.proc.terminate()
        except Exception: pass
