from __future__ import annotations
import sys
import time
import numpy as np
import torch
from .common import CadencedMonoAdapter, INTERNAL_ROOT

class NaccacheCRNNAdapter(CadencedMonoAdapter):
    adapter_id="os01_naccache_crnn"; upstream_name="Naccache CRNN"
    def __init__(self, fs:int, config:dict):
        super().__init__(fs,config)
        source=INTERNAL_ROOT/"third_party"/"naccache_drone_audio_detection"
        sys.path.insert(0,str(source))
        from inference import stream
        self.stream=stream; self.device=torch.device("cpu")
        self.model=stream._load_model(str(INTERNAL_ROOT/"third_party"/"naccache_model"/"drone_classifier_aug_mixed.pt"),self.device)
        self.model.eval(); self.mel,self.to_db=stream._build_mel_transform(self.device)
    def infer_model_chunk(self,audio_16k:np.ndarray):
        t=time.perf_counter(); spec=self.stream._audio_to_tensor(audio_16k,self.mel,self.to_db,self.device); p=(time.perf_counter()-t)*1000
        t=time.perf_counter()
        with torch.no_grad(): score=float(torch.sigmoid(self.model(spec)).squeeze())
        return score,p,(time.perf_counter()-t)*1000
