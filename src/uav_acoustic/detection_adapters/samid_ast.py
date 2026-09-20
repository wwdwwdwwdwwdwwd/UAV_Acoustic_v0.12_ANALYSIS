from __future__ import annotations
import time
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from .common import CadencedMonoAdapter, INTERNAL_ROOT

class SamidASTAdapter(CadencedMonoAdapter):
    adapter_id="os02_samid_ast"; upstream_name="SAMID AST"
    def __init__(self,fs:int,config:dict):
        super().__init__(fs,config); path=str(INTERNAL_ROOT/"third_party"/"samid_ast_model")
        self.extractor=AutoFeatureExtractor.from_pretrained(path,local_files_only=True)
        self.model=AutoModelForAudioClassification.from_pretrained(path,local_files_only=True).eval()
    def infer_model_chunk(self,audio_16k:np.ndarray):
        t=time.perf_counter(); inputs=self.extractor(audio_16k,sampling_rate=16000,return_tensors="pt"); p=(time.perf_counter()-t)*1000
        t=time.perf_counter()
        with torch.no_grad(): score=float(torch.softmax(self.model(**inputs).logits,dim=-1)[0,1])
        return score,p,(time.perf_counter()-t)*1000
