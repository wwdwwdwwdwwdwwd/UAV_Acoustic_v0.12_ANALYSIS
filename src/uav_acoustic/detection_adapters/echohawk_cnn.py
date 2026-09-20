from __future__ import annotations
import sys
import time
import numpy as np
import torch
from .common import CadencedMonoAdapter, INTERNAL_ROOT

class EchoHawkCNNAdapter(CadencedMonoAdapter):
    adapter_id="os03_echohawk_cnn"; upstream_name="EchoHawk CNN"; hop_samples=16000
    def __init__(self,fs:int,config:dict):
        super().__init__(fs,config); source=INTERNAL_ROOT/"third_party"/"echohawk"
        sys.path.insert(0,str(source))
        from echohawk import features,models
        self.features=features; self.model=models.build_cnn(n_mels=40,n_classes=2)
        self.model.load_state_dict(torch.load(INTERNAL_ROOT/"models"/"echohawk_cnn.pt",map_location="cpu",weights_only=True)); self.model.eval()
        scaler=np.load(INTERNAL_ROOT/"models"/"echohawk_scaler.npz"); self.mean=scaler["mean"]; self.std=scaler["std"]
    def infer_model_chunk(self,audio_16k:np.ndarray):
        t=time.perf_counter(); mel=self.features.log_mel_spectrogram(audio_16k,16000)
        mel=np.pad(mel,((0,0),(0,max(0,64-mel.shape[1]))),mode="constant")[:,:64]
        feat=((mel[None,:,:]-self.mean)/self.std)[:,None,:,:]
        tensor=torch.tensor(feat,dtype=torch.float32); p=(time.perf_counter()-t)*1000
        t=time.perf_counter()
        with torch.no_grad(): score=float(torch.softmax(self.model(tensor),dim=1)[0,1])
        return score,p,(time.perf_counter()-t)*1000

