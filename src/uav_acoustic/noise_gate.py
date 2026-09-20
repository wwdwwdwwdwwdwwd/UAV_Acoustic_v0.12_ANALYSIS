"""Robust background calibration and low-cost novelty monitoring; never modifies audio."""
from __future__ import annotations
import hashlib,json
from dataclasses import dataclass
import numpy as np

BANDS=((30,100),(100,250),(250,500),(500,1000),(1000,2000),(2000,4000),(4000,8000))

def representative_channels(xyz:np.ndarray,count=16):
    xyz=np.asarray(xyz,float);chosen=[int(np.argmax(np.linalg.norm(xyz-np.mean(xyz,axis=0),axis=1)))]
    while len(chosen)<min(count,len(xyz)):
        d=np.min(np.linalg.norm(xyz[:,None,:]-xyz[np.asarray(chosen)][None,:,:],axis=2),axis=1);d[chosen]=-1;chosen.append(int(np.argmax(d)))
    return chosen

def band_features(audio:np.ndarray,fs:int,channels:list[int],bands=BANDS):
    x=np.asarray(audio,np.float32)[channels];win=np.hanning(x.shape[1]);spec=np.abs(np.fft.rfft(x*win[None,:],axis=1))**2;freq=np.fft.rfftfreq(x.shape[1],1/fs);values=[]
    for lo,hi in bands:
        p=np.mean(spec[:,(freq>=lo)&(freq<hi)],axis=1)+1e-20;values.append(10*np.log10(p))
    channel_band=np.stack(values,axis=1);return {"band_db":np.median(channel_band,axis=0),"channel_band_db":channel_band,"broadband_db":float(10*np.log10(np.mean(x*x)+1e-20))}

@dataclass
class NoiseBaseline:
    channels:list[int];bands:list;band_median:list;band_mad:list;band_p90:list;band_p95:list;channel_band_median:list;channel_band_mad:list;frame_count:int
    def as_dict(self):
        d=self.__dict__.copy();raw=json.dumps(d,sort_keys=True,separators=(",",":"));d["baseline_sha256"]=hashlib.sha256(raw.encode()).hexdigest().upper();d["adaptive_background"]=False;return d

class BaselineBuilder:
    def __init__(self,channels,bands=BANDS):self.channels=list(channels);self.bands=list(map(list,bands));self.frames=[]
    def add(self,feature):self.frames.append(np.asarray(feature["channel_band_db"],float))
    def build(self):
        if not self.frames:raise ValueError("no valid calibration frames")
        x=np.stack(self.frames);band=np.median(x,axis=1);bm=np.median(band,axis=0);mad=np.median(np.abs(band-bm),axis=0)
        cm=np.median(x,axis=0);cmad=np.median(np.abs(x-cm),axis=0)
        return NoiseBaseline(self.channels,self.bands,bm.tolist(),mad.tolist(),np.percentile(band,90,axis=0).tolist(),np.percentile(band,95,axis=0).tolist(),cm.tolist(),cmad.tolist(),len(x))

class NoveltyGate:
    def __init__(self,baseline:NoiseBaseline,trigger_z=6.,trigger_delta_db=3.,minimum_bands=1,required=3,window=5):self.baseline=baseline;self.trigger_z=trigger_z;self.trigger_delta=trigger_delta_db;self.minimum_bands=minimum_bands;self.required=required;self.window=window;self.history=[];self.trigger_id=0
    def update(self,feature):
        cur=np.asarray(feature["band_db"]);median=np.asarray(self.baseline.band_median);mad=np.maximum(np.asarray(self.baseline.band_mad),.5);delta=cur-median;z=delta/mad;count=int(np.sum((z>=self.trigger_z)&(delta>=self.trigger_delta)));instant=count>=self.minimum_bands;self.history=(self.history+[instant])[-self.window:];triggered=sum(self.history)>=self.required
        if triggered:self.trigger_id+=1;self.history=[]
        return {"band_delta_db":delta.tolist(),"band_novelty_z":z.tolist(),"aggregate_novelty_score":float(np.max(z)),"triggered_band_count":count,"persistence_count":int(sum(self.history)),"triggered":triggered,"trigger_reason":"ROBUST_BAND_NOVELTY" if triggered else "" ,"novelty_trigger_id":self.trigger_id if triggered else None}
