"""Top-K spatial candidate generation and frozen OS02 semantic acquisition."""
from __future__ import annotations
import hashlib,math,threading,time
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import torch
from transformers import AutoFeatureExtractor,AutoModelForAudioClassification
from .coordinates import az_el_to_unit
from .das_tracking import angular_error_deg
from .detection_adapters.common import INTERNAL_ROOT

MODEL_SHA="10455420F5AF15A7287BE1D32D37AD6B681F74E3C152FE3BF386FDB0A0427489"
THRESHOLD=.5

class FrozenSamidEngine:
    def __init__(self,torch_threads:int=16):
        model_dir=INTERNAL_ROOT/"third_party"/"samid_ast_model";actual=hashlib.sha256((model_dir/"model.safetensors").read_bytes()).hexdigest().upper()
        if actual!=MODEL_SHA:raise RuntimeError(f"BLOCK: OS02 model SHA mismatch {actual}")
        torch.set_num_threads(int(torch_threads));self.extractor=AutoFeatureExtractor.from_pretrained(str(model_dir),local_files_only=True);self.model=AutoModelForAudioClassification.from_pretrained(str(model_dir),local_files_only=True).eval();self.lock=threading.Lock()
    def infer_batch_16k(self,waveforms:np.ndarray):
        value=np.asarray(waveforms,np.float32)
        with self.lock:
            t0=time.perf_counter();inputs=self.extractor([x for x in value],sampling_rate=16000,return_tensors="pt");feature_ms=(time.perf_counter()-t0)*1000;t0=time.perf_counter()
            with torch.inference_mode():scores=torch.softmax(self.model(**inputs).logits,dim=-1)[:,1].cpu().numpy().astype(np.float64)
        return scores,feature_ms,(time.perf_counter()-t0)*1000
    def infer_single_16k(self,waveform:np.ndarray):return float(self.infer_batch_16k(np.asarray(waveform,np.float32)[None,:])[0][0])

def nms_directions(az:np.ndarray,el:np.ndarray,scores:np.ndarray,k:int,min_separation_deg:float):
    chosen=[]
    for idx in np.argsort(scores)[::-1]:
        if all(angular_error_deg(float(az[idx]),float(el[idx]),float(az[j]),float(el[j]))>=min_separation_deg for j in chosen):chosen.append(int(idx))
        if len(chosen)>=k:break
    return chosen

def select_uav_candidate(candidates:list[dict],threshold:float=THRESHOLD):
    eligible=[c for c in candidates if c["samid_raw_score"]>=threshold]
    return max(eligible,key=lambda c:c["samid_raw_score"]) if eligible else None

def _contrast_db(numerator:float,baseline:float):return float(10*np.log10(max(numerator,1e-12)/max(baseline,1e-12)))

class SemanticAcquirer:
    def __init__(self,das,engine:FrozenSamidEngine,cfg:dict,diagnostics:Path,logger,artifact_writer=None):self.das=das;self.engine=engine;self.cfg=cfg;self.diagnostics=Path(diagnostics);self.diagnostics.mkdir(parents=True,exist_ok=True);self.logger=logger;self.artifact_writer=artifact_writer;self.search_id=0;self.batch_id=0
    def run(self,physical_block:np.ndarray,state:str,context:dict|None=None):
        x=np.asarray(physical_block,np.float32)
        if x.shape!=(128,80000):raise ValueError(f"semantic acquire frozen block must be (128,80000), got {x.shape}")
        self.search_id+=1;self.batch_id+=1;started=time.perf_counter();azv=np.arange(self.cfg["az_fov_deg"][0],self.cfg["az_fov_deg"][1]+.1,self.cfg["global_az_step_deg"]);elv=np.arange(self.cfg["el_fov_deg"][0],self.cfg["el_fov_deg"][1]+.1,self.cfg["global_el_step_deg"]);coarse=self.das.scan(x,azv,elv)
        initial=nms_directions(coarse["az"],coarse["el"],coarse["scores"],self.cfg["search_top_k"]*3,self.cfg["candidate_min_angular_separation_deg"])
        refined=[]
        for idx in initial:
            a=float(coarse["az"][idx]);e=float(coarse["el"][idx]);fine=self.das.scan(x,np.arange(max(-90,a-self.cfg["global_az_step_deg"]),min(90,a+self.cfg["global_az_step_deg"])+.1,self.cfg["fine_step_deg"]),np.arange(max(-45,e-self.cfg["global_el_step_deg"]),min(45,e+self.cfg["global_el_step_deg"])+.1,self.cfg["fine_step_deg"]));refined.append({"az_deg":fine["best_az"],"el_deg":fine["best_el"],"spatial_score":fine["best_score"]})
        if refined:
            raz=np.asarray([r["az_deg"] for r in refined]);rel=np.asarray([r["el_deg"] for r in refined]);rsc=np.asarray([r["spatial_score"] for r in refined]);keep=nms_directions(raz,rel,rsc,self.cfg["search_top_k"],self.cfg["candidate_min_angular_separation_deg"]);candidates=[refined[i] for i in keep]
        else:candidates=[]
        wave16=[]
        for c in candidates:
            mono,q=self.das.beamform(x,c["az_deg"],c["el_deg"]);prepared=resample_poly(mono,16000,80000).astype(np.float32)[-16000:];wave16.append(prepared);c.update(input_rms_median=q["input_rms_median"],beam_output_rms=float(np.sqrt(np.mean(mono*mono))),beam_output_peak=float(np.max(np.abs(mono))),beam_output_clip_fraction=float(np.mean(np.abs(mono)>=.999)),_mono80=mono,_wave16=prepared)
            distances=np.asarray([angular_error_deg(c["az_deg"],c["el_deg"],a,e) for a,e in zip(coarse["az"],coarse["el"])])
            background=coarse["scores"][distances>=self.cfg["candidate_min_angular_separation_deg"]];offaxis=coarse["scores"][(distances>=25)&(distances<=50)]
            c["coherence_factor"]=float(c["spatial_score"]);c["directional_contrast_db"]=_contrast_db(c["spatial_score"],float(np.median(background)) if len(background) else 1e-12);c["off_axis_contrast_db"]=_contrast_db(c["spatial_score"],float(np.median(offaxis)) if len(offaxis) else 1e-12)
        if wave16:scores,feature_ms,infer_ms=self.engine.infer_batch_16k(np.stack(wave16))
        else:scores=np.empty(0);feature_ms=infer_ms=0.
        for rank,(c,wave,score) in enumerate(zip(candidates,wave16,scores),1):
            c.update(candidate_rank=rank,samid_raw_score=float(score),samid_raw_present=bool(score>=THRESHOLD),samid_batch_id=self.batch_id,samid_inference_ms=infer_ms,samid_feature_ms=feature_ms)
            if (context or {}).get("save_diagnostics",True):
                path=f"search_{self.search_id:04d}_candidate_{rank:02d}.wav"
                if self.artifact_writer:self.artifact_writer.wav(path,16000,wave)
                else:wavfile.write(self.diagnostics/path,16000,wave.astype(np.float32))
        selected=select_uav_candidate(candidates);stamp=time.time()
        for c in candidates:
            chosen=c is selected;c["selected_as_uav"]=chosen;c["selection_reason"]="highest_samid_ge_0p5" if chosen else ("samid_below_0p5" if not c["samid_raw_present"] else "lower_eligible_samid")
            self.logger.log("search_candidates.csv",{"timestamp":stamp,"search_id":self.search_id,**c,**(context or {})})
        return {"state":state,"search_id":self.search_id,"candidates":candidates,"selected":selected,"acquired_by_samid":selected is not None,"global_scan_ms":coarse["duration_ms"],"topk_das_samid_ms":(time.perf_counter()-started)*1000-coarse["duration_ms"],"total_search_ms":(time.perf_counter()-started)*1000}
