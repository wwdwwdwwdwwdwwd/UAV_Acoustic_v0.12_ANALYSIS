"""Geometry-based far-field DAS search, continuous beamforming and tracking."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
import numpy as np
from scipy import fft as scipy_fft

from .coordinates import az_el_to_unit


def angular_error_deg(a0: float, e0: float, a1: float, e1: float) -> float:
    u = az_el_to_unit(a0, e0); v = az_el_to_unit(a1, e1)
    return float(np.rad2deg(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


class FarFieldDAS:
    """Plane-wave delay-and-sum. Search uses frequency-domain phase steering;
    waveform synthesis uses fractional linear interpolation and fixed 1/M gain.
    """
    def __init__(self, mic_xyz: np.ndarray, fs: int, speed: float, band: tuple[float,float], max_bins: int=192):
        self.xyz=np.asarray(mic_xyz,float); self.xyz-=np.mean(self.xyz,axis=0,keepdims=True)
        if self.xyz.shape!=(128,3): raise ValueError("DAS geometry must be (128,3)")
        self.fs=int(fs); self.speed=float(speed); self.band=tuple(map(float,band)); self.max_bins=int(max_bins)
        self.max_delay_samples=float(np.max(np.linalg.norm(self.xyz[:,:2],axis=1))*self.fs/self.speed)

    def _spectrum(self, frame: np.ndarray):
        x=np.asarray(frame,np.float32)
        if x.ndim!=2 or x.shape[0]!=128: raise ValueError("spatial frame must be 128 x samples")
        win=np.hanning(x.shape[1]).astype(np.float32); spec=scipy_fft.rfft(x*win[None,:],axis=1,workers=1)
        freq=scipy_fft.rfftfreq(x.shape[1],1/self.fs); keep=np.flatnonzero((freq>=self.band[0])&(freq<=self.band[1]))
        if len(keep)>self.max_bins: keep=keep[np.linspace(0,len(keep)-1,self.max_bins,dtype=int)]
        return spec[:,keep].astype(np.complex64),freq[keep]

    def scan(self, frame: np.ndarray, az_values: np.ndarray, el_values: np.ndarray) -> dict:
        started=time.perf_counter(); az_grid,el_grid=np.meshgrid(np.asarray(az_values,float),np.asarray(el_values,float),indexing="xy")
        az=az_grid.ravel(); el=el_grid.ravel(); directions=az_el_to_unit(az,el)
        spec,freq=self._spectrum(frame)
        delays=(directions@self.xyz.T)/self.speed
        # Broadband coherent power divided by total incoherent input power.
        # A single scalar denominator prevents empty FFT bins from dominating
        # the score (which happens if per-bin ratios are averaged uniformly).
        denom=max(float(128.0*np.sum(np.abs(spec)**2)),1e-20)
        scores=np.zeros(len(az),np.float64)
        chunk=128
        for start in range(0,len(az),chunk):
            tau=delays[start:start+chunk]
            phase=np.exp(-2j*np.pi*tau[:,:,None]*freq[None,None,:]).astype(np.complex64)
            beam=np.sum(phase*spec[None,:,:],axis=1)
            scores[start:start+len(tau)]=np.sum(np.abs(beam)**2,axis=1)/denom
        order=np.argsort(scores)[::-1]
        return {"az":az,"el":el,"scores":scores,"order":order,"best_az":float(az[order[0]]),"best_el":float(el[order[0]]),"best_score":float(scores[order[0]]),"second_score":float(scores[order[1]]) if len(order)>1 else math.nan,"duration_ms":(time.perf_counter()-started)*1000,"num_directions":len(az)}

    def beamform(self, frame: np.ndarray, az_deg: float, el_deg: float, previous_direction: tuple[float,float]|None=None) -> tuple[np.ndarray,dict]:
        started=time.perf_counter(); x=np.asarray(frame,np.float32); n=x.shape[1]; pos=np.arange(n,dtype=np.float64)
        def one(az,el):
            delays=(self.xyz@az_el_to_unit(az,el))*self.fs/self.speed
            # x_m(t)=s(t+tau_m), so recover s(t) at x_m(t-tau_m).
            return np.mean(np.stack([np.interp(pos-d,pos,x[m],left=0.0,right=0.0) for m,d in enumerate(delays)]),axis=0).astype(np.float32)
        current=one(az_deg,el_deg)
        if previous_direction is not None and previous_direction!=(az_deg,el_deg):
            old=one(*previous_direction); fade=np.linspace(0.0,1.0,n,dtype=np.float32); current=old*(1-fade)+current*fade
        q={"input_rms_median":float(np.median(np.sqrt(np.mean(x*x,axis=1)))),"input_rms_p90":float(np.quantile(np.sqrt(np.mean(x*x,axis=1)),.9)),"output_rms":float(np.sqrt(np.mean(current*current))),"output_peak":float(np.max(np.abs(current))),"output_clip_fraction":float(np.mean(np.abs(current)>=.999)),"normalization_gain":1/128.0,"processing_ms":(time.perf_counter()-started)*1000}
        return current,q


@dataclass
class TrackerConfig:
    alpha: float=.65; beta: float=.18; valid_score: float=.10; valid_peak_ratio: float=1.015
    coast_duration_s: float=.6; radius_min: float=4.; radius_max: float=28.
    velocity_margin: float=.12; uncertainty_gain: float=1.8; uncertainty_growth: float=2.0


class SpatialTracker:
    STATES=("SEARCH","TRACK","COAST","REACQUIRE")
    def __init__(self,cfg:TrackerConfig):
        self.cfg=cfg; self.state="SEARCH"; self.az=self.el=None; self.vaz=self.vel=0.; self.uncertainty=2.; self.misses=0; self.last_time=None; self.last_valid=None; self.just_acquired=False

    def predict(self,now:float):
        if self.az is None:return None,None,0.1
        reference=self.last_time if self.last_time is not None else now
        dt=max(.001,now-reference); return self.az+self.vaz*dt,self.el+self.vel*dt,dt

    def radius(self,dt:float):
        return float(np.clip(self.cfg.radius_min+abs(self.vaz)*dt*self.cfg.velocity_margin+abs(self.vel)*dt*self.cfg.velocity_margin+self.uncertainty*self.cfg.uncertainty_gain+self.misses*2,self.cfg.radius_min,self.cfg.radius_max))

    def acquire(self,now:float,az_deg:float,el_deg:float):
        self.state="TRACK";self.az=float(az_deg);self.el=float(el_deg);self.vaz=self.vel=0.;self.uncertainty=2.;self.misses=0;self.last_time=float(now);self.last_valid=float(now);self.just_acquired=True

    def update(self,now:float,measurement:tuple[float,float]|None,score:float,second:float) -> dict:
        started=time.perf_counter(); previous=self.state; paz,pel,dt=self.predict(now); ratio=score/max(second,1e-12) if np.isfinite(second) else math.nan
        valid=measurement is not None and score>=self.cfg.valid_score and ratio>=self.cfg.valid_peak_ratio
        reason="spatial_valid" if valid else "spatial_gate_failed"
        ia=ie=it=math.nan
        if valid:
            maz,mel=measurement
            if self.az is None: self.az,self.el=maz,mel; self.vaz=self.vel=0.
            else:
                ia=maz-paz; ie=mel-pel; it=float(math.hypot(ia,ie)); self.az=paz+self.cfg.alpha*ia; self.el=pel+self.cfg.alpha*ie
                if not self.just_acquired:self.vaz+=self.cfg.beta*ia/dt; self.vel+=self.cfg.beta*ie/dt
                self.just_acquired=False
            self.uncertainty=max(1.,self.uncertainty*.6);self.misses=0;self.last_valid=now;self.state="TRACK";reason="acquired" if previous in ("SEARCH","REACQUIRE") else "spatial_valid"
        else:
            if self.az is not None:self.az,self.el=paz,pel
            self.misses+=1;self.uncertainty=min(self.cfg.radius_max,self.uncertainty+self.cfg.uncertainty_growth)
            loss=math.inf if self.last_valid is None else now-self.last_valid
            if previous=="SEARCH": self.state="SEARCH";reason="no_initial_peak"
            elif loss<=self.cfg.coast_duration_s:self.state="COAST";reason="temporary_spatial_loss"
            else:self.state="REACQUIRE";reason="coast_timeout"
        self.az=None if self.az is None else float(np.clip(self.az,-90,90)); self.el=None if self.el is None else float(np.clip(self.el,-45,45)); self.last_time=now
        return {"previous_state":previous,"new_state":self.state,"transition_reason":reason,"spatial_valid":bool(valid),"predicted_az_deg":paz,"predicted_el_deg":pel,"tracked_az_deg":self.az,"tracked_el_deg":self.el,"az_velocity_dps":self.vaz,"el_velocity_dps":self.vel,"innovation_az_deg":ia,"innovation_el_deg":ie,"innovation_total_deg":it,"tracking_uncertainty":self.uncertainty,"consecutive_spatial_misses":self.misses,"time_since_last_valid_ms":math.nan if self.last_valid is None else (now-self.last_valid)*1000,"tracker_update_ms":(time.perf_counter()-started)*1000}
