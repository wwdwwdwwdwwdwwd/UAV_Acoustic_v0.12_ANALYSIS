"""UAV Detector V2: monotonic rotor shape plus separate reliability.

Clean-room implementation from standard DSP formulas.  Magnitude/power-only
detection produces one real W(t,f), shared by every microphone; the original
complex 128-channel STFT remains the spatial estimator input.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfilt


class AdaptiveNoisePSD:
    """Online per-frequency minimum-controlled PSD with target protection."""
    def __init__(self,bins:int,rise=.025,fall=.16,minimum_decay=.003):
        self.noise=np.full(int(bins),np.nan,float); self.minimum=self.noise.copy()
        self.rise=float(rise); self.fall=float(fall); self.minimum_decay=float(minimum_decay)
    def update(self,power:np.ndarray,protection:np.ndarray|None=None):
        value=np.maximum(np.asarray(power,float),1e-30)
        if value.shape!=self.noise.shape: raise ValueError("PSD shape changed")
        if np.isnan(self.noise).any():
            baseline=median_filter(value,size=31,mode="reflect")
            self.noise[:]=np.minimum(value,baseline); self.minimum[:]=self.noise
        protected=np.zeros_like(value) if protection is None else np.clip(protection,0,1)
        self.minimum=np.minimum(value,self.minimum*(1+self.minimum_decay))
        target=np.minimum(value,np.maximum(self.minimum*2.5,self.noise))
        alpha=np.where(target<=self.noise,self.fall,self.rise*(1-.95*protected))
        self.noise+=alpha*(target-self.noise)
        snr_db=10*np.log10(np.maximum(value/self.noise,1e-12))
        reliability=np.clip(1-np.exp(-np.maximum(snr_db,0)/8),0,1)
        return self.noise.copy(),reliability


@dataclass(frozen=True)
class V2Features:
    harmonic_score:float
    cepstral_score:float
    envelope_score:float
    candidate_f0_hz:float
    harmonic_count:int
    harmonic_consistency:float


def monotonic_final_evidence(shape:np.ndarray|float,reliability:np.ndarray|float):
    """Monotonic in both arguments; weak reliable evidence remains visible."""
    s=np.clip(np.asarray(shape,float),0,1); r=np.clip(np.asarray(reliability,float),0,1)
    return np.clip(s*(.55+.45*r),0,1)


class UAVDetectorV2:
    def __init__(self,fs:int,nfft:int,config:dict):
        self.fs=int(fs); self.nfft=int(nfft); cfg=config.get("uav_detector_v2",{})
        self.profile_name="UAV_DETECTOR_V2"; self.frequency=np.fft.rfftfreq(nfft,1/fs)
        self.low=float(cfg.get("frequency_low_hz",100)); self.high=float(cfg.get("frequency_high_hz",8000))
        self.f0_low=float(cfg.get("f0_low_hz",60)); self.f0_high=float(cfg.get("f0_high_hz",500))
        self.max_harmonics=int(cfg.get("max_harmonics",20)); self.max_bins=int(cfg.get("max_spatial_bins",4))
        self.shape_candidate=float(cfg.get("shape_candidate_threshold",.36))
        self.present_shape=float(cfg.get("present_shape_threshold",.46))
        self.present_reliability=float(cfg.get("present_reliability_threshold",.18))
        self.present_final=float(cfg.get("present_final_threshold",.38))
        self.present_frames=int(cfg.get("present_frames",2)); self.hold_frames=int(cfg.get("hold_frames",3))
        self.fusion_mode=str(cfg.get("shape_fusion","soft_or"))
        self.shape_weights=np.asarray(cfg.get("shape_weights",[.75,.28,.42]),float)
        if self.shape_weights.shape!=(3,) or np.any(self.shape_weights<0): raise ValueError("shape weights must be three non-negative values")
        self.analysis_bins=np.flatnonzero((self.frequency>=self.low)&(self.frequency<=self.high))
        self.noise_model=AdaptiveNoisePSD(len(self.analysis_bins)); self.protection=np.zeros(len(self.analysis_bins))
        self._sos=butter(2,[max(self.low,80),min(self.high,fs*.45)],btype="bandpass",fs=fs,output="sos")
        self.above=0; self.hold=0; self.state="ABSENT"; self.ema_final=0.; self._initialized=False
        self.selected_bins=self.analysis_bins[:1]; self.selected_frequency_hz=self.frequency[self.selected_bins]

    def _robust_power(self,spectra:np.ndarray):
        p=np.abs(np.asarray(spectra)[:,:,self.analysis_bins])**2
        scale=np.median(p,axis=(0,2)); healthy=np.isfinite(scale)&(scale>0)
        median=float(np.median(scale[healthy])) if np.any(healthy) else 1.
        healthy&=(scale>.05*median)&(scale<20*median)
        if np.count_nonzero(healthy)<4: healthy=np.isfinite(scale)&(scale>0)
        normalized=p[:,healthy,:]/np.maximum(scale[healthy][None,:,None],1e-30)
        return np.median(normalized,axis=1),healthy

    def _harmonic_hps(self,spectrum:np.ndarray,noise:np.ndarray,reliability:np.ndarray):
        freq=self.frequency[self.analysis_bins]
        log_ratio=np.log(np.maximum(spectrum/noise,1e-12)); local=median_filter(log_ratio,size=31,mode="reflect")
        contrast=np.clip((log_ratio-local)/np.log(10),-2,4)
        best=(-1.,self.f0_low,0,0.,np.zeros_like(contrast))
        for f0 in np.arange(self.f0_low,self.f0_high+1e-9,5.):
            harmonics=np.arange(1,self.max_harmonics+1)*f0
            harmonics=harmonics[(harmonics>=self.low)&(harmonics<=self.high)]
            if len(harmonics)<5: continue
            idx=np.clip(np.searchsorted(freq,harmonics),1,len(freq)-2)
            peaks=np.asarray([np.max(contrast[i-1:i+2]) for i in idx])
            rel=np.asarray([np.max(reliability[i-1:i+2]) for i in idx])
            support=np.clip((peaks-.08)/.75,0,1)
            # Log-HPS supports a missing fundamental by also evaluating orders 2..6.
            first=float(np.mean(peaks[:min(6,len(peaks))])); missing=float(np.mean(peaks[1:min(7,len(peaks))]))
            hps=np.clip((max(first,missing)-.05)/.8,0,1)
            count=int(np.count_nonzero(support>.25)); consistency=float(np.mean(support>.15))
            score=float(np.clip(.48*hps+.34*np.mean(np.sort(support)[-min(8,len(support)):])+.18*consistency,0,1))
            if score>best[0]:
                mask=np.zeros_like(contrast)
                for i,s in zip(idx,support*(.35+.65*rel)):
                    mask[max(0,i-1):min(len(mask),i+2)]=np.maximum(mask[max(0,i-1):min(len(mask),i+2)],s)
                best=(score,float(f0),count,consistency,mask)
        return best

    def _cepstral(self,spectrum:np.ndarray,f0:float):
        logmag=np.log(np.maximum(spectrum,1e-30)); logmag-=median_filter(logmag,size=31,mode="reflect")
        full=np.zeros(len(self.frequency)); full[self.analysis_bins]=logmag
        cep=np.abs(np.fft.irfft(full,n=self.nfft)); q=int(np.clip(round(self.fs/max(f0,1)),2,len(cep)-3))
        peak=float(np.max(cep[max(2,q-2):q+3])); base=float(np.median(cep[max(2,q-20):min(len(cep),q+21)]))+1e-12
        return float(np.clip((peak/base-1.15)/4.5,0,1))

    def _robust_envelope(self,audio:np.ndarray,healthy:np.ndarray,f0:float):
        x=np.asarray(audio,float)
        if x.ndim!=2: raise ValueError("V2 audio reference must be channels x samples")
        chosen=np.flatnonzero(healthy)
        if chosen.size>32: chosen=chosen[np.linspace(0,chosen.size-1,32,dtype=int)]
        x=x[chosen]; x-=np.mean(x,axis=1,keepdims=True)
        scale=np.sqrt(np.mean(x*x,axis=1,keepdims=True))+1e-12; x/=scale
        band=sosfilt(self._sos,x,axis=1)
        block=max(1,self.fs//1000); usable=band.shape[1]//block*block
        env=np.sqrt(np.mean(band[:,:usable].reshape(len(band),-1,block)**2,axis=2)+1e-30)
        env/=np.median(env,axis=1,keepdims=True)+1e-12
        reference=np.median(env,axis=0); reference-=np.mean(reference)
        denom=float(np.dot(reference,reference))+1e-30; rate=self.fs/block
        # Search rotor AM independently; candidate f0 is included as a supporting lag.
        lags=np.unique(np.r_[np.arange(max(2,int(rate/80)),min(len(reference)//3,int(rate/4))+1),int(np.clip(rate/max(f0,1),2,len(reference)//3))])
        ac=np.asarray([np.dot(reference[:-k],reference[k:])/denom for k in lags]) if lags.size else np.asarray([0.])
        peak=float(np.max(ac)); baseline=float(np.median(ac))
        return float(np.clip((peak-max(.04,baseline))/.38,0,1))

    def shape_fusion(self,harmonic,cepstral,envelope):
        f=np.clip(np.asarray([harmonic,cepstral,envelope],float),0,1)
        if self.fusion_mode=="hps_alone": return float(f[0])
        if self.fusion_mode=="soft_or": return float(1-np.prod(1-np.clip(self.shape_weights*f,0,.98)))
        if self.fusion_mode=="nonnegative_linear": return float(np.clip(np.dot(self.shape_weights,f)/max(np.sum(self.shape_weights),1e-12),0,1))
        raise ValueError(f"unknown shape fusion: {self.fusion_mode}")

    def _state(self,shape,reliability,final):
        present=shape>=self.present_shape and reliability>=self.present_reliability and final>=self.present_final
        if present:self.above+=1
        else:self.above=max(0,self.above-1)
        if self.above>=self.present_frames:self.state="PRESENT";self.hold=self.hold_frames
        elif shape>=self.shape_candidate or self.hold>0:self.state="CANDIDATE";self.hold=max(0,self.hold-1)
        else:self.state="ABSENT"
        return self.state

    def analyze(self,spectra:np.ndarray,window_sum:float,audio:np.ndarray):
        subpower,healthy=self._robust_power(spectra); frame_power=np.median(subpower,axis=0)
        noise,reliability_tf=self.noise_model.update(frame_power,self.protection)
        harmonic,f0,count,consistency,structure=self._harmonic_hps(frame_power,noise,reliability_tf)
        cepstral=self._cepstral(frame_power,f0); envelope=self._robust_envelope(audio,healthy,f0)
        shape=self.shape_fusion(harmonic,cepstral,envelope)
        supported=structure>.12
        reliability=float(np.median(reliability_tf[supported])) if np.any(supported) else float(np.percentile(reliability_tf,85))
        final=float(monotonic_final_evidence(shape,reliability))
        self.ema_final=final if not self._initialized else .6*final+.4*self.ema_final; self._initialized=True
        state=self._state(shape,reliability,self.ema_final); self.protection=np.clip(.75*structure+.25*reliability_tf,0,1)
        subrel=np.clip(1-np.exp(-np.maximum(10*np.log10(np.maximum(subpower/noise[None,:],1e-12)),0)/8),0,1)
        weight=np.clip(subrel*(.12+.88*structure[None,:])*(.35+.65*shape),0,1)
        rank=np.argsort(np.mean(weight,axis=0))[::-1]; chosen=rank[:min(self.max_bins,len(rank))]; chosen=np.sort(chosen)
        self.selected_bins=self.analysis_bins[chosen]; self.selected_frequency_hz=self.frequency[self.selected_bins]
        features=V2Features(harmonic,cepstral,envelope,f0,count,consistency)
        return {"selected_bins":self.selected_bins,"tf_weights":weight[:,chosen],"spectral_evidence":self.ema_final,
          "all_band_spectral_evidence":shape,"uav_probability":self.ema_final,"shape_score":shape,"snr_reliability":reliability,
          "detector_confidence":reliability,"uav_state":state,"target_present":state=="PRESENT","features":features.__dict__,
          "active_band_count":count,"configured_band_count":self.max_harmonics,"band_score":[harmonic,cepstral,envelope],
          "band_contrast_db":[],"active_bands":[f"f0={f0:.0f}Hz",f"TF={len(chosen)}"],
          "current_noise_floor_db":10*np.log10(noise+1e-30),"noise_floor_db":(10*np.log10(noise+1e-30)).tolist(),
          "healthy_detection_channels":int(np.count_nonzero(healthy))}

    def update_noise(self,*_): return None


def build_uav_detector(fs:int,nfft:int,config:dict):
    return UAVDetectorV2(fs,nfft,config)
