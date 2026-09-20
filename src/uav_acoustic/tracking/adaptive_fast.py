"""Lightweight confidence + motion adaptive AZ/EL tracker."""
from __future__ import annotations
import numpy as np


class AdaptiveFastAzElTracker:
    """Smooth stationary estimates, follow credible motion, confirm weak jumps."""

    def __init__(self, config: dict):
        self.alpha_min=float(config["alpha_min"])
        self.stationary_confidence_gain=float(config["stationary_confidence_gain"])
        self.motion_gain=float(config["motion_gain"])
        self.motion_scale=float(config["motion_scale_deg"])
        self.motion_deadzone=float(config["motion_deadzone_deg"])
        self.confidence_power=float(config["confidence_power"])
        self.alpha_max=float(config["alpha_max"])
        self.high_confidence=float(config["high_confidence_threshold"])
        self.confirm_innovation=float(config["confirmation_innovation_deg"])
        self.confirm_radius=float(config["confirmation_radius_deg"])
        self.confirmed_alpha=float(config["confirmed_relocation_alpha"])
        self.hop=float(config["hop_duration_s"])
        self.coast_frames=int(round(float(config["coast_duration_s"])/self.hop))
        self.max_velocity=float(config.get("max_angular_velocity_dps",120.0))  # diagnostic only
        self.az: float|None=None; self.el: float|None=None; self.missed=0
        self.history:list[bool]=[]; self.state="NO_TARGET"; self.candidate:tuple[float,float]|None=None
        self.reacquire_frames=0; self.velocity=0.0

    @staticmethod
    def _distance(az0: float,el0: float,az1: float,el1: float) -> float:
        return float(np.hypot(az1-az0,el1-el0))

    def _result(self, mode: str, *, alpha=0.0, innovation=0.0, motion=0.0,
                confirmed=False, lost=None) -> dict:
        return {"state":self.state,"az_tracked_deg":self.az,"el_tracked_deg":self.el,
                "lost_for_s":lost,"tracker_mode":mode,"tracker_alpha":float(alpha),
                "tracker_innovation_deg":float(innovation),"tracker_motion_score":float(motion),
                "tracker_candidate_confirmed":bool(confirmed)}

    def step(self, raw_az: float|None,raw_el: float|None,confidence: float,
             credible: bool,edge: bool=False) -> dict:
        self.history=(self.history+[bool(credible)])[-3:]
        if not credible or raw_az is None or raw_el is None:
            self.missed+=1; self.candidate=None; self.reacquire_frames=0
            if self.az is not None and self.missed<=self.coast_frames:
                self.state="COASTING"
                return self._result("HOLD",lost=self.missed*self.hop)
            self.az=None; self.el=None; self.velocity=0.0
            self.state="ACQUIRING" if any(self.history) else "NO_TARGET"
            return self._result("HOLD")
        self.missed=0; raw_az=float(raw_az); raw_el=float(raw_el); conf=float(np.clip(confidence,0,1))
        if self.az is None:
            if sum(self.history)<2:
                self.state="ACQUIRING"; return self._result("HOLD")
            self.az=raw_az; self.el=raw_el; self.state="EDGE_OF_FOV" if edge else "TRACKING"
            return self._result("FAST",alpha=1.0,confirmed=True)

        innovation=self._distance(self.az,self.el,raw_az,raw_el)
        motion=float(1-np.exp(-max(innovation-self.motion_deadzone,0.0)/max(self.motion_scale,1e-9)))
        confirmed=False
        needs_confirmation=innovation>=self.confirm_innovation and conf<self.high_confidence and self.reacquire_frames<=0
        if needs_confirmation:
            if self.candidate is None or self._distance(*self.candidate,raw_az,raw_el)>self.confirm_radius:
                self.candidate=(raw_az,raw_el)
                self.state="EDGE_OF_FOV" if edge else "TRACKING"
                return self._result("HOLD",innovation=innovation,motion=motion)
            confirmed=True; self.candidate=None; self.reacquire_frames=2
        else:
            self.candidate=None

        base=self.alpha_min+self.stationary_confidence_gain*conf
        alpha=base+self.motion_gain*motion*(conf**self.confidence_power)
        if confirmed: alpha=max(alpha,self.confirmed_alpha)
        if self.reacquire_frames>0:
            alpha=max(alpha,self.confirmed_alpha); self.reacquire_frames-=1
        alpha=float(np.clip(alpha,self.alpha_min,self.alpha_max))
        previous=self.az; self.az=float(self.az+alpha*(raw_az-self.az)); self.el=float(self.el+alpha*(raw_el-self.el))
        self.velocity=(self.az-previous)/self.hop
        self.state="EDGE_OF_FOV" if edge else "TRACKING"
        mode="FAST" if alpha>=.55 else "SMOOTH"
        return self._result(mode,alpha=alpha,innovation=innovation,motion=motion,confirmed=confirmed)
