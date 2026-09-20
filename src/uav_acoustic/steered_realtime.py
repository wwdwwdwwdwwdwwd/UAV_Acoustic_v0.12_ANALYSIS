from __future__ import annotations
import hashlib,json,math,os,platform,sys,threading,time
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
from .async_logger import AsyncCSVLogger
from .config import load_config,resolve_geometry_path
from .das_tracking import FarFieldDAS,SpatialTracker,TrackerConfig,angular_error_deg
from .io.channel_mapping import mapping_hash,MAPPING_VERSION,RAW_CHANNEL_ORDER,PHYSICAL_CHANNEL_ORDER
from .io.geometry import load_wavefrag_csv,transform_pcb_to_engineering
from .io.wavefrag_udp import WaveFragUDPConfig,WaveFragUDPSource
from .semantic_acquire import FrozenSamidEngine,SemanticAcquirer,MODEL_SHA,THRESHOLD
from .threaded_stream import BoundedRing,AcquisitionWorker

VERSION="UAV Acoustic v0.9 OS02 SAMID DAS SEMANTIC ACQUIRE"
HEALTH_FIELDS=["timestamp","elapsed_s","received_datagrams_total","observed_datagrams_per_s","expected_datagrams_per_s","observed_expected_ratio","received_bytes_total","ring_fill_s","ring_capacity_s","ring_overrun_count","snapshot_count","stale_count","max_receive_gap_ms","p95_receive_gap_ms","cpu_process_pct","ram_bytes","spatial_update_hz","semantic_update_hz","pipeline_lag_ms","acquisition_warning","logger_queue_depth","logger_dropped_rows"]
SEARCH_FIELDS=["timestamp","search_id","candidate_rank","az_deg","el_deg","spatial_score","coherence_factor","directional_contrast_db","off_axis_contrast_db","input_rms_median","beam_output_rms","beam_output_peak","beam_output_clip_fraction","samid_raw_score","samid_raw_present","samid_batch_id","samid_inference_ms","selected_as_uav","selection_reason"]
TRACK_FIELDS=["timestamp","elapsed_s","spatial_frame_id","state","previous_state","transition_reason","tracked_az_deg","tracked_el_deg","predicted_az_deg","predicted_el_deg","az_velocity_dps","el_velocity_dps","search_radius_deg","spatial_score","coherence_factor","directional_contrast_db","off_axis_contrast_db","spatial_valid","consecutive_spatial_misses","time_since_last_valid_ms","num_directions_evaluated","local_scan_ms","beamforming_ms","total_spatial_ms","frame_age_ms","output_continuity_gap_count"]
SEMANTIC_FIELDS=["timestamp","elapsed_s","semantic_frame_id","waveform_start_time","waveform_end_time","waveform_age_ms","tracker_state","uav_system_state","tracked_az_deg","tracked_el_deg","predicted_az_deg","predicted_el_deg","az_velocity_dps","el_velocity_dps","search_radius_deg","spatial_score","coherence_factor","directional_contrast_db","off_axis_contrast_db","beam_output_rms","beam_output_peak","beam_output_clip_fraction","samid_raw_score","samid_raw_present","threshold","semantic_inference_ms","semantic_update_hz"]

class Shared:
    def __init__(self):self.lock=threading.Lock();self.spatial={"state":"SEARCH","uav_confirmed":False};self.semantic={};self.beam={};self.search={};self.stop=threading.Event();self.spatial_count=0;self.semantic_count=0;self.continuity_gaps=0;self.spatial_error=None;self.semantic_error=None
    def set(self,name,value):
        with self.lock:setattr(self,name,value)
    def snapshot(self):
        with self.lock:return dict(self.spatial),dict(self.semantic),dict(self.beam),dict(self.search)

def _grid(lo,hi,step):return np.arange(lo,hi+step*.25,step,dtype=float)
def _db(a,b):return float(10*np.log10(max(float(a),1e-12)/max(float(b),1e-12)))

class SpatialWorker:
    def __init__(self,ring,mono,das,tracker,cfg,session,shared,acq,acquirer,logger):self.ring=ring;self.mono=mono;self.das=das;self.tracker=tracker;self.cfg=cfg;self.session=session;self.shared=shared;self.acq=acq;self.acquirer=acquirer;self.logger=logger;self.thread=threading.Thread(target=self.run,name="semantic-gated-spatial",daemon=True);self.started=time.monotonic();self.last_end=None;self.last_search_end=None;self.previous_steer=None
    def start(self):self.thread.start()
    def _quality(self,frame,scan,az,el):
        distances=np.asarray([angular_error_deg(az,el,a,e) for a,e in zip(scan["az"],scan["el"])]);background=scan["scores"][distances>=4.0];directional=_db(scan["best_score"],np.median(background) if len(background) else 1e-12)
        off_az=np.clip(np.asarray([az-30,az+30]),-90,90);off_el=np.clip(np.asarray([el-25,el+25]),-45,45);off=self.das.scan(frame,off_az,off_el)["scores"]
        return float(scan["best_score"]),directional,_db(scan["best_score"],np.median(off))
    def _emit_beam(self,frame,frame_n,hop_n,end_unix,steer,source):
        guard=int(math.ceil(self.das.max_delay_samples))+2;mono_full,q=self.das.beamform(frame,*steer,self.previous_steer);mono=mono_full[frame_n-hop_n-guard:frame_n-guard];mono_end=end_unix-guard/self.ring.fs;q.update(output_rms=float(np.sqrt(np.mean(mono*mono))),output_peak=float(np.max(np.abs(mono))),output_clip_fraction=float(np.mean(np.abs(mono)>=.999)),steering_az_deg=steer[0],steering_el_deg=steer[1],steering_source=source,end_unix=mono_end);self.previous_steer=steer;self.mono.write(mono[None,:],mono_end);self.shared.set("beam",q);return q
    def run(self):
        frame_n=int(self.cfg["spatial_frame_s"]*self.ring.fs);hop_n=int(self.cfg["spatial_hop_s"]*self.ring.fs);search_n=80000
        try:
            while not self.shared.stop.is_set():
                if self.tracker.state in ("SEARCH","REACQUIRE"):
                    item=self.ring.latest(search_n,self.last_search_end)
                    if item is None:time.sleep(.01);continue
                    block,start,end,end_unix=item
                    if self.last_search_end is not None and end-self.last_search_end<int(self.cfg["search_retry_s"]*self.ring.fs):time.sleep(.01);continue
                    self.last_search_end=end;self.ring.acknowledge(end);self.acq.snapshots+=1;mode=self.tracker.state;result=self.acquirer.run(block,mode);self.shared.set("search",result)
                    if result["selected"] is None:
                        self.tracker.state="SEARCH";self.shared.set("spatial",{"state":"SEARCH","uav_confirmed":False,"transition_reason":"no_candidate_samid_ge_0p5","candidate_count":len(result["candidates"]),"frame_age_ms":max(0,(time.time()-end_unix)*1000)});continue
                    selected=result["selected"];now=time.monotonic();self.tracker.acquire(now,selected["az_deg"],selected["el_deg"]);self.previous_steer=(selected["az_deg"],selected["el_deg"]);self.mono.reset_continuity();self.mono.write(selected["_mono80"][None,:],end_unix);self.last_end=end
                    row={"state":"TRACK","uav_confirmed":True,"acquired_by_samid":True,"transition_reason":"candidate_samid_ge_0p5","tracked_az_deg":selected["az_deg"],"tracked_el_deg":selected["el_deg"],"spatial_score":selected["spatial_score"],"coherence_factor":selected["coherence_factor"],"directional_contrast_db":selected["directional_contrast_db"],"off_axis_contrast_db":selected["off_axis_contrast_db"],"frame_age_ms":max(0,(time.time()-end_unix)*1000)};self.shared.set("spatial",row);continue
                item=self.ring.latest(frame_n,self.last_end)
                if item is None:time.sleep(.002);continue
                frame,start,end,end_unix=item
                if end-self.last_end<hop_n:time.sleep(.002);continue
                self.ring.acknowledge(end);started=time.perf_counter();now=time.monotonic();paz,pel,dt=self.tracker.predict(now);paz=float(np.clip(paz,-90,90));pel=float(np.clip(pel,-45,45));radius=self.tracker.radius(dt);az_min=max(-90,paz-radius);az_max=min(90,paz+radius);el_min=max(-45,pel-radius);el_max=min(45,pel+radius);scan=self.das.scan(frame,_grid(az_min,az_max,self.cfg["local_step_deg"]),_grid(el_min,el_max,self.cfg["local_step_deg"]));coh,directional,offaxis=self._quality(frame,scan,scan["best_az"],scan["best_el"]);previous=self.tracker.state;update=self.tracker.update(now,(scan["best_az"],scan["best_el"]),scan["best_score"],scan["second_score"]);skipped=max(0,(end-self.last_end)//hop_n-1)
                if skipped:
                    self.acq.skips+=skipped;guard=int(math.ceil(self.das.max_delay_samples))+2;ok=self.previous_steer is not None
                    for i in range(skipped):
                        block_end=self.last_end+(i+1)*hop_n;b=self.ring.read_absolute(block_end-hop_n-2*guard,hop_n+2*guard)
                        if b is None:ok=False;break
                        full,_=self.das.beamform(b,*self.previous_steer,None);self.mono.write(full[guard:guard+hop_n][None,:],end_unix-(end-block_end)/self.ring.fs-guard/self.ring.fs)
                    if not ok:self.shared.continuity_gaps+=1;self.mono.reset_continuity()
                self.last_end=end;steer=(update["tracked_az_deg"],update["tracked_el_deg"]);q={}
                if None not in steer:q=self._emit_beam(frame,frame_n,hop_n,end_unix,steer,"measured" if update["spatial_valid"] else "coast")
                if update["new_state"]=="REACQUIRE":self.mono.reset_continuity();self.shared.continuity_gaps+=1
                self.shared.spatial_count+=1;elapsed=time.monotonic()-self.started;total=(time.perf_counter()-started)*1000
                row={"timestamp":datetime.now().astimezone().isoformat(),"elapsed_s":elapsed,"spatial_frame_id":self.shared.spatial_count,"state":update["new_state"],"uav_confirmed":update["new_state"]!="REACQUIRE","previous_state":previous,"transition_reason":update["transition_reason"],"tracked_az_deg":update["tracked_az_deg"],"tracked_el_deg":update["tracked_el_deg"],"predicted_az_deg":update["predicted_az_deg"],"predicted_el_deg":update["predicted_el_deg"],"az_velocity_dps":update["az_velocity_dps"],"el_velocity_dps":update["el_velocity_dps"],"search_radius_deg":radius,"spatial_score":scan["best_score"],"coherence_factor":coh,"directional_contrast_db":directional,"off_axis_contrast_db":offaxis,"spatial_valid":update["spatial_valid"],"consecutive_spatial_misses":update["consecutive_spatial_misses"],"time_since_last_valid_ms":update["time_since_last_valid_ms"],"num_directions_evaluated":scan["num_directions"],"local_scan_ms":scan["duration_ms"],"beamforming_ms":q.get("processing_ms"),"total_spatial_ms":total,"frame_age_ms":max(0,(time.time()-end_unix)*1000),"output_continuity_gap_count":self.shared.continuity_gaps,"spatial_update_hz":self.shared.spatial_count/max(elapsed,1e-6)};self.logger.log("track.csv",row);self.shared.set("spatial",row)
        except Exception as exc:self.shared.spatial_error=f"{type(exc).__name__}: {exc}"

class SemanticWorker:
    def __init__(self,mono,engine,session,shared,logger):self.mono=mono;self.engine=engine;self.session=session;self.shared=shared;self.logger=logger;self.thread=threading.Thread(target=self.run,name="single-track-samid",daemon=True);self.started=time.monotonic()
    def start(self):self.thread.start()
    def run(self):
        last_end=None;last_completion=None
        try:
            while not self.shared.stop.is_set():
                item=self.mono.latest(80000,last_end)
                if item is None:time.sleep(.01);continue
                audio,start,end,end_unix=item
                if last_end is not None and end-last_end<40000:time.sleep(.01);continue
                self.mono.acknowledge(end);last_end=end;x=audio[0];prepared=resample_poly(x,16000,80000).astype(np.float32)[-16000:];scores,feature_ms,infer_ms=self.engine.infer_batch_16k(prepared[None,:]);completion=time.monotonic();hz=math.nan if last_completion is None else 1/max(completion-last_completion,1e-6);last_completion=completion;self.shared.semantic_count+=1;spatial,_,beam,_=self.shared.snapshot();score=float(scores[0]);system="UAV_CONFIRMED_TRACK" if score>=THRESHOLD else "UAV_ID_LOW_CONFIDENCE"
                row={"timestamp":datetime.now().astimezone().isoformat(),"elapsed_s":completion-self.started,"semantic_frame_id":self.shared.semantic_count,"waveform_start_time":end_unix-1,"waveform_end_time":end_unix,"waveform_age_ms":max(0,(time.time()-end_unix)*1000),"tracker_state":spatial.get("state"),"uav_system_state":system,"tracked_az_deg":spatial.get("tracked_az_deg"),"tracked_el_deg":spatial.get("tracked_el_deg"),"predicted_az_deg":spatial.get("predicted_az_deg"),"predicted_el_deg":spatial.get("predicted_el_deg"),"az_velocity_dps":spatial.get("az_velocity_dps"),"el_velocity_dps":spatial.get("el_velocity_dps"),"search_radius_deg":spatial.get("search_radius_deg"),"spatial_score":spatial.get("spatial_score"),"coherence_factor":spatial.get("coherence_factor"),"directional_contrast_db":spatial.get("directional_contrast_db"),"off_axis_contrast_db":spatial.get("off_axis_contrast_db"),"beam_output_rms":beam.get("output_rms"),"beam_output_peak":beam.get("output_peak"),"beam_output_clip_fraction":beam.get("output_clip_fraction"),"samid_raw_score":score,"samid_raw_present":score>=THRESHOLD,"threshold":THRESHOLD,"semantic_inference_ms":feature_ms+infer_ms,"semantic_update_hz":hz};self.logger.log("semantic.csv",row);self.shared.set("semantic",row)
        except Exception as exc:self.shared.semantic_error=f"{type(exc).__name__}: {exc}"

class SyntheticSource:
    def __init__(self,xyz,fs=80000):
        self.xyz=xyz;self.fs=fs;self.index=0;self.count=0;self.started=None;t=np.arange(fs)/fs;u=np.asarray([np.sin(np.deg2rad(-20)),0,np.cos(np.deg2rad(-20))]);tau=xyz@u/343.;self.fixture=sum(a*np.sin(2*np.pi*f*(t[None,:]+tau[:,None])+k*.31) for k,(f,a) in enumerate(((240,.025),(430,.02),(710,.018),(1100,.014),(1650,.01)))).astype(np.float32)
    def open(self):self.started=time.monotonic()
    def close(self):pass
    def read_frame(self,n):
        from .types import AcousticFrame
        delay=self.started+(self.index+n)/self.fs-time.monotonic()
        if delay>0:time.sleep(delay)
        audio=self.fixture[:,(self.index+np.arange(n))%self.fs].copy();self.index+=n;self.count+=max(1,n//5);return AcousticFrame(audio=audio,fs=self.fs,mic_xyz=self.xyz,timestamp=time.time(),metadata={"datagram_count":self.count,"datagram_sizes":[1280]})

class WavPlaneSource:
    """Realtime-paced exact WAV plane wave, used only by explicit dry-run validation."""
    def __init__(self,xyz,wav_path,fs=80000,az=-20.,el=0.,offset_s=0.):
        from scipy.io import wavfile
        sr,raw=wavfile.read(wav_path);scale=float(max(abs(np.iinfo(raw.dtype).min),np.iinfo(raw.dtype).max)) if raw.dtype.kind in "iu" else 1.;audio=raw.astype(np.float32)/scale;self.mono=resample_poly(audio,fs,sr).astype(np.float32);self.xyz=xyz;self.fs=fs;self.index=int(float(offset_s)*fs)%len(self.mono);self.emitted=0;self.count=0;self.started=None;u=np.asarray([np.cos(np.deg2rad(el))*np.sin(np.deg2rad(az)),np.sin(np.deg2rad(el)),np.cos(np.deg2rad(az))*np.cos(np.deg2rad(el))]);self.delay=(xyz@u)*fs/343.
    def open(self):self.started=time.monotonic()
    def close(self):pass
    def read_frame(self,n):
        from .types import AcousticFrame
        delay=self.started+(self.emitted+n)/self.fs-time.monotonic()
        if delay>0:time.sleep(delay)
        pos=(self.index+np.arange(n,dtype=float))[None,:]+self.delay[:,None];pos%=len(self.mono);left=np.floor(pos).astype(np.int64);frac=pos-left;audio=(self.mono[left]*(1-frac)+self.mono[(left+1)%len(self.mono)]*frac).astype(np.float32);self.index=(self.index+n)%len(self.mono);self.emitted+=n;self.count+=max(1,n//5);return AcousticFrame(audio=audio,fs=self.fs,mic_xyz=self.xyz,timestamp=time.time(),metadata={"datagram_count":self.count,"datagram_sizes":[1280]})

def _key():
    if sys.platform!="win32":return None
    import msvcrt
    return msvcrt.getwch().upper() if msvcrt.kbhit() else None

def run_realtime(config_path:Path,results_root:Path,*,dry_run=False,frames=20)->int:
    config,resolved=load_config(config_path);hw=config["hardware"];cfg=config["semantic_acquire_pipeline"];geom=resolve_geometry_path(config,resolved);xyz=transform_pcb_to_engineering(load_wavefrag_csv(geom),x_sign=int(config["coordinates"]["pcb_x_to_engineering_x_sign"]),y_sign=int(config["coordinates"]["pcb_y_to_engineering_y_sign"]));fs=int(hw["fs"]);model_path=Path(__file__).resolve().parents[2]/"third_party"/"samid_ast_model"/"model.safetensors";actual=hashlib.sha256(model_path.read_bytes()).hexdigest().upper()
    if actual!=MODEL_SHA:raise RuntimeError("BLOCK: OS02 model SHA mismatch")
    bytes_per_sync_sample=128*2;datagram_bytes=int(hw["expected_datagram_bytes"]);samples_per_datagram=datagram_bytes/bytes_per_sync_sample
    if not samples_per_datagram.is_integer():raise RuntimeError("packet contract is not an integer synchronous sample count")
    samples_per_datagram=int(samples_per_datagram);expected_dps=fs/samples_per_datagram;expected_bps=expected_dps*datagram_bytes
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S");session=Path(results_root).resolve()/f"realtime_{stamp}";session.mkdir(parents=True);(session/"diagnostics").mkdir()
    schemas={"acquisition_health.csv":HEALTH_FIELDS,"search_candidates.csv":SEARCH_FIELDS,"track.csv":TRACK_FIELDS,"semantic.csv":SEMANTIC_FIELDS};logger=AsyncCSVLogger(session,schemas,queue_size=int(cfg["logger_queue_size"]));logger.start()
    info={"software_version":VERSION,"package_version":"v0.9","session_start":datetime.now().astimezone().isoformat(),"os":platform.platform(),"python":sys.version,"cpu":platform.processor() or "unavailable","sample_rate":fs,"channels":128,"udp":{"bind_host":hw["udp_host"],"bind_port":hw["udp_port"],"source_ip":hw.get("expected_source_ip"),"source_port":hw.get("expected_source_port"),"expected_datagram_bytes":datagram_bytes,"bytes_per_synchronous_sample":bytes_per_sync_sample,"samples_per_datagram":samples_per_datagram,"expected_datagrams_per_s":expected_dps,"expected_bytes_per_s":expected_bps,"sequence_gap_detection":"unavailable","kernel_drop_counter":"unavailable"},"geometry_path":str(geom),"geometry_sha256":hashlib.sha256(geom.read_bytes()).hexdigest().upper(),"h1":{"raw_order":RAW_CHANNEL_ORDER,"physical_order":PHYSICAL_CHANNEL_ORDER,"version":MAPPING_VERSION,"mapping_hash":mapping_hash(),"mapping_count":1},"beamformer":{"type":"far-field Delay-and-Sum","fractional_delay":"linear interpolation with guards/crossfade","normalization":"fixed 1/128","coherence_factor":"sum_f |sum_m X_m(f) exp(-j2pi f tau_m)|^2 / (128 sum_f sum_m |X_m(f)|^2)","directional_contrast_db":"10log10(candidate coherence / median global-or-local background coherence outside main lobe)","off_axis_contrast_db":"10log10(candidate coherence / median coherence at diagonal probes 25-39 degrees away)"},"search":{"top_k":cfg["search_top_k"],"candidate_min_angular_separation_deg":cfg["candidate_min_angular_separation_deg"],"same_frozen_block_s":1.0,"samid_batch":True},"samid":{"revision":"3a12f618dd8aebf180945bf04ebfeef262d65795","model_sha256":MODEL_SHA,"threshold":THRESHOLD,"window_s":1,"hop_s":.5,"preprocessing":"unchanged pinned AutoFeatureExtractor"},"real_hardware_validated":False}
    (session/"session_info.json").write_text(json.dumps(info,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    engine=FrozenSamidEngine(int(cfg["semantic_torch_threads"]));ring=BoundedRing(128,int(cfg["acquisition_ring_s"]*fs),fs);mono=BoundedRing(1,int(cfg["enhanced_ring_s"]*fs),fs);shared=Shared();dry_wav=Path(os.environ["UAV_DRY_RUN_WAV"]) if dry_run and os.environ.get("UAV_DRY_RUN_WAV") else None;source=(WavPlaneSource(xyz,dry_wav,fs,offset_s=float(os.environ.get("UAV_DRY_RUN_OFFSET_S",0))) if dry_wav else SyntheticSource(xyz,fs)) if dry_run else WaveFragUDPSource(xyz,WaveFragUDPConfig(host=str(hw["udp_host"]),port=int(hw["udp_port"]),channels=128,fs=fs,reorder_vendor_channels=True,recv_bytes=int(hw["receive_buffer_bytes"]),timeout_s=float(hw["udp_timeout_s"]),socket_buffer_bytes=int(hw["socket_buffer_bytes"]),expected_source_ip=hw.get("expected_source_ip"),expected_source_port=int(hw["expected_source_port"]),expected_datagram_bytes=datagram_bytes));source.open();acq=AcquisitionWorker(source,ring,int(cfg["acquisition_chunk_samples"]));das=FarFieldDAS(xyz,fs,float(cfg["speed_of_sound_mps"]),tuple(cfg["spatial_band_hz"]),int(cfg["max_frequency_bins"]));tracker=SpatialTracker(TrackerConfig(**cfg["tracker"]));acquirer=SemanticAcquirer(das,engine,cfg,session/"diagnostics",logger);sp=SpatialWorker(ring,mono,das,tracker,cfg,session,shared,acq,acquirer,logger);sem=SemanticWorker(mono,engine,session,shared,logger);acq.start();sp.start();sem.start();started=time.monotonic();last_health=started;last_datagrams=0
    try:
        import psutil;proc=psutil.Process()
    except Exception:proc=None
    try:
        while not shared.stop.wait(.25):
            elapsed=time.monotonic()-started;spatial,semantic,beam,search=shared.snapshot();key=_key()
            if key=="Q":break
            if key=="S":
                snapshot=ring.latest(fs)
                if snapshot is not None:
                    block,_,end,end_unix=snapshot;np.savez(session/"diagnostics"/f"snapshot_{end}.npz",audio=block,fs=fs,end_unix=end_unix,mapping_version=MAPPING_VERSION,mapping_hash=mapping_hash())
            if time.monotonic()-last_health>=1:
                dt=time.monotonic()-last_health;observed=(acq.datagrams-last_datagrams)/dt;ratio=observed/expected_dps;g=acq.stats();sp_hz=float(spatial.get("spatial_update_hz",0) or 0);sem_hz=float(semantic.get("semantic_update_hz",0) or 0) if np.isfinite(semantic.get("semantic_update_hz",math.nan)) else 0;lag=max(float(spatial.get("frame_age_ms",0) or 0),float(semantic.get("waveform_age_ms",0) or 0));warning="ACQUISITION_RATE_WARNING" if ratio<float(cfg["acquisition_warning_ratio"]) else "OK";logger.log("acquisition_health.csv",{"timestamp":datetime.now().astimezone().isoformat(),"elapsed_s":elapsed,"received_datagrams_total":acq.datagrams,"observed_datagrams_per_s":observed,"expected_datagrams_per_s":expected_dps,"observed_expected_ratio":ratio,"received_bytes_total":acq.bytes,"ring_fill_s":ring.valid/fs,"ring_capacity_s":ring.capacity/fs,"ring_overrun_count":ring.overruns,"snapshot_count":acq.snapshots,"stale_count":acq.stale,"cpu_process_pct":proc.cpu_percent() if proc else "unavailable","ram_bytes":proc.memory_info().rss if proc else "unavailable","spatial_update_hz":sp_hz,"semantic_update_hz":sem_hz,"pipeline_lag_ms":lag,"acquisition_warning":warning,"logger_queue_depth":logger.queue.qsize(),"logger_dropped_rows":logger.dropped,**g});last_health=time.monotonic();last_datagrams=acq.datagrams
            confirmed=bool(spatial.get("uav_confirmed"));state=spatial.get("state","SEARCH");raw=semantic.get("samid_raw_score");raw_text="--" if raw is None else f"{raw:.3f} {'PRESENT' if semantic.get('samid_raw_present') else 'ABSENT'}";uav="CONFIRMED" if confirmed else "NOT CONFIRMED";candidate_count=len(search.get("candidates",[]));ratio_now=(acq.datagrams/max(elapsed,1e-6))/expected_dps;acq_state="OK" if ratio_now>=cfg["acquisition_warning_ratio"] else "ACQUISITION_RATE_WARNING"
            if sys.stdout.isatty():print("\x1b[2J\x1b[H",end="")
            print(f"STATE: {state}   UAV: {uav}\nCandidates: {candidate_count}   AZ/EL: {spatial.get('tracked_az_deg','--')} / {spatial.get('tracked_el_deg','--')}\nSAMID RAW: {raw_text}\nSPATIAL: {spatial.get('spatial_update_hz',0):.2f} Hz   SAMID: {semantic.get('semantic_update_hz',0) if np.isfinite(semantic.get('semantic_update_hz',math.nan)) else 0:.2f} Hz\nCOHERENCE: {spatial.get('coherence_factor','--')}   CONTRAST: {spatial.get('directional_contrast_db','--')} dB\nACQ: {acq_state} ratio={ratio_now:.3f}   RING OVERRUN: {ring.overruns}\nS=snapshot Q=quit")
            if dry_run and elapsed>=float(frames)*float(cfg["spatial_hop_s"]):break
        return 0 if not(acq.error or shared.spatial_error or shared.semantic_error or logger.error) else 2
    finally:
        shared.stop.set();acq.stop();sp.thread.join(timeout=10);sem.thread.join(timeout=10);source.close();logger.close();print(f"Session saved to: {session}");
        for label,error in (("ACQUISITION",acq.error),("SPATIAL",shared.spatial_error),("SEMANTIC",shared.semantic_error),("LOGGER",logger.error)):
            if error:print(label+" ERROR",error)
