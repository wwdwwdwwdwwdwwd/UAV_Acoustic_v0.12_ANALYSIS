"""Dedicated WaveFrag UDP acquisition process and lock-free shared-memory ring."""
from __future__ import annotations
from dataclasses import dataclass
import ctypes,math,multiprocessing as mp,os,socket,time
from multiprocessing import shared_memory
import numpy as np

@dataclass
class AcquisitionConfig:
    host:str="0.0.0.0";port:int=6666;source_ip:str|None="192.168.0.100";source_port:int|None=5555
    channels:int=128;fs:int=80000;datagram_bytes:int=1280;socket_buffer_bytes:int=8<<20;ring_s:float=8.0
    affinity:list[int]|None=None;high_priority:bool=True;synthetic:bool=False;synthetic_rate_ratio:float=1.0;replay_path:str|None=None

@dataclass
class WindowSnapshot:
    raw_int16:np.ndarray;start_sample:int;end_sample:int;wall_start_s:float;wall_end_s:float
    nominal_audio_duration_s:float;wall_collection_duration_s:float;audio_clock_ratio:float
    receive_rate_ratio:float;max_receive_gap_ms:float;valid:bool;invalid_reason:str

class SharedRawRing:
    """Raw-order int16 ring. Full audio never crosses a multiprocessing queue."""
    def __init__(self,cfg:AcquisitionConfig,create=True,names:dict|None=None):
        self.cfg=cfg;self.capacity=int(cfg.fs*cfg.ring_s);audio_bytes=self.capacity*cfg.channels*2
        packet_capacity=int(math.ceil(cfg.fs*cfg.ring_s*1.05/(cfg.datagram_bytes/(cfg.channels*2))))
        if create:
            self.audio_shm=shared_memory.SharedMemory(create=True,size=audio_bytes);self.time_shm=shared_memory.SharedMemory(create=True,size=packet_capacity*8);self.packet_index_shm=shared_memory.SharedMemory(create=True,size=packet_capacity*8)
        else:
            self.audio_shm=shared_memory.SharedMemory(name=names["audio"]);self.time_shm=shared_memory.SharedMemory(name=names["times"]);self.packet_index_shm=shared_memory.SharedMemory(name=names["packet_indices"])
        self.packet_capacity=packet_capacity;self.audio=np.ndarray((self.capacity,cfg.channels),dtype='<i2',buffer=self.audio_shm.buf);self.packet_times=np.ndarray(packet_capacity,dtype='<f8',buffer=self.time_shm.buf);self.packet_indices=np.ndarray(packet_capacity,dtype='<i8',buffer=self.packet_index_shm.buf)
        if create:self.audio.fill(0);self.packet_times.fill(np.nan);self.packet_indices.fill(-1)
    def names(self):return {"audio":self.audio_shm.name,"times":self.time_shm.name,"packet_indices":self.packet_index_shm.name}
    def close(self):self.audio_shm.close();self.time_shm.close();self.packet_index_shm.close()
    def unlink(self):self.audio_shm.unlink();self.time_shm.unlink();self.packet_index_shm.unlink()

class AcquisitionCounters:
    def __init__(self):
        self.write_samples=mp.Value(ctypes.c_longlong,0,lock=False);self.consumer_samples=mp.Value(ctypes.c_longlong,-1,lock=False);self.datagrams=mp.Value(ctypes.c_longlong,0,lock=False);self.bytes=mp.Value(ctypes.c_longlong,0,lock=False);self.malformed=mp.Value(ctypes.c_longlong,0,lock=False);self.unexpected_source=mp.Value(ctypes.c_longlong,0,lock=False);self.ring_overwrites=mp.Value(ctypes.c_longlong,0,lock=False);self.last_receive_ns=mp.Value(ctypes.c_longlong,0,lock=False);self.started_ns=mp.Value(ctypes.c_longlong,0,lock=False);self.actual_rcvbuf=mp.Value(ctypes.c_longlong,0,lock=False);self.alive=mp.Value(ctypes.c_int,0,lock=False);self.error_code=mp.Value(ctypes.c_int,0,lock=False)

def _set_resources(cfg:AcquisitionConfig,status_queue):
    result={"pid":os.getpid(),"affinity_requested":cfg.affinity,"affinity_actual":None,"priority":"unchanged","errors":[]}
    try:
        import psutil;p=psutil.Process()
        if cfg.affinity:p.cpu_affinity(cfg.affinity)
        result["affinity_actual"]=p.cpu_affinity()
        if cfg.high_priority:
            p.nice(psutil.HIGH_PRIORITY_CLASS if os.name=="nt" else -5);result["priority"]=str(p.nice())
    except Exception as exc:result["errors"].append(f"resource_policy: {type(exc).__name__}: {exc}")
    return result

def acquisition_entry(cfg:AcquisitionConfig,names:dict,counters:AcquisitionCounters,stop:mp.Event,status_queue):
    ring=SharedRawRing(cfg,create=False,names=names);resource_status=_set_resources(cfg,status_queue);counters.started_ns.value=time.monotonic_ns();counters.alive.value=1
    samples_per_packet=cfg.datagram_bytes//(cfg.channels*2);packet_no=0;write_abs=0
    def commit(x,stamp):
        nonlocal packet_no,write_abs
        n=x.shape[0];consumer=int(counters.consumer_samples.value)
        if consumer>=0 and write_abs+n-consumer>ring.capacity:counters.ring_overwrites.value+=1;counters.consumer_samples.value=write_abs+n-ring.capacity
        start=write_abs%ring.capacity;first=min(n,ring.capacity-start);ring.audio[start:start+first]=x[:first]
        if first<n:ring.audio[:n-first]=x[first:]
        slot=packet_no%ring.packet_capacity;ring.packet_indices[slot]=write_abs;ring.packet_times[slot]=stamp;packet_no+=1;write_abs+=n;counters.write_samples.value=write_abs;counters.datagrams.value=packet_no;counters.bytes.value=packet_no*cfg.datagram_bytes;counters.last_receive_ns.value=time.perf_counter_ns()
    sock=None
    try:
        if cfg.synthetic or cfg.replay_path:
            replay=None
            if cfg.replay_path:
                from scipy.io import wavfile
                from scipy.signal import resample_poly
                sr,value=wavfile.read(cfg.replay_path);integer_input=np.issubdtype(value.dtype,np.integer);value=value.astype(np.float32);value=np.mean(value,axis=1) if value.ndim>1 else value
                if integer_input:value=value/32768.
                replay=resample_poly(value,cfg.fs,sr);replay=np.clip(replay,-1,1);replay_pos=0;resource_status["replay_path"]=str(cfg.replay_path)
            status_queue.put(resource_status);phase=0;period=.01;packets_per_batch=max(1,int(16000*float(cfg.synthetic_rate_ratio)*period));deadline=time.perf_counter()
            while not stop.is_set():
                deadline+=period
                for _ in range(packets_per_batch):
                    if replay is None:v=np.arange(samples_per_packet*cfg.channels,dtype=np.int32)+phase;x=((v%2000)-1000).astype('<i2').reshape(samples_per_packet,cfg.channels);phase+=v.size
                    else:
                        idx=(replay_pos+np.arange(samples_per_packet))%len(replay);mono=(replay[idx]*32767).astype('<i2');x=np.repeat(mono[:,None],cfg.channels,axis=1);replay_pos=(replay_pos+samples_per_packet)%len(replay)
                    commit(x,time.perf_counter())
                delay=deadline-time.perf_counter()
                if delay>0:time.sleep(delay)
        else:
            sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);sock.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,cfg.socket_buffer_bytes);sock.bind((cfg.host,cfg.port));sock.settimeout(.25);counters.actual_rcvbuf.value=sock.getsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF);resource_status["socket_actual_receive_buffer"]=int(counters.actual_rcvbuf.value);status_queue.put(resource_status);buf=bytearray(cfg.datagram_bytes);view=memoryview(buf)
            while not stop.is_set():
                try:n,address=sock.recvfrom_into(view)
                except socket.timeout:continue
                if (cfg.source_ip and address[0]!=cfg.source_ip) or (cfg.source_port and address[1]!=cfg.source_port):counters.unexpected_source.value+=1;continue
                if n!=cfg.datagram_bytes:counters.malformed.value+=1;continue
                commit(np.frombuffer(view[:n],dtype='<i2').reshape(samples_per_packet,cfg.channels),time.perf_counter())
    except Exception as exc:counters.error_code.value=1;status_queue.put({"fatal":f"{type(exc).__name__}: {exc}"})
    finally:
        counters.alive.value=0
        if sock:sock.close()
        ring.close()

class AcquisitionProcess:
    def __init__(self,cfg:AcquisitionConfig):
        self.cfg=cfg;self.ring=SharedRawRing(cfg);self.counters=AcquisitionCounters();self.stop_event=mp.Event();self.status_queue=mp.Queue();self.process=mp.Process(target=acquisition_entry,args=(cfg,self.ring.names(),self.counters,self.stop_event,self.status_queue),name="wavefrag-acquisition",daemon=True)
    def start(self):self.process.start()
    def close(self):
        self.stop_event.set();self.process.join(3)
        if self.process.is_alive():self.process.terminate();self.process.join(1)
        self.ring.close();self.ring.unlink()
    def status_messages(self):
        out=[]
        while not self.status_queue.empty():out.append(self.status_queue.get_nowait())
        return out
    def snapshot(self,duration_s:float,min_rate=.98,min_clock=.98)->WindowSnapshot|None:
        n=int(round(duration_s*self.cfg.fs));end1=int(self.counters.write_samples.value)
        if end1<n:return None
        start_abs=end1-n
        if start_abs<end1-self.ring.capacity:return None
        start=start_abs%self.ring.capacity;first=min(n,self.ring.capacity-start);raw=self.ring.audio[start:start+first].copy() if first==n else np.concatenate((self.ring.audio[start:].copy(),self.ring.audio[:n-first].copy()))
        end2=int(self.counters.write_samples.value)
        if end2-end1>self.ring.capacity-n:return None
        mask=(self.ring.packet_indices>=start_abs)&(self.ring.packet_indices<end1)&np.isfinite(self.ring.packet_times);times=np.sort(self.ring.packet_times[mask]);expected=duration_s*self.cfg.fs/(self.cfg.datagram_bytes/(self.cfg.channels*2));rate=len(times)/max(expected,1);wall=(times[-1]-times[0]+1/16000) if len(times)>1 else math.inf;clock=duration_s/wall if wall>0 and np.isfinite(wall) else 0.;gaps=np.diff(times)*1000;max_gap=float(np.max(gaps)) if len(gaps) else math.inf;valid=rate>=min_rate and clock>=min_clock;reason="" if valid else "ACQUISITION_DISCONTINUITY"
        self.counters.consumer_samples.value=end1
        return WindowSnapshot(raw,start_abs,end1,float(times[0]) if len(times) else math.nan,float(times[-1]) if len(times) else math.nan,duration_s,wall,clock,rate,max_gap,valid,reason)
