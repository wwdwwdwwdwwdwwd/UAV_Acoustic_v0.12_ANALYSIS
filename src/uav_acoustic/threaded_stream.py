from __future__ import annotations
from collections import deque
import math,threading,time
import numpy as np

class BoundedRing:
    def __init__(self,channels:int,capacity:int,fs:int):self.data=np.empty((channels,capacity),np.float32);self.capacity=capacity;self.fs=fs;self.write_abs=0;self.consumer_abs=0;self.valid=0;self.overruns=0;self.lock=threading.Lock();self.last_write_unix=math.nan
    def write(self,x,unix_time=None):
        value=np.asarray(x,np.float32);n=value.shape[1]
        with self.lock:
            if n>self.capacity:value=value[:,-self.capacity:];n=self.capacity
            if self.write_abs+n-self.consumer_abs>self.capacity:self.overruns+=1;self.consumer_abs=self.write_abs+n-self.capacity
            start=self.write_abs%self.capacity;first=min(n,self.capacity-start);self.data[:,start:start+first]=value[:,:first]
            if first<n:self.data[:,:n-first]=value[:,first:]
            self.write_abs+=n;self.valid=min(self.capacity,self.valid+n);self.last_write_unix=float(unix_time or time.time())
    def latest(self,n,after_abs=None):
        with self.lock:
            end=self.write_abs
            if self.valid<n or (after_abs is not None and end<=after_abs):return None
            start_abs=end-n;idx=start_abs%self.capacity
            out=self.data[:,idx:idx+n].copy() if idx+n<=self.capacity else np.concatenate((self.data[:,idx:],self.data[:,:n-(self.capacity-idx)]),axis=1)
            return out,start_abs,end,self.last_write_unix
    def read_absolute(self,start_abs,n):
        with self.lock:
            if start_abs<self.write_abs-self.valid or start_abs+n>self.write_abs:return None
            idx=start_abs%self.capacity
            return self.data[:,idx:idx+n].copy() if idx+n<=self.capacity else np.concatenate((self.data[:,idx:],self.data[:,:n-(self.capacity-idx)]),axis=1)
    def acknowledge(self,end_abs):
        with self.lock:self.consumer_abs=max(self.consumer_abs,int(end_abs))
    def reset_continuity(self):
        with self.lock:self.valid=0

class AcquisitionWorker:
    """Only this thread calls source.read_frame/recvfrom; it never writes disk."""
    def __init__(self,source,ring,chunk_samples):
        self.source=source;self.ring=ring;self.chunk=int(chunk_samples);self.stop_event=threading.Event();self.error=None;self.datagrams=0;self.bytes=0;self.snapshots=0;self.stale=0;self.skips=0;self.gaps=deque(maxlen=50000);self.started=time.monotonic();self.last_receive=None;self.thread=threading.Thread(target=self._run,name="wavefrag-udp-acquisition",daemon=True)
    def start(self):self.thread.start()
    def _run(self):
        try:
            while not self.stop_event.is_set():
                frame=self.source.read_frame(self.chunk);now=time.monotonic()
                if self.last_receive is not None:self.gaps.append((now-self.last_receive)*1000)
                self.last_receive=now;self.ring.write(frame.audio,frame.timestamp);self.datagrams=int(frame.metadata.get("datagram_count",self.datagrams));sizes=frame.metadata.get("datagram_sizes",[]);self.bytes=self.datagrams*(sizes[-1] if sizes else 0)
        except Exception as exc:self.error=f"{type(exc).__name__}: {exc}"
    def stats(self):
        g=np.asarray(self.gaps,float);return {"max_receive_gap_ms":float(np.max(g)) if len(g) else math.nan,"p95_receive_gap_ms":float(np.quantile(g,.95)) if len(g) else math.nan}
    def stop(self):self.stop_event.set();self.thread.join(timeout=2)
