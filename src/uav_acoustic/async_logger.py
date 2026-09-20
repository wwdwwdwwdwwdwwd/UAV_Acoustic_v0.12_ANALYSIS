"""Bounded, asynchronous, buffered CSV logging for realtime workers."""
from __future__ import annotations
import csv,queue,threading,time
from pathlib import Path

class AsyncCSVLogger:
    def __init__(self,root:Path,schemas:dict[str,list[str]],queue_size:int=8192,flush_s:float=1.0):
        self.root=Path(root);self.schemas=schemas;self.queue=queue.Queue(maxsize=queue_size);self.flush_s=flush_s;self.stop_event=threading.Event();self.dropped=0;self.error=None;self.thread=threading.Thread(target=self._run,name="async-csv-writer",daemon=True)
    def start(self):self.thread.start()
    def log(self,name:str,row:dict):
        try:self.queue.put_nowait((name,row))
        except queue.Full:self.dropped+=1
    def _run(self):
        files={};writers={};last_flush=time.monotonic()
        try:
            for name,fields in self.schemas.items():
                f=(self.root/name).open("w",newline="",encoding="utf-8-sig",buffering=65536);w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();files[name]=f;writers[name]=w
            while not self.stop_event.is_set() or not self.queue.empty():
                try:name,row=self.queue.get(timeout=.1);writers[name].writerow({k:row.get(k) for k in self.schemas[name]})
                except queue.Empty:pass
                if time.monotonic()-last_flush>=self.flush_s:
                    for f in files.values():f.flush()
                    last_flush=time.monotonic()
        except Exception as exc:self.error=f"{type(exc).__name__}: {exc}"
        finally:
            for f in files.values():f.flush();f.close()
    def close(self):self.stop_event.set();self.thread.join(timeout=5)
