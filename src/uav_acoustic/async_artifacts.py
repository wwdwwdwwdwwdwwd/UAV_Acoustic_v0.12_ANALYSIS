from __future__ import annotations
import json,queue,threading
from pathlib import Path
import numpy as np
from scipy.io import wavfile
class AsyncArtifactWriter:
    def __init__(self,root:Path,max_queue=64):self.root=Path(root);self.queue=queue.Queue(maxsize=max_queue);self.stop=threading.Event();self.dropped=0;self.error=None;self.thread=threading.Thread(target=self._run,name="artifact-writer",daemon=True)
    def start(self):self.thread.start()
    def wav(self,relative,fs,audio):
        try:self.queue.put_nowait(("wav",str(relative),int(fs),np.asarray(audio).copy()))
        except queue.Full:self.dropped+=1
    def json(self,relative,data):
        try:self.queue.put_nowait(("json",str(relative),dict(data)))
        except queue.Full:self.dropped+=1
    def _run(self):
        try:
            while not self.stop.is_set() or not self.queue.empty():
                try:item=self.queue.get(timeout=.1)
                except queue.Empty:continue
                path=self.root/item[1];path.parent.mkdir(parents=True,exist_ok=True)
                if item[0]=="wav":wavfile.write(path,item[2],item[3].astype(np.float32))
                else:path.write_text(json.dumps(item[2],indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        except Exception as exc:self.error=f"{type(exc).__name__}: {exc}"
    def close(self):self.stop.set();self.thread.join(10)
