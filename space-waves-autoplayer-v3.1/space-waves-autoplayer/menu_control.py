"""Local OCR with exact menu labels, stable-frame confirmation, and cooldowns."""
from dataclasses import dataclass
import queue
import re
import threading
import time
import cv2
import numpy as np

ALLOWED={'giveup':0,'next':1,'nextlevel':1,'retry':2,'tryagain':2,'restart':2,'playagain':2,'continueplaying':3,'nothanks':3}


def _merge_split_labels(rows):
    """Merge OCR rows likely split off the same button label (e.g. 'GIVE' and
    'UP' recognized as two regions in a tight/small button) into one row, so
    the ALLOWED check below sees the full text. Rows far apart, or on a
    different line, are left untouched."""
    items=[]
    for row in rows or []:
        box,text,score=row
        points=np.asarray(box)
        x,y=np.floor(points.min(axis=0)).astype(int)
        x2,y2=np.ceil(points.max(axis=0)).astype(int)
        items.append(dict(x=x,y=y,x2=x2,y2=y2,text=str(text),score=float(score)))
    n=len(items)
    parent=list(range(n))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i in range(n):
        for j in range(i+1,n):
            a,b=items[i],items[j]
            height=min(a['y2']-a['y'],b['y2']-b['y'])
            if height<=0:continue
            overlap=min(a['y2'],b['y2'])-max(a['y'],b['y'])
            gap=max(a['x']-b['x2'],b['x']-a['x2'])
            if overlap>.5*height and gap<=max(10,height*.6):
                ri,rj=find(i),find(j)
                if ri!=rj:parent[ri]=rj
    groups={}
    for i in range(n):groups.setdefault(find(i),[]).append(items[i])
    merged=[]
    for group in groups.values():
        group.sort(key=lambda it:it['x'])
        x=min(it['x'] for it in group);y=min(it['y'] for it in group)
        x2=max(it['x2'] for it in group);y2=max(it['y2'] for it in group)
        text=' '.join(it['text'] for it in group)
        score=min(it['score'] for it in group)
        merged.append(([[x,y],[x2,y],[x2,y2],[x,y2]],text,score))
    return merged

@dataclass
class Button:
    label:str
    box:tuple
    score:float
    @property
    def center(self):
        x,y,w,h=self.box
        return x+w/2,y+h/2


def buttons_from_ocr(rows,shape):
    h,w=shape[:2];out=[]
    for row in _merge_split_labels(rows):
        box,text,score=row
        label=re.sub(r'[^a-z]','',str(text).lower())
        if label not in ALLOWED or float(score)<.86:continue
        points=np.asarray(box)
        x,y=np.floor(points.min(axis=0)).astype(int)
        x2,y2=np.ceil(points.max(axis=0)).astype(int)
        if not (0<=x<x2<w and .08*h<y<y2<.96*h):continue
        if not (12<=y2-y<=h*.20 and 18<=x2-x<w*.80):continue
        out.append(Button(label,(x,y,x2-x,y2-y),float(score)))
    return sorted(out,key=lambda b:ALLOWED[b.label])


class Confirmation:
    def __init__(self):
        self.previous=None;self.when=-100.;self.last_click=-100.

    def observe(self,button,now):
        ready=False
        if button and self.previous:
            ready=(button.label==self.previous.label and
                   np.hypot(*(np.array(button.center)-self.previous.center))<12 and
                   .15<now-self.when<2.0 and now-self.last_click>2.0)
        self.previous=button;self.when=now
        return ready

    def clicked(self,now):
        self.last_click=now;self.previous=None


def same_button(button,old,current):
    if old.shape!=current.shape:return False
    x,y,w,h=button.box
    x0=max(0,x-6);y0=max(0,y-6);x1=min(old.shape[1],x+w+6);y1=min(old.shape[0],y+h+6)
    a=old[y0:y1,x0:x1].astype(np.int16)
    b=current[y0:y1,x0:x1].astype(np.int16)
    return a.size>0 and np.mean(np.abs(a-b))<16


class MenuWorker:
    def __init__(self):
        self.pending=queue.Queue(maxsize=1)
        self.results=queue.Queue(maxsize=1)
        self.stop=threading.Event()
        self.error=None
        self.thread=threading.Thread(target=self.run,daemon=True)
        self.thread.start()

    def submit(self,frame,now):
        if self.pending.full():return
        self.pending.put_nowait((frame.copy(),now))

    def _process_one(self,engine,frame,stamp):
        """One frame's worth of OCR + result publishing. Split out from run()
        so a single bad frame can't kill menu detection for the rest of the
        session, and so this step is testable without a real OCR engine or
        background thread."""
        rows,_=engine(frame,use_cls=False)
        buttons=buttons_from_ocr(rows,frame.shape)
        result=(buttons[0] if buttons else None,frame,stamp)
        try:self.results.get_nowait()
        except queue.Empty:pass
        self.results.put_nowait(result)

    def run(self):
        try:
            from rapidocr_onnxruntime import RapidOCR
            engine=RapidOCR(intra_op_num_threads=1,inter_op_num_threads=1,use_cls=False)
        except Exception as error:
            self.error=str(error);return
        while not self.stop.is_set():
            try:frame,stamp=self.pending.get(timeout=.2)
            except queue.Empty:continue
            try:
                self._process_one(engine,frame,stamp)
                self.error=None
            except Exception as error:
                # A single bad frame (odd shape, transient OCR hiccup) should
                # not permanently disable menu detection for the rest of the
                # run; only a failure to load the engine itself is fatal.
                self.error=str(error)

    def poll(self):
        try:return self.results.get_nowait()
        except queue.Empty:return None

    def close(self):self.stop.set()
