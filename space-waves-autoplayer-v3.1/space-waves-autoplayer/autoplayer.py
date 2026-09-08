"""Space Waves visual autoplayer v2. No game files or memory are modified."""
from __future__ import annotations
import argparse
import ctypes
import json
import math
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
WIDTH = 800


from navigation import Vision, Planner, HazardTracker
from menu_control import MenuWorker, Confirmation, same_button


class Motion:
    def __init__(self):
        self.previous = None
        self.last_player = None
        self.last_time = None
        self.vx = WIDTH * .28
        self.slope = 1.0
        self.response = 0.0
        self.last_scroll = -100.0
        self.shift = 0.0
        self._window = None
        self._window_shape = None

    def update(self, frame, player, now):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # Ignore HUD, arrow, and trail; correlate the obstacle area ahead.
        crop = gray[round(.14*h):round(.94*h), round(.47*w):round(.94*w)]
        small = cv2.resize(crop, None, fx=.5, fy=.5).astype(np.float32)
        if self.previous is not None and self.previous.shape == small.shape:
            dt = now - self.last_time
            if .006 < dt < .15:
                # The capture region's size is effectively constant frame to
                # frame; recomputing this window every call wastes cycles
                # that matter on weaker machines.
                shape=(small.shape[1], small.shape[0])
                if self._window_shape != shape:
                    self._window = cv2.createHanningWindow(shape, cv2.CV_32F)
                    self._window_shape = shape
                (dx, dy), response = cv2.phaseCorrelate(self.previous, small, self._window)
                speed = -2*dx/dt
                self.response = response
                if response > .16 and WIDTH*.04 < speed < WIDTH*1.5 and abs(dy) < 3:
                    self.shift = 2*dx
                    self.last_scroll = now
                    self.vx = .88*self.vx + .12*speed
                    if player and self.last_player:
                        vy = abs(player[1]-self.last_player[1])/dt
                        ratio = vy/self.vx
                        if .35 < ratio < 2.5:
                            self.slope = .96*self.slope + .04*ratio
        self.previous = small
        self.last_player = player
        self.last_time = now


def annotate(frame, player, clearance, path, text, danger_threshold=12):
    out=frame.copy()
    if clearance is not None:
        danger=clearance<danger_threshold
        out[danger]=(out[danger]*.55+np.array([10,10,110])*.45).astype(np.uint8)
    if player:
        cv2.circle(out,(round(player[0]),round(player[1])),round(player[2]),(0,255,255),2)
    if len(path)>1:
        cv2.polylines(out,[np.asarray(path,np.int32)],False,(0,255,0),2)
    cv2.rectangle(out,(0,0),(out.shape[1],25),(22,22,22),-1)
    cv2.putText(out,text,(7,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    return out


def replay(args):
    cap=cv2.VideoCapture(args.replay)
    if not cap.isOpened():
        raise RuntimeError("Cannot open recording")
    fps=cap.get(cv2.CAP_PROP_FPS) or 60
    roi=tuple(map(int,args.roi.split(','))) if args.roi else None
    vision,motion,planner=Vision(args.fast),Motion(),Planner(args.margin,args.latency/1000)
    hazards=HazardTracker()
    writer=None
    detected=planned=count=safe_routes=recovery_routes=0
    previous_y=None
    held=False
    speeds=[]
    times=[]
    while True:
        ok,frame=cap.read()
        if not ok: break
        if roi:
            x,y,w,h=roi;frame=frame[y:y+h,x:x+w]
        start=time.perf_counter()
        frame,p,c=vision.read(frame)
        motion.update(frame,p,count/fps)
        planner.hazards=hazards.update(vision.hazards,count/fps,motion.vx)
        action,path,status=(None,[],"Arrow not found")
        if p:
            detected+=1
            action,path,status=planner.choose(p,c,motion.vx,motion.slope,held)
            speeds.append(motion.vx)
            if action is not None: planned+=1
            if status=="Route found":safe_routes+=1
            else:recovery_routes+=1
        # Recorded input is unknown. Infer direction of the visible arrow next
        # time, rather than pretending the suggested controls affected replay.
        if p and previous_y is not None:
            dy=p[1]-previous_y
            if abs(dy)>.7: held=dy<0
        if p: previous_y=p[1]
        times.append(time.perf_counter()-start)
        if args.output:
            out=annotate(frame,p,c,path,f"REPLAY ONLY | {count/fps:.2f}s | {status}",
                         (p[2]+args.margin) if p else 12)
            if writer is None:
                writer=cv2.VideoWriter(args.output,cv2.VideoWriter_fourcc(*'mp4v'),fps,(out.shape[1],out.shape[0]))
            writer.write(out)
        count+=1
    cap.release()
    if writer: writer.release()
    report=dict(frames=count,arrow_detected=detected,commands_suggested=planned,
                full_routes=safe_routes,recovery_routes=recovery_routes,
                median_processing_ms=round(float(np.median(times))*1000,2),
                p95_processing_ms=round(float(np.percentile(times,95))*1000,2),
                estimated_scroll_pixels_per_second=round(float(np.median(speeds)),2) if speeds else None,
                note="Offline perception/planning check only. Does not measure autonomous completion or input latency.")
    print(json.dumps(report,indent=2))


def accepted_game_titles(extra=None):
    """Window-title substrings accepted as 'this is the game tab'. Kept as a
    plain function (no Windows APIs) so the matching rule itself is testable
    independent of live() and its win32 dependencies."""
    titles=['space waves','spacewaves']
    if extra:titles.append(extra.lower())
    return titles


def title_matches(title_text,accepted_titles):
    return any(s in title_text.lower() for s in accepted_titles)


def selection_is_oversized(w,h,monitor_width,monitor_height):
    """True when the selected capture rectangle covers more than half the
    monitor -- every frame is processed at that size, so this is usually the
    single biggest available speed lever on a slow machine."""
    return w*h>monitor_width*monitor_height*.5


def live(args):
    if sys.platform!='win32':
        raise RuntimeError('Live control supports Windows; replay works elsewhere.')
    try:ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:ctypes.windll.user32.SetProcessDPIAware()
    import mss
    from pynput import keyboard,mouse
    user32=ctypes.windll.user32
    user32.GetForegroundWindow.restype=ctypes.c_void_p
    user32.GetWindowTextW.argtypes=[ctypes.c_void_p,ctypes.c_wchar_p,ctypes.c_int]
    controller=keyboard.Controller();pointer=mouse.Controller()
    key=keyboard.Key.up if args.key=='up' else keyboard.KeyCode.from_char('w')
    gate=threading.RLock()
    state={'armed':False,'quit':False,'held':False,'window':None,'edge':set()}

    def release():
        if state['held']:
            controller.release(key);state['held']=False

    accepted_titles=accepted_game_titles(args.title)

    def game_focused():
        hwnd=user32.GetForegroundWindow()
        title=ctypes.create_unicode_buffer(1024)
        user32.GetWindowTextW(hwnd,title,1024)
        return hwnd==state['window'] and title_matches(title.value,accepted_titles)

    def on_press(k):
        if k not in (keyboard.Key.f8,keyboard.Key.f9):return
        with gate:
            if k in state['edge']:return
            state['edge'].add(k)
            if k==keyboard.Key.f9:
                state['quit']=True;state['armed']=False;release()
                print('\nF9: stopped and released Up.',flush=True)
            elif not state['armed']:
                hwnd=user32.GetForegroundWindow()
                state['window']=hwnd
                if not game_focused():
                    title=ctypes.create_unicode_buffer(1024)
                    user32.GetWindowTextW(hwnd,title,1024)
                    print(f'\nClick the Space Waves tab, then press F8. (Saw window title: "{title.value}" - '
                          'if this really is the game, rerun with --title "some unique part of that text")',flush=True)
                    return
                state['armed']=True
                print('\nRUNNING UNTIL F9. Next / Give up / Retry detection enabled.',flush=True)

    def on_release(k):
        with gate:state['edge'].discard(k)

    listener=keyboard.Listener(on_press=on_press,on_release=on_release)
    with mss.mss() as capture:
        monitor=capture.monitors[args.monitor]
        shot=np.array(capture.grab(monitor))[:,:,:3]
        print('Select the entire colored playfield, excluding the bottom CrazyGames toolbar. Enter confirms.')
        scale=min(1.,1280/shot.shape[1],720/shot.shape[0])
        roi=cv2.selectROI('Select game area',cv2.resize(shot,None,fx=scale,fy=scale),False,False)
        cv2.destroyAllWindows()
        x,y,w,h=(round(v/scale) for v in roi)
        if w<250 or h<180:raise RuntimeError('No usable game area selected')
        if selection_is_oversized(w,h,monitor['width'],monitor['height']):
            print(f'\nHeads up: that selection is {w}x{h}px, over half your screen. Every frame gets '
                  'processed at that size, so a tighter box around just the game playfield (not the '
                  'browser chrome or empty space) is usually the single biggest speed win available. '
                  'Consider re-running and drawing a smaller box, and adding --fast on a slow PC.\n',flush=True)
        region={'left':monitor['left']+x,'top':monitor['top']+y,'width':w,'height':h}
        vision,motion=Vision(args.fast),Motion()
        planner=Planner(args.margin,args.latency/1000)
        hazards=HazardTracker();menu=MenuWorker();confirm=Confirmation()
        listener.start()
        print('\nClick the game, resume/start your level, then press F8 once. F9 is the only stop hotkey.')
        print('Tracking losses and menus no longer disable the bot. Refocusing the game resumes control.')
        print('Keep the game in the selected rectangle. Move the mouse outside it.\n')
        last_seen=-100.;last_log=0.;last_ocr=0.;last_loop=time.perf_counter()
        previous_y=None;last_change=0.;pending_direction=None;delay=args.latency/1000
        good=0;status='Waiting for F8';errors=0;control_dt=1/60
        try:
            while not state['quit']:
                started=time.perf_counter()
                try:
                    raw=np.array(capture.grab(region))
                    frame,p,clearance=vision.read(raw)
                    motion.update(frame,p,started)
                    planner.hazards=hazards.update(vision.hazards,started,motion.vx)
                    if p:
                        good+=1;last_seen=started
                        if previous_y is not None and pending_direction is not None:
                            dy=p[1]-previous_y
                            if abs(dy)>.8 and (dy<0)==pending_direction:
                                measured=started-last_change
                                if .003<measured<.15:delay=.85*delay+.15*measured
                                pending_direction=None
                        previous_y=p[1]
                    else:
                        good=0;previous_y=None
                    with gate:
                        active=state['armed'] and game_focused() and not state['quit']
                        held=state['held']
                        if not active:release()
                    action=None;path=[]
                    if active and p:
                        # Include measured processing age in addition to the observed
                        # command-to-motion delay, bounded to avoid wild prediction.
                        planner.latency=float(np.clip(delay+(time.perf_counter()-started),.008,.15))
                        control_dt=.9*control_dt+.1*float(np.clip(started-last_loop,1/120,.1))
                        planner.control_dt=control_dt
                        action,path,status=planner.choose(p,clearance,motion.vx,motion.slope,held)
                    elif active:status='Waiting for arrow / checking menu (still enabled)'
                    elif state['armed']:status='Waiting for game focus (still enabled)'
                    else:status='Press F8 in the game'
                    if active and started-last_ocr>.55:
                        menu.submit(frame,started);last_ocr=started
                    click=None;result=menu.poll()
                    if result:
                        button,old,stamp=result
                        confirmed=confirm.observe(button,stamp)
                        if (active and confirmed and button and
                            time.perf_counter()-stamp<1.6 and same_button(button,old,frame)):
                            click=button
                    with gate:
                        active=state['armed'] and game_focused() and not state['quit']
                        if not active:release()
                        elif click is not None:
                            release()
                            cx,cy=click.center
                            destination=(round(region['left']+cx*w/frame.shape[1]),
                                         round(region['top']+cy*h/frame.shape[0]))
                            # The OCR text itself is the grounded click target. There
                            # are no guessed menu coordinates or off-game clicks.
                            old_position=pointer.position
                            pointer.position=destination
                            pointer.click(mouse.Button.left,1)
                            pointer.position=old_position
                            confirm.clicked(started)
                            vision=Vision(args.fast);motion=Motion();hazards=HazardTracker();good=0;pending_direction=None
                            previous_y=None;last_seen=-100.;delay=args.latency/1000
                            status='Clicked '+click.label+'; waiting for next gameplay'
                            print(status,flush=True)
                        elif p and action is not None:
                            if action!=state['held']:
                                if action:controller.press(key)
                                else:controller.release(key)
                                state['held']=action;last_change=time.perf_counter();pending_direction=action
                        elif started-last_seen>.10 or started-last_loop>.25:
                            # Release during unknown frames, but never disarm.
                            release()
                    if args.preview:
                        cv2.imshow('Bot v2 - keep outside game',
                                   annotate(frame,p,clearance,path,status,(p[2]+args.margin) if p else 12))
                        cv2.waitKey(1)
                    if started-last_log>2:
                        print(f"{'ARMED' if state['armed'] else 'IDLE'} | {status} | {motion.vx:.0f}px/s | delay {delay*1000:.0f}ms",flush=True)
                        if menu.error:print('Menu OCR unavailable: '+menu.error,flush=True)
                        last_log=started
                    errors=0;last_loop=started
                    time.sleep(max(0,1/120-(time.perf_counter()-started)))
                except Exception as error:
                    with gate:release()
                    errors+=1
                    if errors==1 or errors%20==0:print('Recovering; still enabled: '+str(error),flush=True)
                    time.sleep(.05)
        finally:
            with gate:release()
            menu.close();listener.stop();cv2.destroyAllWindows()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',help='Analyze a recording without sending input')
    p.add_argument('--roi',help='Replay crop x,y,width,height')
    p.add_argument('--output',help='Annotated replay MP4')
    p.add_argument('--monitor',type=int,default=1,help='Screen number, default 1')
    p.add_argument('--key',choices=['up','w'],default='up')
    p.add_argument('--margin',type=float,default=2,help='Extra obstacle clearance in normalized pixels')
    p.add_argument('--latency',type=float,default=25,help='Initial estimated capture/input latency in milliseconds')
    p.add_argument('--preview',action='store_true',help='Show debug view; keep it outside the game')
    p.add_argument('--fast',action=argparse.BooleanOptionalAction,default=True,
                   help='Fast 5x5 obstacle-distance approximation (default: on). Use --no-fast for exact diagonal clearance.')
    p.add_argument('--title',help='Extra window-title text to accept as the game tab (in addition to "space waves"/"spacewaves"), for sites like MSN or other mirrors whose tab title differs')
    args=p.parse_args()
    cv2.setNumThreads(1)
    if args.replay: replay(args)
    else: live(args)


if __name__=='__main__':
    try: main()
    except KeyboardInterrupt: pass
    except Exception as e:
        print(f'ERROR: {e}',file=sys.stderr)
        sys.exit(1)
