import sys, time
sys.path.insert(0, "/home/eldar/Desktop/Lingestra"); sys.path.insert(0, "/home/eldar/Desktop/Lingestra/tests")
import test_learning as T
from test_learning import *
from lingestra.app import worker as W
e = Env("test_store_learning_worker_dbg")
raw = stream(fr(QUERY), tail_s=6.0)[0]
print("frames", len(raw))
class Slow:
    last_frame_bgr = None
    def __iter__(self):
        for i, f in enumerate(raw):
            time.sleep(0.015); yield f, i / FPS
    def close(self): pass
W.make_source = lambda spec, loop_speed=1.0: Slow()
wk = W.RecognitionWorker("webcam", None, send_mode="none", templates=e.lib.templates())
got, skipped, evs = [], [], []
wk.learn_ready.connect(got.append); wk.learn_skipped.connect(skipped.append)
wk.events.connect(lambda l: evs.extend([(x.kind, x.label, round(x.t, 2), [c[0] for c in x.candidates][:4]) for x in l]))
wk.failed.connect(lambda m: print("FAILED", m))
wk.start(); t0 = time.time()
while not any(x[0] == "clarify" for x in evs) and time.time() - t0 < 20: T.pump()
print("clarify at wall", round(time.time() - t0, 2), [x for x in evs])
wk.confirm("alpha"); t1 = time.time()
while not got and not skipped and time.time() - t1 < 10 and wk.isRunning(): T.pump()
wk.stop(); wk.wait(5000)
print("got", len(got), "skipped", skipped, "events:", evs)
