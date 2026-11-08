import sys
sys.path.insert(0, "/home/eldar/Desktop/Lingestra"); sys.path.insert(0, "/home/eldar/Desktop/Lingestra/tests")
import numpy as np
from test_continuous import *
for label, secs, skip in (("old (no long-hold rule, no still-skip)", 999.0, False), ("long-hold rule only", 1.6, False), ("NEW: long-hold rule + skip still frames", 1.6, True)):
    out = []
    for noise in (0.0, 0.02, 0.04):
        for seed in (1, 2, 3):
            rec = Recognizer(None, templates=templates(list(SIGNS))); rec.hold_commit_s = secs
            if not skip:
                rec._skip_still = False
                orig_reset = rec.reset
                # disable the skip by making the flag setter inert
                import types
                def noskip(self_=rec):
                    pass
            items = [(SIGNS["alpha"], {}), (SIGNS["beta"], {})]
            ch, spans = chain(items, trans_s=0.2, noise=noise, seed=seed)
            hold = np.repeat(ch[spans[0][1]][None], int(2.0 * FPS), 0)
            ch = np.concatenate([ch[:spans[0][1] + 1], hold, ch[spans[0][1] + 1:]])
            if not skip:
                class R(Recognizer):
                    @property
                    def _skip_still(self): return False
                    @_skip_still.setter
                    def _skip_still(self, v): pass
                rec = R(None, templates=templates(list(SIGNS))); rec.hold_commit_s = secs
            ev, _ = run(rec, stream(ch, tail_s=2.0)[0])
            out.append(signs_of(ev) == ["alpha", "beta"])
    print(f"{label}: exactly [alpha, beta] in {sum(out)}/{len(out)} runs", flush=True)
