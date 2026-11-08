import numpy as np, warnings
warnings.simplefilter('error')
from lingestra.library import *
from lingestra.live.recorder import SignRecorder
lib=SignLibrary()
for shape in [(20,67,3),(1,67,3),(2,67,3)]:
    a=np.full(shape,np.nan,np.float32)
    print(shape, prepare_animation(a).shape)
lib2=SignLibrary()
i=lib2.add_recorded('x',np.full((20,67,3),np.nan,np.float32))
print(lib2.thumbnail_frame(i).shape)
try: lib2.update(i,label='a‌b'); print('ok zwnj')
except Exception as e: print('ZWNJ rejected',e)
r=SignRecorder(countdown_s=0)
r.start(0.0)
for k in range(300): s=r.push(np.full((67,3),np.nan,np.float32),k/25)
print(s)
