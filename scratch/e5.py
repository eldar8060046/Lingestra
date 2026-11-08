import sys
sys.path.insert(0,'tests'); sys.path.insert(0,'.')
from test_continuous import *
rec=Recognizer(templates=templates(list(SIGNS)))
rec.words=['a',object()]; rec.pending={'t':0,'candidates':[],'source':'personal','hole':rec.words[1]}
rec.suspend(True); rec.suspend(False)
print(rec.words, rec.pending)
print(rec.confirm('x')[0].label, rec.words)
rec.words=['a',object()]; rec.pending=None
rec.undo_last_word(); print(rec.words)
