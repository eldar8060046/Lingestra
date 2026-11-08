import sys
sys.path.insert(0,'/home/eldar/Desktop/Lingestra/scripts')
import benchmark_continuous as C
C.EVAL_SEEDS=(9301,9302)
sys.argv=['x','--n','15']+sys.argv[1:]
C.main()
