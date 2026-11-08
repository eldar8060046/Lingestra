import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'scratch/cont'); sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import tune
seeds=tuple(int(x) for x in sys.argv[1].split(','))
d=tune.build(seeds=seeds,n=30)
tune.evaluate(d)
