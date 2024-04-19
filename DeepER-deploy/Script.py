from lib import quickloader
from lib import PreDict
from lib import reserve
from Model import DRBiLSTM
import sys


fasta = sys.argv[1] 
parapath = sys.argv[2]
strand = sys.argv[3]
resultp = sys.argv[4]
bacth = int(sys.argv[5])

if __name__ == "__main__":
    fa = quickloader(fasta,5000,bacth)
    predict = PreDict(fa,DRBiLSTM,parapath,"hc",strand,True)
    a = predict.getallresults()

    print("Saving")
    for v,k in a.items():
        reserve.save(k[1],v,resultp,mode="a")

    print("Finished")