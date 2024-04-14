from lib import quickloader
from lib import PreDict
from lib import reserve
from Model import DRBiLSTM
import sys


fasta = sys.argv[1] #fasta文件
cut_off = float(sys.argv[2]) #0.95
strand = sys.argv[3] #forward
resultp = sys.argv[4]
resultr = sys.argv[5]
bacth = int(sys.argv[6])
mutli = True if sys.argv[7] == "true" else False



fa = quickloader(fasta,bacth)
print("Start Predict")
predict = PreDict(fa,DRBiLSTM,"./DRL-30.pkl.Epoch36.pkl","hc",strand,cut_off,mutli)
a = predict.getallresults()

print("Saving")
for v,k in a.items():
    reserve.save(k[1],v,resultp,mode="a")
    #reserve.save(k[2],v,resultr,mode="a")

print("Finished")

#读取

#re = reserve.read(resultp)
#for i in re:
  #print(i[0],i[1])
  
