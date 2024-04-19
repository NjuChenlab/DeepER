import torch
from loopy import Evaluate
from loopy.myData import DataBuffer
#损失函数等
import sys
from torch.utils.data import DataLoader

modelpara = sys.argv[1]
testposdata = sys.argv[2]   #正例文件
extra_neg_data = sys.argv[3]
logfile = "model.log"

from loopy.Model.DRBiLSTM import DRBiLSTM as Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

testdata = DataBuffer(
    testposdata,
    maxlen=5000,
    zoomrate=1,
    label_mode="si",
    tech_score=None,
    strand=True
)
if extra_neg_data != "None":
    testdata.addneg(
        path=extra_neg_data
    )

test_loader = DataLoader(
    dataset=testdata,
    batch_size=32,
    shuffle=True,
    num_workers=2
)

if __name__ =="__main__":
    re = Evaluate.CommonPlot(test_loader,[Model],[modelpara],device,["hc"],0.05,200,200,param=[[4,2,2,64]],savepath=None)