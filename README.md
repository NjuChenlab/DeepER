## Introduction
DeepER is a deep learning-based tool to predict R-loop forming sequences. The basic framework of DeepER includes one layer of Bi-LSTM and two layers of Bi-LSTM with residual blocks, followed by a fully connected layer activated by softmax. Base-level probability of R-loop fromation will be predicted for a given 5kb-long sequence.  
DeepER-deploy is the most simple version to deploy DeepER model. 
You can predict R-loop formation sites with DeepER web server ([https://rloopbase.nju.edu.cn/deepr/tool/model/](https://rloopbase.nju.edu.cn/deepr/tool/model/)) or download DeepER's source code from XXX runing it locally.

DeepER-deploy contains the following files:

- lib.py (All tools function,use help() in python to see more detail)
- model.py (The defination of the model)
- DeepER.pkl (The parameter file of the DeepER model)
- Script.py (Sample script)


## The Deploy process
Our DeepER deploy steps are as follows:
### step1. load the path of parameter and model class
```python
from model.py import DRBiLSTM
para = "DeepER.pkl" 
```
### step2. read fasta data 
```python
from lib import quickloader
batch = 64
fasta = "inputfastapath"
fa = quickloader(fasta,5000,bacth)
```
### step3. Predict the result
Use method a to get all result.

```
predict = PreDict(fa,DRBiLSTM,para,"hc","forward",0.95,True)
a = predict.getallresults()
```


### Another way
you can use the sample script to run the model. The order is as follow
`python script.py fastapath cutoffvalue strand ouputfile1 ouputfile2 batchsize True`
Outputfile2 param is useless now, it will be useful in later version.
