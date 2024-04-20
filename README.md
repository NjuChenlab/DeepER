## Introduction
DeepER is a deep learning-based tool to predict R-loop forming sequences. The basic framework of DeepER includes one layer of Bi-LSTM and two layers of Bi-LSTM with residual blocks, followed by a fully connected layer activated by softmax. Base-level probability of R-loop fromation will be predicted for a given 5kb-long sequence.  
loopy is a package we built to train DeepER. Theoretically, user just need to provide a bed file which record the R-loop sequence and a extra negtive bed file and a reference genome. The package will deal them automatically.
You can predict R-loop formation sites with DeepER web server ([https://rloopbase.nju.edu.cn/deepr/tool/model](https://rloopbase.nju.edu.cn/deepr/tool/model/) ) or download DeepER's source code from  our github library to run it locally.
DeepER contains the following files:

- DeepER-Train (all code and data to train the model)
   - loopy (the package which contain all train and evaluate method)
      - Model (all model defination)
         - DRBiLSTM.py (the DeepER's architecture)
         - Unet.py (the Unet's architecture)
         - Model.py (other model such as feedforward network)
      - myData.py (the module to prepare data and read data)
      - Evaluate.py (the module to evaluate the results)
      - Train.py (the module which packages the main training workflow)
      - utils.py (the tool functions)
   - main.py (main training workflow)
   - evaluate_model.py (evaluate the model)
   - make_pos.py (prepare positive data)
   - make_neg.py (prepare extra-negative data)
- DeepER-Deploy (all code to deploy DeepER locally)
   - lib.py (all tools function,use help() in python to see more detail)
   - model.py (the defination of the model)
   - DeepER.pkl (the parameter file of the DeepER model)
   - Script.py (sample script)
   - def_rloop.py (def the rloop region)
- Data (the source data we use)
  - RChIP.intersect.bed (source data)
  - neg_5k.bed (extra negative data)
  - Ind_testing.fasta (Independent testing set)
- DeepER.yml (the conda environment)
- readme.md (this file)

## Create a correspond environment
Warning：please keep sure you have already install anaconda or mini-conda
`conda env create -f DeepER-Train.yml`

## The Train process

### step1. Prepare data
#### Positive data
Use make_pos.py script to preprocess the positive source data 
```bash
python make_pos.py bedfile cutratio outputpath

# bedfile: the positive rloop bed reads
# cutratio: a series of number indicate the splitting ratio of bedfile; like 70;20;10
# outputpath: the directory of ouput files

#example:
python make_pos.py RChIP.intersect.bed 70;20;10 ./data/

```
#### Negative data
Use make_pos.py script to preprocess the negative source data 
```bash
python make_neg.py bedfile cutratio outputpath

# bedfile: the negitive bed reads
# cutratio: a number indicate the retention ratio
# outputpath: the name and path of ouput files

#example:
python make_neg.py neg_5k.bed 10 ./data/negdata.bed

```
#### Get fasta file
The upon step will produce training bedfiles. But user need to transform them into fasta files.
Use command
`bedtools getfasta -fi refgenome -bed bedfile -fo outputpath -s -name`

warnning: Please use -s and -name param to make sure next step can work.

### step2. Train model
Use main.py to train a model..The command is as follow:
```bash
python main.py pos_train_data extra_neg_train_data validation_data validation_extra_neg_data
                para_save_path

# pos_train_data: the positive rloop data to train the model
# extra_neg_data: the extra negative data to train the model
# validation_data: the validation data to indicate the training progress. If validation_data is set to None,the pos_train_data and extra_neg_data will be cut into two part in the ratio of 7:2. the second part will be the validation set
# validation_extra_neg_data: the extra negative data to validate training progress.
# para_save_path: the path which the model parameter saved
```
Use the model.log file to pick the best model (sorry we haven't support to pick it automatically, we'll support it in later version)
### step3. Evaluate model
Use Evaluate.py to evalute the trained model.
```bash
python evaluate.py model_para_path test_pos_data test_extra_neg_data

# model_para_path: the parameter user get in training progress 
# test_pos_data: the positive rloop data to test the model
# test_extra_neg_data: the extra negative data to test the model. Can be set as None.
```
## The Deploy process
Our DeepER deploys steps are as follows:
### Console way
You can use script.py to run the model and get the base probability. The order is as follow.
```bash
python script.py fastapath parapath strand ouputfile1 batchsize

# fastapath: the fasta file path to predict
# parapath: the para of the model.
# strand: the predict strand direction, forward or reverse
# output: the output file path.
# batchsize: the larger number will speed up predict, but we lead to more memory cost.
```
You can use def_rloop.py to process the probability file and get the relative position of each rloop interval in the input fasta file.
```bash
python def_Rloop.py  cutoff probability_file outputfile

# cutoff: the cutoff value
# probability_file: the output file from last step
# outputfile: the output file path
```
### python script way
If user-defined workflow is necessary, we recommand user to write python script themselves.
Here is the way:
#### step1. load the path of parameter and model class
```python
from model.py import DRBiLSTM
para = "DeepER.pkl" 
```
#### step2. read fasta data 
```python
from lib import quickloader
batch = 64
fasta = "inputfastapath"
fa = quickloader(fasta,5000,bacth)
```
#### step3. Predict the result
```python
from lib import PreDict
predict = PreDict(fa,DRBiLSTM,para,"hc","forward",True)
a = predict.getallresults()
```
 a is a list object which save all the result.





