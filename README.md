# DeepER

## Introduction

DeepER is a deep learning-based tool to predict R-loop forming sequences. The basic framework of DeepER includes one layer of Bi-LSTM and four layers of Bi-LSTM with residual blocks, followed by a fully connected layer activated by softmax. Base-level probability of R-loop fromation will be predicted for a given 5kb-long sequence. A sliding window approach (window size = 200bp and step size = 10bp) is then applied to find R-loop-forming regions, defined as consecutive sequences showing average probability >= 0.947 (default cut_off).

You can predict R-loop formation sites with DeepER web server 
(https://rloopbase.nju.edu.cn/deepr/tool/model/) or pull up a DeepER Docker image to use the model locally.

DeepER contains the following files:

- predata.py (Prepare the genome files and label files)
- config.py (The configuration file of the model, in which you can change the hyperparameters of the model)
- model.py (Model structure)
- main.py (Run model)
- train.py (Code to train the model)
- test.py (Code to validate the model)
- Functions.py (Test model and output of various evaluation indicators)
- def_peaks.py (Define the R-loop regions)

## The training process

Our DeepER training steps are as follows:

### step1.Convert genome file (.fa.gz) into pkl format file.

```bash
python predata.py convert_pkl -gzip_fa_file traindata/genomefile/hg38.fa.gz
-gzip_fa_file (The path to genome file)
```

### step2.Prepare label file

```bash
python predata.py label -pkl_path traindata/hg38_pkl/ -bedfile traindata/RChIP.intersect.bed
-pkl_path (The path to the converted genomic files)
-bedfile (The more accurate R-loop peaks identified by R-ChIP technology in the paper)
```

### step3.Train/Validate/Test the model

Warning: Note that when running the following code, make sure that the `config.py` file contains `checkpoint=''` and `is_train = True`.

```bash
python main.py -width 5000 -label traindata/label_intersect/ -neg traindata/neg_5k.bed -pkl_path traindata/hg38_pkl/
-width (The length of the fragment divided by the genome, which is 5000bp by default in this paper)
-label (the path to label files)
-neg (Carefully selected negative control intervals)
-pkl_path (The path to the converted genomic files)
```

Running the above code yields the following two files:

`Checkpoints_window/` : This file stores the best models produced during the model run.

`img_window/`                  : This file stores each indicator graph during the operation of the model, including loss function, accuracy rate, PR curve, etc.

### step4.Predict the genome-wide R-loops (The output result is the probability value of each base predicted to be R-loop)

Warning: Note that when running the following code, make sure that the `config.py` file contains `checkpoint='./Checkpoints_window/' + name_modle + '/best_model.pth'` and `is_train = None`.

```bash
python main.py -width 5000 -label traindata/label_intersect/ -neg traindata/negtest.bed -pkl_path traindata/hg38_pkl/
The meanings of each parameter options are as above (step3)
```

Running the above code yields the following file:

`predict/`: This file stores the probability values of each chromosome predicted to be R-loop.

### step5.Define the R-loop forming regions

```bash
python def_peaks.py Rloop_position -pre_result_path predict/result/ -prob 0.947 -path_save predict/Res_Bidir_LSTM/
-pre_result_path (The path to predicted result)
-prob (Define the probability value of the R-loop forming regions)
-path_save (The path to the predicted R-loop regions)
```









