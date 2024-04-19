'''
NAME
    lib.py
--------------------------------------------------------
DESCRIPTION
    This is the lib which contains all the function when deploy the model.

    该库包括部署模型所用的全部函数
--------------------------------------------------------
NAVIGATION
    To know how to load the fasta sequence,read strloader and quickloader class.
    To know how to use model and predict sequence, read Predict class.
    The reserve class present a convenient way to save the result.

    要了解如何加载fasta序列，请阅读strloader和quickloader类。
    要了解如何使用模型和预测序列，请阅读PreDict类。
    reserve类提供了一种保存结果的方便方法。
--------------------------------------------------------
Example:
    from lib import quickloader
    from lib import PreDict
    from lib import reserve
    from Model import DRBiLSTM  # load the model class, use the corresponding model.py
    import sys

    fasta = sys.argv[1] 
    cut_off = float(sys.argv[2])
    strand = sys.argv[3]
    resultp = sys.argv[4]
    resultr = sys.argv[5]
    bacth = int(sys.argv[6])
    mutli = True if sys.argv[7] == "true" else False

    fa = quickloader(fasta,bacth)
    print("Start Predict")
    predict = PreDict(fa,DRBiLSTM,r"\best_model.pkl","hc",strand,cut_off,mutli)
    a = predict.getallresults()

    print("Saving")
    for v,k in a.items():
        reserve.save(k[1],v,resultp,mode="a")
        # reserve.save(k[2],v,resultr,mode="a")
'''
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from torch.nn.parallel import DataParallel


def coding(string,mode="dc",ebsize=1,step=1):
    '''
    Description:
        This function is an API to transfer DNA string to one-hot coding or label coding numpy array.
        该函数将输入的DNA序列编码为np数组
    Args:
        string:str type input,must be consisted of 'A''T''C''G''N';
        mode:coding mode,one-hot coding("hc") or label coding("dc");
        ebsize:furture update content;
        step:furture update content;

        :string:字符串输入,需为A,T,C,G,N组成.
        :mode:编码模式，"hc":独热编码,"dc":标签编码.
    Example:
        >>>seq = "ATC"
        >>>print(coding(seq))
        [[1,0,0,0],[0,0,0,1],[0,1,0,0]]
    '''
    drmap = {
        "N":0,
        "A":1,
        "C":2,
        "G":3,
        "T":4
    }
    htmap = {
        "N":[0,0,0,0],
        "A":[1,0,0,0],
        "C":[0,1,0,0],
        "G":[0,0,1,0],
        "T":[0,0,0,1]
    }
    cd = []
    for i in list(string):
        if mode=="hc":
            cd.append(htmap[i])
        elif mode=="dc":
            cd.append(drmap[i])
    if ebsize==1:
        return torch.from_numpy(np.array(cd))
    
    if step ==0 or step>ebsize:
        raise

    #如果还要嵌入
    eb = []
    j = 0
    while 1:
        a = cd[j:j+ebsize]
        if type(a[0]) == list:
            a = np.array([item for sublist in a for item in sublist])
        eb.append(a)
        j+=step
        if j+ebsize>len(cd):
            break
    return np.array(cd)


def codeall(strings,mode="dc",ebsize=1,step=0):
    '''
    Description
        This function is an API to transform multi sequence to coding numpy array
        该函数接受元组或者列表包含的等长序列组，并返回其整体的编码用于输入模型。
    Args:    
        strings:tuple or list contain DNA sequence to code.the sequence must have the same length.
        mode:see explain in 'coding' function.

        strings:以tuple或list装载的等长DNA序列字符串组.
        mode:见于coding()的用户文档.
    Example:
        >>>seqs = ["ATC","ACT"]
        >>>codeall(seqs,mode="dc")
        [[1,4,3],[1,3,4]]
    '''
    res = []
    for i in strings:
        res.append(coding(i,mode,ebsize,step))
    try:
        val = torch.tensor(np.stack(res,axis=0)).cuda()
    except:
        for i in strings:
          print(len(i))
        raise
    #val= torch.tensor(np.stack(res,axis=0)).cuda()
    # return torch.from_numpy(np.array(res))
    return val


def rev_chain(chain:str):
    '''
    Description:
        get a Reverse complementary sequence of an input sequence.
        该函数接受一个DNA序列字符串,返回其反向互补序列字符串.注意序列本身也会反向.
    Args:
        chain:a sequence of DNA.
    Example:
        >>>seq = "ACT"
        >>>rev_chain(seq)
        AGT
    '''
    dic = {"A":"T","T":"A","G":"C","C":"G","N":"N"}
    chain = chain[::-1]
    re = ""
    for i in range(len(chain)):
        try:
            re += dic[chain[i]]
        except:
            raise ValueError("Unknown base"+chain[i])
    return re

#从文件中读取序列
class quickloader(Dataset):
    '''
    NAME
        quickloader
    ---------------------------------------------
    DESCRIPTION
        This class was disigned to load fasta sequence from the file.
        It will keep the header infomation and sequence in data structure [[headerinfo,seq],[headerinfo,seq],...,[headerinfo,seq]].
        The sequence longer or shorter than expected will be dealed into the expected length(also see in quickloader.Intercept Method)
        User can get the sequence from the index or header info.
    
        该类用来从fasta文件中轻量化读取序列,它以[[headerinfo,seq],[headerinfo,seq],...,[headerinfo,seq]]的形式保存数据。
        比预期长度长或短的序列将被处理为预期长度（另请参阅quickloader.Intercept Method）
        用户可以使用index或者headerinfo来获取所需序列
    ---------------------------------------------
    '''
    def __init__(self,path,cutlen,batch_size=16) -> None:
        '''
        Args:
            path:the path of fasta file;
            cutlen:Sequences longer than the cutlen will be cut into the cutlen.The shorter will be added to this length.
                See more details in quickloader.Intercept method;
            batch_size:The size of parallel operation segments,larger values can accelerate operations, 
                    but will increase memory overhead;
            
            path:fasta文件的路径
            cutlen:较长的序列将被切成这个长度。较短的序列将增加成这个长度。查看quickloader.Intercept方法以见更多细节
            batch_size:并行运算序列的大小,更大的值能加速运算,但是会增加内存开销

        Example:
            >>>path = "/path/usr/1.fasta"
            >>>fasta = quickloader(path,5000,bacth_size=16)
            >>>len(fasta)
            10
            >>>fasta[1]
            ["test_seq1","ACTATGCGGTT"]
        '''
        self.batch_size = batch_size
        self.cutlen = cutlen
        self.obj = []
        with open(path,mode="r",encoding="utf-8") as f:
            for i in f.readlines():
                if i[0]==">":
                    self.obj.append([i[1:len(i)].rstrip("\n"),""])
                    continue
                else:
                    self.obj[-1][1] += i.rstrip("\n")
        self.Intercept()

    # 截取序列
    def Intercept(self):
        '''
        Description:
            this method deal with sequence that its length does not equal to self.cutlen
            when the sequence is longer than self.cutlen,it cut the sequence into pieces which is self.cutlen in length.
            when shorter,it use "N" base to fill the part less than self.cutlen.
            It will automatically use when initialization.
        Args:
            It only needs class instance
        Example:
            >>>path = "/path/usr/2.fasta"
            >>>fasta = quickloader(path,5000,bacth_size=16)
            >>>len(fasta)
            2
            >>>fasta[0]
            ["testseq1(0-4999)","AAATATAATTCAAAAAAAAATT....ATCTTATT"]
            >>>fasta[1]
            ["testseq1(5000-10000)","GGGGCCCATTCAAATCTTA....NNNNNNNNNNN"]
        '''
        newobj = []
        for i in self.obj:
            if len(i[1]) < self.cutlen:
                i[1] = polish_seq(i[1],self.cutlen)
                newobj.append(i)
            elif len(i[1]) > self.cutlen:
                adlabel = 0
                for j in cut_seq(i[1],self.cutlen):
                    newobj.append([i[0]+f"({adlabel}-{adlabel+self.cutlen-1})",j])
                    adlabel = adlabel+self.cutlen
            else:
                newobj.append(i)
        self.obj = newobj

    # 
    def getitems_from_name(self,headerinfo):
        '''
        Description:
            This method get all sequence contain headerinfo sequence.
            It was disigned to get the list of splited sequence.
        
            此方法获取所有包含headerinfo序列的序列。
            设计它是为了将获取拆分的序列的集合。
        Args:
            headerinfo:the header information in the original fasta file.

            headerinfo:最初fasta序列中的headerinfo
        Example:
            >>>path = "/path/usr/2.fasta"
            >>>fasta = quickloader(path,5000,bacth_size=16)
            >>>fasta.getitems_from_name("testseq1")
            [["testseq1(0-4999)","AAATATAATTCAAAAAAAAATT....ATCTTATT"],
            ["testseq1(5000-9999)","GGGGCCCATTCAAATCTTA....NNNNNNNNNNN"]]
        '''
        re = []
        for i in self.obj:
            if headerinfo in i[0]:
                re.append(i)
        return re

    def __getitem__(self,index):
        '''
        Example:
            >>>path = "/path/usr/2.fasta"
            >>>fasta = quickloader(path,5000,bacth_size=16)
            >>>fasta[0][0]
            'testseq1(0-4999)'
            >>>fasta[0][1]
            'AAATATAATTCAAAAAAAAATT....ATCTTATT'
        '''
        return self.obj[index][1],self.obj[index][0]

    def __len__(self):
        return len(self.obj)

#从字符串中读取
class strloader(Dataset):
    '''
    NAME
        strloader
    ---------------------------------------------
    DESCRIPTION
        This class was disigned to load fasta sequence from the fasta formar string.
        It will keep the header infomation and sequence in data structure [[headerinfo,seq],[headerinfo,seq],...,[headerinfo,seq]].
        The sequence longer or shorter than expected will be dealed into the expected length(also see in strloader.Intercept Method)
        User can get the sequence from the index or header info.
    
        该类用来从fasta格式字符串中轻量化读取序列,它以[[headerinfo,seq],[headerinfo,seq],...,[headerinfo,seq]]的形式保存数据。
        比预期长度长或短的序列将被处理为预期长度（另请参阅strloader.Intercept Method）
        用户可以使用index或者headerinfo来获取所需序列
    ---------------------------------------------
    Navigation
        To see how to load the fasta string,please read __init__ method.
        To see how to visit the DNA string and its header information label,please read __getitem__ method and getitems_from_name method.
        To see how to we deal with unfixed length sequence,please check Intercept method.

        请检阅__init__方法，以知如何加载fasta格式字符串
        请检阅__getitem__和getitems_from_name方法得知如何访问加载的fasta中的DNA序列以及其标签
        查看Intercept方法得知我们处理不定长序列的方式
    ---------------------------------------------
    '''
    def __init__(self,fasta_string:str,cutlen=5000,batch_size=16) -> None:
        '''
        Args:
            fasta_string:the strings of fasta format.
            cutlen:the longer sequence will be cut into this length.the shorter will be polished to this length.
                   See more details in quickloader.Intercept method;
            batch_size:The size of parallel operation sequences,larger values can accelerate operations, 
                       but will increase memory overhead;
            
            fasta_string:fasta格式的字符串
            cutlen:较长的序列将被切成这个长度。较短的序列将增加成这个长度。查看quickloader.Intercept方法以见更多细节
            batch_size:并行运算序列的大小,更大的值能加速运算,但是会增加内存开销

        Example:
            >>>fastring = "
            >testseq1\n\
            ATCTCTATTTATTGGC\n\
            >testseq2\n\
            ATCTTTCTTTTCTTTAATC\n\
            "
            >>>fasta = quickloader(path,5000,bacth_size=16)
            >>>len(fasta)
            10
            >>>fasta[1]
            ["testseq1","ATCTCTATTTATTGGC"]
        '''
        self.batch_size = batch_size
        self.cutlen = cutlen
        self.obj = []
        reads = fasta_string.split("\n")
        for i in reads:
            if i[0]==">":
                self.obj.append([i[1:len(i)].rstrip("\n"),""])
                continue
            else:
                self.obj[-1][1] += i.rstrip("\n")
        self.Intercept()

    # 截取序列
    def Intercept(self):
        '''
        Description:
            this method deal with sequence that its length does not equal to self.cutlen
            when the sequence is longer than self.cutlen,it cut the sequence into pieces which is self.cutlen in length.
            when shorter,it use "N" base to fill the part less than self.cutlen.
            It will automatically use when initialization.
        Args:
            It only needs class instance
        Example:
            >>>path = "/path/usr/2.fasta"
            >>>fasta = quickloader(path,5000,bacth_size=16)
            >>>len(fasta)
            2
            >>>fasta[0]
            ["testseq1(0-4999)","AAATATAATTCAAAAAAAAATT....ATCTTATT"]
            >>>fasta[1]
            ["testseq1(5000-10000)","GGGGCCCATTCAAATCTTA....NNNNNNNNNNN"]
        '''
        newobj = []
        for i in self.obj:
            if len(i[1]) < self.cutlen:
                i[1] = polish_seq(i[1],self.cutlen)
                newobj.append(i)
            elif len(i[1]) > self.cutlen:
                adlabel = 0
                for j in cut_seq(i[1],self.cutlen):
                    newobj.append([i[0]+f"|({adlabel}-{adlabel+self.cutlen})",j])
                    adlabel = adlabel+self.cutlen
            else:
                newobj.append(i)
        self.obj = newobj

    def getitems_from_name(self,headerinfo):
        '''
        Description:
            This method get all sequence contain headerinfo sequence.
            It was disigned to get the list of splited sequence.
        
            此方法获取所有包含headerinfo序列的序列。
            设计它是为了将获取拆分的序列的集合。
        Args:
            headerinfo:the header information in the original fasta file.

            headerinfo:最初fasta序列中的headerinfo
        Example:
            >>>path = "/path/usr/2.fasta"
            >>>fasta = quickloader(path,5000,bacth_size=16)
            >>>fasta.getitems_from_name("testseq1")
            [["testseq1(0-4999)","AAATATAATTCAAAAAAAAATT....ATCTTATT"],
            ["testseq1(5000-9999)","GGGGCCCATTCAAATCTTA....NNNNNNNNNNN"]]
        '''
        re = []
        for i in self.obj:
            if headerinfo in i[0]:
                re.append(i)
        return re

    def __getitem__(self,index):
        '''
        Example:
            >>>path = "/path/usr/2.fasta"
            >>>fasta = quickloader(path,5000,bacth_size=16)
            >>>fasta[0][0]
            'testseq1(0-4999)'
            >>>fasta[0][1]
            'AAATATAATTCAAAAAAAAATT....ATCTTATT'
        '''
        return self.obj[index][1],self.obj[index][0]

    def __len__(self):
        return len(self.obj)

# 以pobase将输入序列补充到length长度
def polish_seq(seq,length,pabase="N") -> str:
    '''
    Descrption:
        This is a tool function which pad the sequence to the given length using N base.

        将输入序列使用N补齐到希望长度的函数
    Args:
        seq:input sequence.
        length:the expect length to padding.
        pabase:the base used for paddings.

        seq:输入序列
        length:希望补齐到的长度
        pabase:补齐用的碱基
    Example:
        >>>a = "ATC"
        >>>polish_seq(a,5)
        'ATCNN'
    '''
    assert len(seq)<=length,"There is no need to use this function,want to split it?Use cut_seq() function."
    re = seq
    for i in range(length-len(seq)):
        re += pabase
    return re

# 将较长序列切割成length长度的序列列表
def cut_seq(seq,length) -> list:
    '''
    Description:
        This function takes a sequence and split them into the expect length.
        If the last segmented sequence does not meet this length, use Polish() to complete it
        
        该函数接受一个序列，将其切分为期望的长度，最后一个切分的序列若不满足该长度，使用polish()将其补齐。
    Args:
        seq:the sequence user want to deal with.
        length:the expect length user want to cut the sequence into

        seq:用户想要处理的序列.
        length:用户要将序列剪切到的预期长度.
    Example:
        >>>a = 5000*"A" + 5000*"G" + 4500*"T"
        >>>cut_seq(a)
        ["AAAA...AAAA","GGGG...GGGG","TTTT...NNNN"]
    '''
    re = []
    start = 0
    while 1:    
        re.append(seq[start:start+length] if start+length < len(seq) else polish_seq(seq[start:len(seq)],length))
        if start + length > len(seq):
            break
        else:
            start += length
    return re




#部署使用的类
class PreDict():
    '''
    NAME
        PreDict
    ---------------------------------------------
    Description
        This class is used to predict the sequence from the loader(strloader or quickloader).
        The results will save as the form -- [(headerinfo,point_probability,additional message),...]
        Every result can be visited by using headerinfo.
        
        此类用于从loader（strloader或fastaloader）预测序列.
        结果以[(headerinfo,point_probability,additional message),...]的形式存储
        结果可以使用headerinfo来查询.
    ---------------------------------------------
    Navigation
        To know how to initialize the Predict class and get a easy example,read __init__ method.
        To see how to get the predict result,read getitems and getitem and getallresults method.
    '''
    def __init__(self,data,model,para,codemode,
                 strand,mutli=False,
                 device='cuda' if torch.cuda.is_available() else 'cpu') -> None:
        '''
        Args:
            data:The loader(see in strloader or fastaloader class)
            model:The class(nn.Module) of used model.The define of the model are writen in model.py 
                  Or user can define a model using pytorch themself.
            para:The corresponding para file of the model class.
            codemode: The code mode which the model class take.(either one-hot coding or label coding)
            strand:The sequence model will predict.When user gives 'forward',model will predict the sequence directly.
                   Otherwise,model will predict the reverse complementarity sequence of the given DNA sequence.
            mutli:Whether to use multi-gpu resoucre to speed up.Please keep it False when there is not gpu to use.
            device:The enviornment the model will work on(gpu or cpu).We recommend gpu to speed up prediction.
        
            data:加载程序（请参阅strloader或fastaloader类中的）
            model：所用模型的类（nn.Module）。模型的定义写在model.py中
                   或者用户可以使用pytorch自己定义模型。
            para：模型类的相应para文件。
            codemode：模型类采用的代码模式。（一个热编码或标签编码）
            stand：序列模型会预测。当用户给出“正向”时，模型将直接预测序列。
                   否则，该模型将预测给定DNA序列的反向互补序列。
            mutli：是否使用多gpu资源加速。当没有gpu可供使用时，请将其保留为False。
            device：模型将工作的环境（gpu或cpu）。我们建议gpu加快预测速度。

        Example:
            >>>from Model import DRBiLSTM
            >>>fa = quickloader("path_to_fastafile",16)
            >>>predict = PreDict(fa,DRBiLSTM,r"\best_model.pkl","hc",strand,cut_off,mutli=True)
        '''

        assert strand == "forward" or strand == "reverse", "Please enter correct strand direction"

        loader = DataLoader(
            dataset=data,
            batch_size=data.batch_size,
            shuffle=False,
            num_workers=1
        )
        self.device = device
        self.codemode = codemode
        param = model.default
        self.model = model(*param).to(device)
        self.model.load_state_dict(torch.load(para,map_location=device))
        if mutli:
            self.model = DataParallel(self.model)
        self.strand = strand
        self.model.eval()
        self.re = {}
        self.length = len(data)
        j = 0
        for step,(batch_x,headinfo) in enumerate(loader):
            if self.strand !="forward":
                new = []
                for i in batch_x:
                    new.append(rev_chain(i))
                batch_x = new
            x = codeall(batch_x,codemode).to(device).to(torch.float)
            pred = self.model(x.to(self.device))
            for i in range(pred.shape[0]):
                if self.strand =="forward":
                    self.re[headinfo[i]] = (batch_x[i],pred[i].cpu().detach().numpy())
                else:
                    self.re[headinfo[i]] = (batch_x[i],np.flip(pred[i].cpu().detach().numpy(),axis=0))
            j += len(batch_x)
            print(f"Finished: {j}/{self.length}")

    #重新预测某一链的负链
    def get_rev(self,headinfo):
        '''
        Description:
            Get rev seq predict results of the input seq.
            Please notice that this method is unsafe,if you want to predict rev chain,please use param strand.
            对某条序列的反向互补链预测，但请注意，这个方法是不安全的，
            如有需要，请在一开始预测的时候就设置strand变量为'reverse'
        Example:
            >>>predict = PreDict(fa,DRBiLSTM,r"\best_model.pkl","hc",strand,mutli=True)
            >>>predict.get_rev("testseq1")
        '''
        chain = rev_chain(self.re[headinfo][0])
        x = codeall(chain,self.codemode)
        pred = self.model(x)
        res = (self.re[headinfo][0],pred[0])
        self.re[headinfo+"_rev"] = res
        return res


    def getitems_from_name(self,headerinfo,montage=False):
        '''
        Description:
            This method get all sequence contain headerinfo sequence.
            It was disigned to get the list of splited sequence.
        
            此方法获取所有包含headerinfo序列的序列。
            设计它是为了将获取拆分的序列的集合。
        Args:
            headerinfo:the header information in the original fasta file.

            headerinfo:最初fasta序列中的headerinfo
        Example:
            >>>path = "/path/usr/2.fasta"
            >>>fasta = quickloader(path,5000,bacth_size=16)
            >>>fasta.getitems_from_name("testseq1")
            [["testseq1(0-4999)",np.array([0.1,0.1,0.1,...,0.9,0.99])],
            ["testseq1(5000-9999)",np.array([0.1,0.1,0.1,...,0.0,0.0,0.0]]]
        '''
        res = []
        for v,k in self.re.items():
            if headerinfo in v:
                res.append([v,k])
        return res

    #
    def getitem(self,headerinfo):
        '''
        Description:
            This method get the sequence of the headerinfo sequence.
        
            此方法获取headerinfo对应的序列。
        Args:
            headerinfo:the header information in the original fasta file.

            headerinfo:DNA序列对应的headerinfo
        Example:
            >>>path = "/path/usr/2.fasta"
            >>>fasta = quickloader(path,5000,bacth_size=16)
            >>>fasta.getitems_from_name("testseq1(0-4999)")
            ["testseq1(0-4999)",np.array([0.1,0.1,0.1,...,0.9,0.99)]]
        '''
        try:
            return self.re[headerinfo]
        except:
            print("No seq named")

    # 获取所有预测信息,以
    def getallresults(self):
        '''
        Description:
            return all the predict result for user to analyse.

            返回所有预测结果使得用户可以自己分析
        Example:
            >>>predict.getallresults()
        '''
        return self.re

#
class reserve():
    '''
    NAME:
        reserve
    Description:
        the class to save and read probability.
    '''
    @staticmethod
    def save_preidict(pred:PreDict,filepath):
        '''
        Description:
            save the result of Predict class instance. 
        Args:
            pred:the PreDict class instance.
            filepath:the save path.
        '''
        a = pred.getallresults()
        print("Saving")
        for v,k in a.items():
            reserve.save(k[1],v,filepath,mode="a")

    @staticmethod
    def save(array_:np.ndarray,header,file,mode="a"):
        '''
        Description:
            save the pobability. 
        Args:
            array_:the probability numpy.ndarray.
            header:the information you want to add.
            file:save path.
            model:save mode
        '''
        f = open(file,mode=mode,encoding="utf-8")
        f.write(header+"\t")
        for i in array_:
            f.write(f"{i:.5f}")
            f.write(",")
        f.write("\n")
        f.close()

    @staticmethod
    def read(file,montage=False):
        '''
        :file:读取的文件名
        :montage:是否串接文件结果,不太稳定建议别用,False就行
        '''
        f = open(file,mode="r")
        re = []
        if montage:
            arr = []
        for i in f.readlines():
            lis = i.rstrip("\n").split("\t")
            header = lis[0][0:len(lis[0])]
            if montage:
                t = lis[1].split(",")
                arr += [float(j) for j in t[0:len(t)-1]]
            else:
                t = lis[1].split(",")
                arr = np.array([float(j) for j in t[0:len(t)-1]])
                re.append((header,arr))
        if montage:
            return (np.array(arr))
        else:
            return re

    
if __name__ =="__main__":
    help(quickloader)
    pass