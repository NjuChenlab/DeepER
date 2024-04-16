'''
这个模块包括预处理类和数据读取类。还包括其他一些统计数据大致情况的文件\n
预处理类接受bed文件,给出处理好的bed文件\n
数据读取类接受fasta文件,寄存在实例中,以待训练测试过程调用
'''
import random
from random import randint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.utils import shuffle


# 当只需要切分bed文件时，或者可以包含在Preprocess中
def split_bed():
    pass

#实际上,最好书写预处理代码方式是定义一个预处理父类,之后再写吧
class Preprocess():
    '''
    所有预处理共同的父类,
    '''
    def __init__(self,presplit=[1.0,],
                 merge=True,
                 colname=["chr","start","end","name","mid","strand","peak"]
                 ) -> None:
        '''
        :presplit:预切分比例
        :merge:是否融合记录
        -------------------------------
        '''
        # 输入检查
        assert isinstance(merge,bool),"param merge must be bool"
        
        # 主体流程
        self.merge,self.colname = merge,colname
        # 预计算分割百分比
        self.presplit = [0]
        for i in range(len(presplit)):
            self.presplit.append(presplit[i] if i==0 else presplit[i] + presplit[i-1])
        self.orignbed = pd.DataFrame(columns=colname)
        self.tmpbed = []

    def addbed(self,filepath,tech=None,cell=None,sep="\t",header=None):
        '''
        这个方法负责读取数据，并对数据进行预切分
        '''
        # 读取bed文件
        readbed = pd.read_csv(filepath,sep=sep,header=header,comment='#')
        # 随机切分并存储到tmpbed里
        # 随机打乱行顺序
        bed = self.orignbed.reindex(shuffle(self.orignbed.index))

class SlidePrepocess():
    '''
    滑窗数据正例预处理类，通过滑动窗口的方式进行截取
    '''
    def __init__(self,seqlen=5000,padding=5000,step=(200,600),
                 presplit=(0.7,0.2,0.1),
                 merge=True,
                 clean_ratio=0.2,
                 Techlis=("RCHIP"),
                 cellis=(0,),
                 colname:list=["chr","start","end","name","mid","strand","peak"],) -> None:
        '''
        初始化类,声明所需的变量\n
        seqlen:参数接受一个值或者一个二维元组。为截取的序列的长度,如果为一个值即定长序列,如果为两个值即变长序列。\n
        padding:参数接受一个值或者一个二维元组。当给定一个值的时候，为两侧取同样padding，当给定两个值的时候，为左右不同padding\n
        step:参数接受一个值或者一个二维元组。为滑窗截取时的移动步长\n
        presplit:参数接受一个元组或者None。这个参数代表是否在滑窗前对数据进行预先切割,若为元组，按照元组的参数切割，若为None代表不切割\n
        merge:是否将重叠的区域融合到标记中。（之后使用BNN来搞）\n
        clean_ratio:当目标序列被截取的比例低于该值,就抛弃该条记录,若为0则不抛弃\n
        Techlis:所有测序技术的元组,若为None则之后读取数据不包括技术类型\n
        cellis:细胞系的元组，为None时与上面同理\n
        colname:列名，没啥用，相当于标记下形式，然后防止报错的，可以自己随便改。
        ---------------

        '''
        self.processbed = None
        #输入检查
        if not isinstance(seqlen, int) and not isinstance(seqlen, tuple):
            raise TypeError("seqlen must be int or tuple")
        if not isinstance(padding, int) and not isinstance(padding, tuple):
            raise TypeError("padding must be int or tuple")
        if not isinstance(step, int) and not isinstance(step, tuple):
            raise TypeError("step must be int or tuple")
        # if presplit != None and not isinstance(presplit, tuple):
        #     raise TypeError("presplit must be None or tuple")
        if clean_ratio == 1:
            raise ValueError("Are you sure you want to abandon all data?")

        self.seqlen,self.padding,self.step,self.presplit,self.merge,self.clean_ratio = seqlen,padding,step,presplit,merge,clean_ratio
        if Techlis == None:
            self.techlis = None
        else:
            self.techlis = list(Techlis)
            # colname.append("techtype")
        if cellis == None:
            self.cellis = None
        else:
            self.cellis = list(cellis)
            # colname.append("celltype") #暂时先不加细胞名称
        self.orignbed = pd.DataFrame(columns=colname)
        self.colname = colname
        
    def addbed(self,path,tech=None,cell=None,sep="\t",header=None):
        '''
        向self.bed中添加一份bed文件。\n
        path:bed文件的路径\n
        tech:使用的测序技术类型,如果是None代表不需要特别的技术类型标记只读取数据\n
        cell:\n
        sep:bed文件分隔符\n
        header:是否包含行名
        '''
        #输入检查
        if not isinstance(tech, str) and tech != None:
            raise TypeError("tech must be None or str")
        
        bed = pd.read_csv(path,sep=sep,header=header,comment='#')
        bed.columns = self.colname
        #需要加技术类型时的处理
        if self.techlis == None:
            if tech != None:
                raise NameError("tech should be none,if you want add a technology,Please reinitialize and reset parameter 'Techlis'.")
            self.orignbed = pd.concat([bed,self.orignbed])
        else:
            #之后测试下看成功了没
            self.orignbed = pd.concat([bed,self.orignbed])
            bed["Tech"] = tech


    def addtechfile(self,path,readtech=True):
        '''
        读取带有测序技术标记的文件\n
        path:bed文件路径\n
        readtech:是否读取技术类型，若Fasle则不读取技术类型
        '''
        #这个之后再搞,先把之前的写完跑起来

    #
    def process(self,method=None):
        '''
        处理数据的总流程，先根据给定的切分方式切分数据，再对每个切分处理\n
        method:为用户提供了撰写自己数据处理方式的接口，如果为None，就采用默认处理方式。method的编写要求接受dataframe输入，以处理好的dataframe为输出。?再考虑下另外的输入方式        
        '''
        # 随机打乱行顺序
        bed = self.orignbed.reindex(shuffle(self.orignbed.index))
        # 计算分割点
        split_point = [0] + [int(len(bed)*sum(self.presplit[0:i+1])) for i in range(len(self.presplit)-1)]+[len(bed)]
        # 分割成两个Dataframe
        self.processbed = [bed.iloc[split_point[i]:split_point[i+1]] for i in range(len(split_point)-1)]
        for i in range(len(self.processbed)):
            self.processbed[i] = self.single_process(self.processbed[i],method)

    #
    def single_process(self,df:pd.DataFrame,method=None):
        '''
        处理数据，基本思路是分染色体，之后滑窗再将区域内重叠（或否）的序列标记\n
        method为用户提供了撰写自己数据处理方式的接口，如果为None，就采用默认处理方式。method的编写要求接受dataframe输入，以处理好的dataframe为输出。?再考虑下另外的输入方式
        '''
        if method != None:
            self.processbed = method(df)
        #分染色体处理
        chrbeds = {}
        for key, group in df.groupby("chr"):
            chrbeds[key] = group
        print(chrbeds.keys())
        #循环的处理每个染色体
        for chr_,dt in chrbeds.items():
            chrbeds[chr_] = self.slide_window(dt)
        return pd.concat(list(chrbeds.values()))

    def save(self,path,filename=None):
        '''
        储存文件\n
        path:路径\n
        filename:文件名,为None时使用默认命名方式
        '''

        #输入检查

        #默认命名方式，根据不同参数加上不同前缀
        if filename == None:
            filename = ""
            if isinstance(self.seqlen,int):
                filename += f"D{self.seqlen}-L1-"
            else:
                filename += f"{self.seqlen[0]}D{self.seqlen[1]}-L2-"
            if isinstance(self.padding,int):
                filename += f"P{self.padding}-"
            else:
                filename += f"{self.padding[0]}P{self.padding[1]}-"
            if isinstance(self.step,int):
                filename += f"S{self.step}-"
            else:
                filename += f"{self.step[0]}S{self.step[1]}-"
            
            filename += f"M{1 if self.merge else 0}-"            
            filename += f"Slide"
        if isinstance(self.processbed,list):
            j=0
            for i in self.processbed:
                #这里和切分比例关联上，方便分辨类型
                i.to_csv(path+"\\"+filename+f"-{int(self.presplit[j]*100)}"+".bed",sep='\t',index=0,header=0,mode='w')
                j+=1
        elif isinstance(self.processbed,pd.DataFrame):
            self.processbed.to_csv(path+"\\"+filename+f"-{j}"+".bed",sep='\t',index=0,header=0,mode='w')


    #工具函数
    #滑动窗口
    def slide_window(self,df):
        '''
        给一个染色体上的数据,对每一条数据取滑窗,再将标注信息写好\n
        序列标注信息格式：每种标注均使用;分开，其次序为:\n
        1.与当前序列相交的目标序列在染色体上位置.2.相交的目标序列在当前序列上标记.3.技术类型标记.4.细胞类型标记.5.peak标记.
        '''
        re = []
        res = []
        # 变量处理
        if isinstance(self.seqlen,int):
            maxlen = self.seqlen
            minlen = self.seqlen
        else:
            minlen = self.seqlen[0]
            maxlen = self.seqlen[1]
        if isinstance(self.padding,int):
            lpad = self.padding
            rpad = self.padding
        else:
            lpad = self.padding[0]
            rpad = self.padding[1]
        if isinstance(self.step,int):
            maxstep = self.step
            minstep = self.step
        else:
            minstep = self.step[0]
            maxstep = self.step[1]

        for index, row in df.iterrows():
            a = list(row)
            re.append(a)
            start = a[1]-lpad
            end = a[2]+rpad
            rloopst = lpad
            rlooped = lpad + a[2]-a[1]
            rlooplen =  rlooped - rloopst
            k = 0
            while 1:
                if minstep!=maxstep:
                    #采用随机步长
                    sp = randint(minstep,maxstep)
                else:
                    #固定步长
                    sp = minstep
                if maxlen == minlen:
                    #序列等长
                    ln = maxlen  #序列长度
                else:
                    #序列不定长
                    ln = randint(minlen,maxlen)
                pos = start + ln
                if pos>end:
                    break
                b = a[:]  #浅拷贝
                b[1],b[2] = start,pos
                if rloopst >= ln:
                    b[3] = f";{ln}-{ln}"
                    # b.append(ln)
                    # b.append(ln)
                    rloopcover = 0
                elif rloopst < ln and rlooped > ln:
                    b[3] = f"{a[0]}:{a[1]}-{a[2]};{rloopst}-{ln}"
                    # b.append(rloopst)
                    # b.append(ln)
                    rloopcover = ln - rloopst
                else:
                    # b.append(rloopst)
                    # b.append(rlooped)
                    rloopcover = rlooped - rloopst
                    if rloopcover <= 1:
                        b[3] = f";{rloopst}-{rlooped}"
                    else:
                        b[3] = f"{a[0]}:{a[1]}-{a[2]};{rloopst}-{rlooped}"
                #判断是否为脏数据
                if (rlooplen*self.clean_ratio < rloopcover or rloopcover == 0):
                    res.append(b)
                else:
                    k+=1
                rloopst = rloopst - sp if rloopst -sp>0 else 0
                rlooped = rlooped - sp if rlooped -sp>0 else 0
                start = start+sp
        #重叠区域的处理,这里可能有点问题，之后额外注意一下
        if self.merge:
            for i in range(len(res)):
                lap = SlidePrepocess.overlapping(res[i],re) #这里处理还是没把过短的筛掉
                lp = []
                allchr = []
                for j in lap:
                    lp.append(j[0])
                    lp.append(j[1])
                    allchr.append(j[2])
                if len(lp)!=0:
                    ll = [str(i) for i in lp]
                    ret = "&".join(allchr)+";"+"-".join(ll)
                else:
                    ret = res[i][3]
                res[i][3] = ret
        #之后这里加上对技术还有细胞类型的处理
        if self.techlis!=None:
            pass
        if self.cellis!=None:
            pass
        #这里之后加一下表达量的
        return pd.DataFrame(res)
            
   
    #检查重叠
    @staticmethod
    def overlapping(a,lis):
        '''
        这个函数检查数据记录a是否与lis中的某个记录重叠，并返回重叠的信息\n
        a:数据记录\n
        lis:总数据记录
        '''
        st,ed = a[1],a[2]
        laplis = []
        for i in lis:
            i1,i2 = i[1],i[2]
            if (i1 < ed and i1 >st)or(i2 < ed and i2 >st):
                start = i1 - st if i1-st>0 else 0
                if i2-ed>0:
                    end = ed-st
                else:
                    end = i2-st
                laplis.append((start,end,f"{i[0]}:{i1}-{i2}"))    
        return laplis





#新的数据预处理方式,随机边长处理,在Rloop前后随机一定长度
class RandPreProcess():
    def __init__(self,
                 seqlen=5000,
                 lpad=(0,0.5), #还是给个比例吧
                 copynum=1,
                 merge=True,
                 presplit=[0.9,0.1],
                 colname:list=["chr","start","end","name","mid","strand","peak"],
                 Techlis=None,
                 cellis =None
                 ) -> None:
        '''
        在Rloop两侧随机一段长度进行截取
        '''
        
        self.processbed = None
        self.seqlen,self.padding,self.copynum,self.merge = seqlen,lpad,copynum,merge
        self.presplit = presplit
        if Techlis == None:
            self.techlis = None
        else:
            self.techlis = list(Techlis)
            # colname.append("techtype")
        if cellis == None:
            self.cellis = None
        else:
            self.cellis = list(cellis)
            # colname.append("celltype") #暂时先不加细胞名称
        self.orignbed = pd.DataFrame(columns=colname)
        self.colname = colname
        
    def addbed(self,path,tech=None,cell=None,sep="\t",header=None):
        '''
        向self.bed中添加一份bed文件。\n
        path:bed文件的路径\n
        tech:使用的测序技术类型,如果是None代表不需要特别的技术类型标记只读取数据\n
        cell:\n
        sep:bed文件分隔符\n
        header:是否包含行名
        '''
        #输入检查
        if not isinstance(tech, str) and tech != None:
            raise TypeError("tech must be None or str")
        
        bed = pd.read_csv(path,sep=sep,header=header,comment='#')
        bed.columns = self.colname
        #需要加技术类型时的处理
        if self.techlis == None:
            if tech != None:
                raise NameError("tech should be none,if you want add a technology,Please reinitialize and reset parameter 'Techlis'.")
            self.orignbed = pd.concat([bed,self.orignbed])
        else:
            #之后测试下看成功了没
            self.orignbed = pd.concat([bed,self.orignbed])
            bed["Tech"] = tech


    def addtechfile(self,path,readtech=True):
        '''
        读取带有测序技术标记的文件\n
        path:bed文件路径\n
        readtech:是否读取技术类型，若Fasle则不读取技术类型
        '''
        #这个之后再搞,先把之前的写完跑起来

    #
    def process(self,method=None):
        '''
        处理数据的总流程，先根据给定的切分方式切分数据，再对每个切分处理\n
        method:为用户提供了撰写自己数据处理方式的接口，如果为None，就采用默认处理方式。method的编写要求接受dataframe输入，以处理好的dataframe为输出。?再考虑下另外的输入方式        
        '''
        # 随机打乱行顺序
        bed = self.orignbed.reindex(shuffle(self.orignbed.index))
        # 计算分割点
        split_point = [0] + [int(len(bed)*sum(self.presplit[0:i+1])) for i in range(len(self.presplit)-1)]+[len(bed)]
        # 分割成两个Dataframe
        self.processbed = [bed.iloc[split_point[i]:split_point[i+1]] for i in range(len(split_point)-1)]
        for i in range(len(self.processbed)):
            self.processbed[i] = self.single_process(self.processbed[i],method)

    #
    def single_process(self,df:pd.DataFrame,method=None):
        '''
        处理数据，基本思路是分染色体，之后滑窗再将区域内重叠（或否）的序列标记\n
        method为用户提供了撰写自己数据处理方式的接口，如果为None，就采用默认处理方式。method的编写要求接受dataframe输入，以处理好的dataframe为输出。?再考虑下另外的输入方式
        '''
        if method != None:
            self.processbed = method(df)
        #分染色体处理
        chrbeds = {}
        for key, group in df.groupby("chr"):
            chrbeds[key] = group
        print(chrbeds.keys())
        #循环的处理每个染色体
        for chr_,dt in chrbeds.items():
            chrbeds[chr_] = self.slide_window(dt)
        return pd.concat(list(chrbeds.values()))

    def save(self,path,filename=None):
        '''
        储存文件\n
        path:路径\n
        filename:文件名,为None时使用默认命名方式
        '''

        #输入检查

        #默认命名方式，根据不同参数加上不同前缀
        if filename == None:
            filename = ""
            if isinstance(self.seqlen,int):
                filename += f"D{self.seqlen}-L1-"
            else:
                filename += f"{self.seqlen[0]}D{self.seqlen[1]}-L2-"
            if isinstance(self.padding,int):
                filename += f"P{self.padding}-"
            else:
                filename += f"{self.padding[0]}P{self.padding[1]}-"
            
            filename += f"M{1 if self.merge else 0}-"            
            filename += f"Rand"
        if isinstance(self.processbed,list):
            j=0
            for i in self.processbed:
                #这里和切分比例关联上，方便分辨类型
                i.to_csv(path+"\\"+filename+f"-{int(self.presplit[j]*100)}"+".bed",sep='\t',index=0,header=0,mode='w')
                j+=1
        elif isinstance(self.processbed,pd.DataFrame):
            self.processbed.to_csv(path+"\\"+filename+f"-{j}"+".bed",sep='\t',index=0,header=0,mode='w')

    #工具函数
    #滑动窗口
    def slide_window(self,df):
        '''
        给一个染色体上的数据,对每一条数据取滑窗,再将标注信息写好\n
        序列标注信息格式：每种标注均使用;分开，其次序为:\n
        1.与当前序列相交的目标序列在染色体上位置.2.相交的目标序列在当前序列上标记.3.技术类型标记.4.细胞类型标记.5.peak标记.
        '''
        re = []   #原始数据
        res = []  #处理数据
        # 变量处理
        if isinstance(self.seqlen,int):
            maxlen = self.seqlen
            minlen = self.seqlen
        else:
            minlen = self.seqlen[0]
            maxlen = self.seqlen[1]
        
        if isinstance(self.padding,float):
            pad = (self.padding,self.padding)
        else:
            pad = self.padding

        for index, row in df.iterrows():   # 这段代码有问题，在过长序列时肯定会出毛病，但只能先这样了
            a = list(row)
            re.append(a)
            length = a[2] - a[1]      # 这条Rloop数据对应的序列长度
            for i in range(self.copynum):
                intercept = randint(minlen,maxlen)
                rs = intercept-length if intercept > length else 0
                lpad = randint(int(rs*pad[0]),int(rs*pad[1]))            
                rpad = rs - lpad
                start = a[1]-lpad
                end = a[2]+rpad
                rloopst = lpad
                rlooped = lpad + length
                b = a[:]
                pos = start + intercept   #有错
                b[1],b[2] = start,pos
                if rloopst >= intercept:
                    b[3] = f";{maxlen}-{maxlen}"
                    # b.append(ln)
                    # b.append(ln)
                    rloopcover = 0
                elif rloopst < intercept and rlooped > intercept:
                    b[3] = f"{a[0]}:{a[1]}-{a[2]};{rloopst}-{intercept}"
                    # b.append(rloopst)
                    # b.append(ln)
                    rloopcover = intercept - rloopst
                else:
                    # b.append(rloopst)
                    # b.append(rlooped)
                    rloopcover = rlooped - rloopst
                    if rloopcover <= 1:
                        b[3] = f";{rloopst}-{rlooped}"
                    else:
                        b[3] = f"{a[0]}:{a[1]}-{a[2]};{rloopst}-{rlooped}"
                res.append(b)

        #重叠区域的处理,这里可能有点问题，之后额外注意一下
        if self.merge:
            for i in range(len(res)):
                lap = SlidePrepocess.overlapping(res[i],re) #这里处理还是没把过短的筛掉
                lp = []
                allchr = []
                for j in lap:
                    lp.append(j[0])
                    lp.append(j[1])
                    allchr.append(j[2])
                if len(lp)!=0:
                    ll = [str(i) for i in lp]
                    ret = "&".join(allchr)+";"+"-".join(ll)
                else:
                    ret = res[i][3]
                res[i][3] = ret

        #之后这里加上对技术还有细胞类型的处理
        if self.techlis!=None:
            pass
        if self.cellis!=None:
            pass
        #这里之后加一下表达量的
        return pd.DataFrame(res)
            
   
    #检查重叠
    @staticmethod
    def overlapping(a,lis):
        '''
        这个函数检查数据记录a是否与lis中的某个记录重叠，并返回重叠的信息\n
        a:数据记录\n
        lis:总数据记录
        '''
        st,ed = a[1],a[2]
        laplis = []
        for i in lis:
            i1,i2 = i[1],i[2]
            if (i1 < ed and i1 >st)or(i2 < ed and i2 >st):
                start = i1 - st if i1-st>0 else 0
                if i2-ed>0:
                    end = ed-st
                else:
                    end = i2-st
                laplis.append((start,end,f"{i[0]}:{i1}-{i2}"))    
        return laplis




#先这样把
def makeneg(negfile,new,rate):
    f = open(negfile,mode="r")
    a = f.readlines()
    new = open(new,mode="w")
    if isinstance(type,int):
        rate = rate/len(a)
    for i in a:
        if random.randint(0,1000) < 1000*rate:
            lis = i.split("\t")
            lis[3] = ";0-0"
            new.write("\t".join(lis))
        




#读取fasta类型文件的类
class FastaLoader():
    def __init__(self,
                 fastapath,
                 strand=True,
                 seqname=None,
                 comment="#") -> None:
        '''
        读取fasta文件的方法，读取fasta文件的同时，为用户提供处理头信息的方式。\n
        fastapath:fasta文件路径。\n
        strand:Fasta文件是否包含正负链信息\n
        seqname:用户自定义的处理头信息的方法。要求输入为之前打包好的headerinfo，输出为一个list或tuple，其至少按序包括如下信息:\n
        comment:去除注释类型
        '''
        #输入检查
        self.strand =strand
        if comment==">":
            raise ValueError("'comment' can't equal to >,> is the header information of ever seqence")
        self.obj = []   #self.obj的注释，每一数据项为(序列,标头,正负链,目标序列染色体信息)
        with open(fastapath,mode="r",encoding="utf-8") as f:
            for i in f.readlines():
                if i[0] == ">":
                    headerinfo = i[1:len(i)].rstrip("\n")
                    if strand:
                        stnd = headerinfo[-2]
                        #监控是否给链信息
                        if stnd != "+" and stnd != "-":
                            raise NameError("strand information error.please check fasta file or para")
                        headerinfo = headerinfo[-len(headerinfo):-3]
                    else:
                        stnd = None
                    if seqname==None:
                        unpack = FastaLoader.header_unpack(headerinfo)
                        unpack[1] = ";".join(unpack[1]) if unpack[1] !=[""]  else ";"
                        a = ["",unpack[0],stnd]+unpack[1:len(unpack)]   #定义不能用到负参数真奇怪
                        #这里是info顺序的关键
                    else:
                        unpack = seqname(headerinfo)
                        self.obj.append(["",unpack[0],stnd]+unpack[1:len(unpack)])   
                elif i[0]==comment:
                    continue
                else:
                    a[0] = i.rstrip("\n")
                    self.obj.append(a)

    def addneg(self,path):
        with open(path,mode="r",encoding="utf-8") as f:
            for i in f.readlines():
                if i[0] == ">":
                    headerinfo = i[1:len(i)].rstrip("\n")
                    if self.strand:
                        stnd = headerinfo[-2]
                        #监控是否给链信息
                        if stnd != "+" and stnd != "-":
                            raise NameError("strand information error.please check fasta file or para")
                        headerinfo = headerinfo[-len(headerinfo):-3]
                    else:
                        stnd = None
                    unpack = FastaLoader.header_unpack(headerinfo)
                    unpack[1] = ";".join(unpack[1]) if unpack[1] !=[""]  else ";"
                    a = ["",unpack[0],stnd]+unpack[1:len(unpack)]
                else:
                    a[0] = i.rstrip("\n")
                    self.obj.append(a)
    #
    def rdabandon(self,rate):
        if rate >1 or rate <0:
            raise ValueError("rate should be between 0 and 1")
        new = []
        for i in range(len(self.obj)):
            b = self.obj[i]
            a = randint(0,100)
            if a<int(100*rate):
                continue
            else:
                new.append(b)
        self.obj = new

    #根据染色体信息建立序列索引
    def makeindex(self):
        self.indexdic = {}
        for i in range(len(self.obj)):
            a = self.obj[i]
            chrinfo = a[3]
            for j in chrinfo:
                if j in self.indexdic.keys():
                    self.indexdic[j].append(i)
                else:
                    self.indexdic[j] = [i]

    #根据所给的染色体位置信息，找到所有对应的index,必须先建索引
    def get_index(self,chrpos:str):
        try:
            indexs = self.indexdic[chrpos]
            return indexs
        except:
            #如果不存在的处理
            print("Not Found",chrpos)
            return None

    #根据染色体位置寻找所有对应序列
    def seqfrom_chrpos(self,chrpos:str):
        indexs = self.get_index(chrpos)
        if indexs==None:
            #对不存在情况的处理
            raise ValueError(chrpos,"不存在于当前数据集")
        return self.getseq(indexs)
    
    #详细信息
    def allfrom_chrpos(self,chrpos:str):
        indexs = self.get_index(chrpos)
        if indexs==None:
            #对不存在情况的处理
            raise ValueError(chrpos,"不存在于当前数据集")
        return self.getitemall(indexs)  #之后还是得重新搞下API

    #获得序列的方法
    def getseq(self,index):
        if isinstance(index,int):
            return self.obj[index][1]
        elif isinstance(index,tuple) or isinstance(index,list):
            re = []
            for i in index:
                re.append(self.obj[i][1])
            return re

    #返回全部详细信息
    #不知道为啥只能接受int,不然会报错
    def getitemall(self,index):
        # if isinstance(index,int):
        return self.obj[index]
        # elif isinstance(index,tuple) or isinstance(index,list):
        #     re = []
        #     for i in index:
        #         re.append(self.obj[i])
        #     return re

    #根据index返回序列的方式
    def __getitem__(self,index):
        # if isinstance(index,int):
        return self.obj[index][0]
        # elif isinstance(index,tuple) or isinstance(index,list):
        #     re = []
        #     for i in index:
        #         re.append(self.obj[i][0])
        #     return re

    def __len__(self):
        return len(self.obj)

    #融合两个东西
#     def __add__(self,other):
#         self.obj = self.obj + other.obj

    #################
    #设置内容
    def setitem(self,index,obj):
        #还得写错误处理
        self.obj[index] = obj

    def __str__(self) -> str:
        return f"totol data:{len(self.obj)};\n"+"contain tuple (string,(start_postion_rloop,end_postion_rloop))"


    #解包序列标注信息
    @staticmethod
    def header_unpack(headerinfo:str):
        '''
        详细的headerinfo的打包方式见于Preprocess类
        '''
        re = []
        lis = headerinfo.split(";")
        pos = [int(j) for j in lis[1].split("-")]
        chrinfo = lis[0].split("&")
        #后面三种标记还没开发，之后记得写

        re.append(pos)
        re.append(chrinfo)
        return re
    

class DataBuffer(Dataset):
    '''
    数据读取类,读取整理好的
    '''
    def __init__(self,
                 path,
                 maxlen,
                 zoomrate=1,
                 label_mode="si",
                 tech_score=None,
                 strand=True,
                 colname:list=["chr","start","end","name","mid","strand","peak"],) -> None:
        '''
        解析Fasta文件\n
        path:fasta文件路径。\n
        maxlen:序列最大长度\n
        zoomrate:序列放大倍数，例如maxlen=5000下rate=2，则label返回2500长度的标记，如果参数为"all",则只返回该段序列是否包含目标序列。\n
        label_mode:返回的label的形式，如果为"si"就只返回1或0，若"bi"就返回[1,0]或[0,1]。\n
        tech_score:对不同技术的置信度赋分,（之后可以写用boosting算法自动生成）\n
        colname:见bedProcess
        '''
        super().__init__()
        #输入检查
        self.mode,self.maxlen,self.rate,self.label_mode,self.tech_score,self.strand = label_mode,maxlen,zoomrate,label_mode,tech_score,strand
        self.fasta = FastaLoader(path,strand)
        
    #额外负例的添加
    def addneg(self,path,saverate=0):
        '''
        path:负例样本路径
        saverate:保留负例的比例，为了保持正负例均衡,如果设置为"default"则为保留和正例一样多的负例
        '''
        self.fasta.addneg(path=path)
        # negfasta = FastaLoader(path,strand=self.strand)
        # if saverate!=1 and saverate!="default":
        #     negfasta.rdabandon(saverate)
        # elif saverate == "default":
        #     rate = len(self.fasta)/len(negfasta)
        #     if rate<1:
        #         negfasta.rdabandon(1-rate)
        # self.fasta = self.fasta + negfasta


    #与FastaLoader相同的接口
    def makeindex(self):
        self.fasta.makeindex()
    def getindex(self,chrpos:str):
        return self.fasta.get_index(chrpos)
    def seqfrom_chrpos(self,chrpos:str):
        return self.fasta.seqfrom_chrpos(chrpos)
    def allfrom_chrpos(self,chrpos):
        return self.fasta.allfrom_chrpos(chrpos)
    #返回全部详细信息
    def getitemall(self,index):
        return self.fasta.getitemall(index)

    def __str__(self) -> str:
        return f"totol data:{len(self.fasta)};\n"+"contain tuple (string,(start_postion_rloop,end_postion_rloop))"
    
    def __len__(self):
        return len(self.fasta)
  
    def __getitem__(self,index):
        info = self.getitemall(index)
        b = makelabel(info[1],self.maxlen,info[2],self.rate,self.mode)
        return info[0],b,info[3] #把chr信息也一并返回


#新接口,实现了可变的label
def makelabel(tp:tuple,maxlen,chain,rate=1,mode="si"):
    #输入tp为位置标记
    if mode=="si":
        i = 0
        re = np.zeros((maxlen//rate))
        while 1:
            re[tp[i]//rate:tp[i+1]//rate] = 1
            i+=2
            if i>=len(tp):
                break
        if chain=="+":
            return re
        elif chain=="-":
            return np.flip(re,(0,)).copy()
        return re
    elif mode=="bi":
        i = 0
        re = np.zeros(((maxlen//rate),2))
        re[:,1] = 1
        while 1:
            re[tp[i]//rate:tp[i+1]//rate,0] = 1
            re[tp[i]//rate:tp[i+1]//rate,1] = 0
            i+=2
            if i>=len(tp):
                break
        if chain=="+":
            return re
        elif chain=="-":
            return np.flip(re,(0,)).copy()  #要copy不然会报错
        return re
