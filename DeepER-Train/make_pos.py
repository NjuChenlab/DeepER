import sys
from loopy.myData import RandPreProcess

posdata = sys.argv[1]
posratio = sys.argv[2] # 70;20;10
ouputpath = sys.argv[3] #输出文件夹

pr = [float(i)/100 for i in posratio.split(';')]

# 处理正例
pre = RandPreProcess(5000,(0,0.5),copynum=9,presplit=pr)
pre.addbed(posdata)
pre.process()
pre.save(ouputpath,filename="Rand-pos")
