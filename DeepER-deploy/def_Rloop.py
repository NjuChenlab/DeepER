import sys

#read
def read(file):
    f = open(file,mode="r")
    arr = []
    header =[]
    for l in f:
        line = l.strip().rstrip(",").split("\t")
        value = line[1].split(",")
        #print(line[0])
        arr.append([float(j) for j in value])
        header.append(line[0])
    return header,arr
    
#定义Rloop区间的方式
def defregion(data01,cut_off):
    re = []
    for i in data01:
      rloop = {}
      count = 0
      length = len(i)-200
      for m in range(0,length+1,10):
          n = m+200
          region = i[m:n]
          if sum(region)/len(region)>=cut_off:
              count += 1
              rloop[count]=[m,n]
      for m in range(1,count):
          x = rloop[m][0]
          y = rloop[m][1]
          z = rloop[m+1][0]
          h = rloop[m+1][1]
          if y>z:
              del rloop[m]
              del rloop[m+1]
              rloop[m+1] = [x,h]
      re.append(list(rloop.values()))
    return re

cut_off = float(sys.argv[1])
porbfile = sys.argv[2]
resultr = sys.argv[3]

#读取
header,re = read(porbfile)
print(len(re))
#根据cut_off定义R-loop区间

xregion = defregion(re,cut_off)

f = open(resultr,'w',encoding="utf-8")
for i in range(len(xregion)):
  rr = ["-".join([str(qq) for qq in q]) for q in xregion[i]]
  st = '\t'.join(rr)
  f.write(f"{header[i]}\t{st}\n")
f.close()



  