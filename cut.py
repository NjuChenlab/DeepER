path = r"data\RChIP.intersect.bed"
negpath = r"data\RChIP.intersect.bed"

from loopy.myData import RandPreProcess
from loopy.myData import makeneg

pre = RandPreProcess(5000,(0,0.5),copynum=9,presplit=(0.7,0.2,0.1))
pre.addbed(path)
pre.process()
pre.save("data",filename="rand")

makeneg(r"F:\Development\loopy_deploy\data\neg_5k.bed",r"F:\Development\Final-solution\Package\data",0.10)

pre = RandPreProcess(5000,(0,0.5),copynum=9,presplit=(0.7,0.2,0.1))
pre.addbed(r"F:\Development\Final-solution\Package\data\neg-10.bed")
pre.process()
pre.save("data",filename="rand")

# use bedtools to change bed to fasta , please use -name option