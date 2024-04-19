from loopy.myData import makeneg
import sys

negbed = sys.argv[1]
negratio = float(sys.argv[2])/100
ouputfile = sys.argv[3]

makeneg(negbed,ouputfile,negratio)

