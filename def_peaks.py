import argparse,time
from joblib import load,dump
import os,collections
from sparse_vector.sparse_vector import SparseVector

def get_bw(args):
    print(args)
    path_pkl=args.pre_result_path
    chrom=args.chrom
    strand=args.strand
    path_save=args.path_save
    chrom_size=args.chrom_size

    if strand=="forward":
        path_get=path_pkl+chrom+"_for"
    else:
        path_get=path_pkl+chrom+"_rev"
    chr_bd= load(path_get)
    prob = SparseVector(chr_bd[1][:])
    if strand=="forward":
        path_s=path_save+chrom+"_for.bd"
    else:
        path_s = path_save + chrom + "_rev.bd"
    if strand=="forward":
        with open(path_s, 'w') as file:
            for i in range(0, list(prob.indices.shape)[0]-1):
                file.write(chrom + '\t' + str(prob.indices[i]) + '\t' + str(prob.indices[i + 1]) + '\t' + str(prob.data[i]) + '\n')
    else:

        with open(path_s, 'w') as file:
            for i in range(0, list(prob.indices.shape)[0]-1):
                file.write(chrom + '\t' + str(prob.indices[i]) + '\t' +str(prob.indices[i + 1]) + '\t' + str(prob.data[i]*(-1)) + '\n')
    bdg_to_bw="~/soft/bedGraphToBigWig"
    if strand=="forward":
        os.system("%s %s %s %s.bw && mv %s to %s" %(bdg_to_bw,path_s, chrom_size,chrom+'_for',chrom+'_for.bw',path_save))
    else:
        os.system("%s %s %s %s.bw && mv %s to %s" %(bdg_to_bw,path_s, chrom_size, chrom + '_rev',chrom+'_rev.bw',path_save))


def Rloop_peak(args):
    print(args)
    path_pkl = args.pre_result_path
    prob1=args.prob
    path_save=args.path_save

    for file in os.listdir(path_pkl):
        print(file)
        chrom = file.split('_')[0]
        strand1 = file.split('_')[1]
        if strand1 == "for":
            strand = "+"
        else:
            strand = "-"
        path = path_pkl + file
        pre_chrom = load(path)[1]
        print(pre_chrom)
        length = pre_chrom.shape[0]
        name = file + ".bed"
        with open(name, 'w') as file_obj:
            s = 0
            e = 0
            prob = prob1
            d = 0
            for i in range(0, length):
                if pre_chrom[i] >= prob:
                    if d == 0:
                        s = e = i
                        d += 1
                    else:
                        e += 1
                        d += 1
                        if e == length - 1:
                            file_obj.write(
                                chrom + '\t' + str(s) + '\t' + str(e) + '\t' + '.' + '\t' + '0' + '\t' + strand + '\n')
                else:
                    if d == 0:
                        s += 1
                        e = s
                    else:
                        if s != e:
                            file_obj.write(
                                chrom + '\t' + str(s) + '\t' + str(e) + '\t' + '.' + '\t' + '0' + '\t' + strand + '\n')
                        d = 0
    os.system("mv chr*.bed to %s" %(path_save))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='commands')

    convert_parser = subparsers.add_parser(name='convert_to_bw', help='convert to .bw file')
    convert_parser.add_argument('-pre_result_path',dest="pre_result_path",required=True,type=str,help='require the path of predict result')
    convert_parser.add_argument('-chrom',dest="chrom",required=True,type=str,help='require the chrom you want to convert')
    convert_parser.add_argument('-strand',dest="strand",required=True,help='require the strand you want to convert')
    convert_parser.add_argument('-path_save',dest="path_save",required=True,help='require the path of you want to save')
    convert_parser.add_argument('-chrom_size',dest="chrom_size",required=True,help='require the hg38 chroms size file')
    convert_parser.set_defaults(func=get_bw)  # 绑定处理函数

    peak_parser = subparsers.add_parser(name='Rloop_position', help='defining the R-loop regions')
    peak_parser.add_argument('-pre_result_path', dest="pre_result_path", required=True, type=str,
                               help='require the path of predict result')
    peak_parser.add_argument('-prob', dest="prob", type=float,required=True,help='require the optimal classification probability value')
    peak_parser.add_argument('-path_save', dest="path_save", required=True,help='require the path to save peaks files')
    peak_parser.set_defaults(func=Rloop_peak)	# 绑定处理函数

    args = parser.parse_args()
    # 执行函数功能
    args.func(args)

if __name__ == '__main__':
    main()













