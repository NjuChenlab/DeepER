import gzip
from joblib import dump, load
from Bio import SeqIO
import argparse
import numpy as np
import os
from sparse_vector.sparse_vector import SparseVector

def save_chromosome_as_pickle(args):
    gzip_fa_file = args.gzip_fa_file
    with gzip.open(gzip_fa_file, "rt") as f:
        for record in SeqIO.parse(f, "fasta"):
            chromosome_name = record.id
            chromosome_sequence = str(record.seq).upper()

            chromosome = {
                "name": chromosome_name,
                "sequence": chromosome_sequence
            }
            chroms = [f'chr{i}' for i in list(range(1, 23)) + ['X', 'Y']]
            # 将染色体对象保存为 pkl 文件
            if chromosome_name in chroms:
                pickle_file = f"traindata/hg38_pkl/{chromosome_name}.pkl"
                with open(pickle_file, "wb") as pkl_file:
                    dump(chromosome_sequence, pkl_file)

                print(f"Saved {chromosome_name} as {pickle_file}")


def labelfile(args):
    pkl_path = args.pkl_path
    bedfile = args.bedfile
    len_chrom={}
    for file in os.listdir(pkl_path):
        path=pkl_path+file
        chrom=file.split('.')[0]
        len_chrom.update({chrom:len(load(path))})
    print(len_chrom)
    for_label = {}
    for chrom in len_chrom.keys():
        chrom_np = np.zeros(len_chrom[chrom])
        with open(bedfile) as ff:
            for line in ff:
                line = line.strip().split('\t')
                if line[0] == chrom and line[5] == "+":
                    start = int(line[1])
                    end = int(line[2])
                    chrom_np[start:end] = np.maximum(chrom_np[start:end], 1)
        for_label.update({chrom: SparseVector(chrom_np)})
    dump(for_label,f"traindata/label_intersect/label_for")
    rev_label = {}
    for chrom in len_chrom.keys():
        chrom_np = np.zeros(len_chrom[chrom])
        with open(bedfile) as ff:
            for line in ff:
                line = line.strip().split('\t')
                if line[0] == chrom and line[5] == "-":
                    start = int(line[1])
                    end = int(line[2])
                    chrom_np[start:end] = np.maximum(chrom_np[start:end], 1)
        rev_label.update({chrom: SparseVector(chrom_np)})
    dump(rev_label,f"traindata/label_intersect/label_rev")
    print('label down!')


def main():
    # 创建解析器对象
    parser = argparse.ArgumentParser(description='Utility for chromosome operations')

    # 创建子解析器对象
    subparsers = parser.add_subparsers(title='subcommands', dest='subcommand')

    # 创建子命令解析器：save
    parser_save = subparsers.add_parser('convert_pkl', help='Save chromosomes as pickle files')
    parser_save.add_argument('-gzip_fa_file',dest='gzip_fa_file',required=True,help='Path to the .fa.gz file')
    parser_save.set_defaults(func=save_chromosome_as_pickle)

    # 创建子命令解析器：label
    parser_label = subparsers.add_parser('label', help='Generate labels from pickle files')
    parser_label.add_argument('-pkl_path',dest='pkl_path',required=True,help='Path to the pickle files')
    parser_label.add_argument('-bedfile', dest='bedfile',required=True,help='R-loop bed file')
    parser_label.set_defaults(func=labelfile)

    # 解析命令行参数
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()


