import argparse
import numpy as np
from test import isFasta
import torch
import torch.nn as nn
from modelrun import get_dataset_all, visial, rloop, net
import os.path
import re
import requests
import json


def isFasta(text):
    nameList = []
    seqList = []
    if text[0] != '>':
        return False
    # 检测是否有标题行
    records = text.split('>')
    if len(records) < 2:
        return False
    else:
        for i in range(1, len(records)):
            lines = records[i].split('\n')
            if len(lines) < 2:
                return False
            if len(lines[0]) == 0 or lines[0].strip == False:
                return False
            sequence = lines[1]
            if len(sequence) != 5000:
                return False
            for j in sequence:
                if j not in ['A','T','C','G','a','t','c','g']:
                    return False

            nameList.append(lines[0])
            seqList.append(lines[1])
    return (nameList, seqList)

def checkCoordinate(coordinate):
    pattern = r'^chr(1[0-9]|[1-9]|2[0-2]|X|Y|M):\d+-\d+$'
    if re.match(pattern, coordinate):
        # url = 'https://rloopbase.nju.edu.cn/deepr/coordinate'
        url = 'http://114.212.171.238:5000/coordinate'
        data = {'coor': coordinate}
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        result = json.loads(requests.post(url, headers=headers, data=data).text)
        nameList, seqList = [], []
        for name, seq in result.items():
            nameList.append(name)
            seqList.append(seq)
        return nameList, seqList
    else:
        return False


def getArgs():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-i", "--input", help="fasta sequence as input")
    group.add_argument("-f", "--file", help="file in fasta format as input")
    group.add_argument("-C", "--coordinate", help="human genome coordinate as input")

    parser.add_argument('-c', "--cutoff", default=0.947, help="cutoff value, filter RLoop region, between 0 and 1")
    parser.add_argument("-s", "--strand", default="forward", choices=['forward', "reverse"],
                        help="The direction of the strand")

    args = parser.parse_args()
    cutoff = args.cutoff
    strand = args.strand
    if args.input:
        result = isFasta(args.input)
        if result:
            return result, cutoff, strand
        else:
            print('The sequence you input is not in fasta format.')
            return False
    elif args.file:
        if os.path.isfile(args.file):
            with open(args.file, 'r') as f:
                text = f.read()
                result = isFasta(text)
                if result:
                    return result, cutoff, strand
                else:
                    print('The sequence you input is not in fasta format.')
                    return False
        else:
            print("The path you input is not a file.")
            return False
    elif args.coordinate:
        result = checkCoordinate(args.coordinate)
        return result, cutoff, strand
    else:
        print("You need to input at least a seqeunce as input.")
        return False


def main():
    result = getArgs()
    if result:
        (nameList, seqList), cutOff, strand = result
        totalProbabilityList, seqRegionList, xregionList, probabilityPassList = [], [], [], []

        seqDict = dict(zip(nameList, seqList))
        for name, seq in seqDict.items():
            inputs = get_dataset_all(seq, strand)
            test_h = net.init_hidden(len(inputs))
            inputs = torch.from_numpy(inputs)
            test_h = tuple([each.data for each in test_h])
            output, output_test = net(inputs.float(), test_h)
            visialProbability = output[:, 1].detach().numpy().flatten()
            totalProbability = visialProbability.tolist()
            totalProbabilityList.append(visialProbability)

            xregion, textlabel = rloop(totalProbability, cutOff)
            seqChosen = []
            for i in range(len(xregion)):
                seqChosen.append(seq[xregion[i][0]:xregion[i][1]])
            seqRegionList.append(seqChosen)
            xregionList.append(xregion)
            probabilityPass = [totalProbability[xregion[i][0]:xregion[i][1]] for i in range(len(xregion))]
            probabilityPassList.append(probabilityPass)

            position = './predict/{}.png'.format(name)
            visial(visialProbability, xregion, textlabel, position, name)

        # 预测结果保存
        ## 保留三位小数
        for i in range(len(totalProbabilityList)):
            totalProbabilityList[i] = np.around((totalProbabilityList[i]), 3)

        for i in range(len(xregionList)):
            with open('./predict/' + nameList[i] + '.txt', 'w') as f:
                f.write('Sequence Name: ' + nameList[i] + '\n')
                f.write('1. Predicted R-loop regions\n')
                for j in range(len(xregionList[i])):
                    f.write(str(xregionList[i][j][0]) + '-' + str(xregionList[i][j][1] + 1) + ': ' + '\n')
                    text = seqRegionList[i][j]
                    for k in range(0, len(text), 100):
                        f.write(
                            f'{text[k:k + 10]}\t{text[k + 10:k + 20]}\t{text[k + 20:k + 30]}\t{text[k + 30:k + 40]}\t{text[k + 40:k + 50]}\t{text[k + 50:k + 60]}\t{text[k + 60:k + 70]}\t{text[k + 70:k + 80]}\t{text[k + 80:k + 90]}\t{text[k + 90:k + 100]}\n')

                f.write('2. Base-level probility' + '\n')
                for j in range(0, 5000, 10):
                    f.write(str(j) + '-' + str(j + 10) + ': ')
                    for k in range(len(totalProbabilityList[i][j:j + 10])):
                        f.write(str(totalProbabilityList[i][j:j + 10][k]) + ', ')
                    f.write('\n')



if __name__ == "__main__":
    main()

