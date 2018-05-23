import os
import random
from scipy.misc import imread, imresize
import numpy as np

class datacontainer:
    def __init__(self, train_set_ratio):
        nlkind = 6 ###########define night light kinds here
        path = ['data/NL_%d' % i for i in range(1, nlkind + 1)]
        # print(path)
        self.imgs = {}
        self.labels = {}
        for i in range(nlkind):
            imgstemp = []
            labelstemp = []
            filelist = os.listdir(path[i])
            for each in filelist[:]:  #####todo here to limit the data size
                im = imread(path[i] + '/' + each, mode='RGB')
                imgstemp.append(im)
                label = [0 for _ in range(nlkind)]
                label[i] = 1
                labelstemp.append(label)

            l = len(imgstemp)
            shuffleindex = [j for j in range(l)]
            random.seed(1)
            random.shuffle(shuffleindex)
            self.imgs[i] = [imgstemp[shuffleindex[j]] for j in range(l)]
            # self.labels[i] = [labelstemp[shuffleindex[i]] for i in range(l)]
            self.labels[i] = labelstemp ##since they all belong to the same class, no necessary to shuffle

        # print(self.imgs)
        # print(self.labels)

        self.trainimgs = []
        self.trainlabels = []
        self.testimgs = []
        self.testlabels = []
        for i in range(nlkind):
            l = len(self.imgs[i])
            train_set_l = int(l*train_set_ratio)
            self.trainimgs += self.imgs[i][:train_set_l]
            self.trainlabels += self.labels[i][:train_set_l]
            self.testimgs += self.imgs[i][train_set_l:]
            self.testlabels += self.labels[i][train_set_l:]


        # print(self.trainimgs)
        # print(self.trainimgs)
        for eachkey in self.imgs:
            self.imgs[eachkey] = np.array(self.imgs[eachkey])
        self.trainimgs = np.array(self.trainimgs)
        self.trainlabels = np.array(self.trainlabels)
        self.testimgs = np.array(self.testimgs)
        self.testlabels = np.array(self.testlabels)

        l = len(self.trainimgs)
        shuffleindex = [j for j in range(l)]
        random.shuffle(shuffleindex)
        self.trainimgs = self.trainimgs[shuffleindex, :, :, :]
        self.trainlabels = self.trainlabels[shuffleindex, :]

        # print(111)
        # print(self.trainimgs)

    def getTrainMean(self):
        ########very very very low efficiency code, please use numpy
        # wholemean = [0., 0., 0.]
        # for eachim in self.trainimgs:
        #     eachimmean = [0., 0., 0.]
        #     for eachrow in eachim:
        #         for eachpixel in eachrow:
        #             for i in range(3):
        #                 eachimmean[i] += eachpixel[i]
        #
        #     row = len(eachim)
        #     column = len(eachim[0])
        #     # print(row, column)
        #     for i in range(3):
        #         eachimmean[i] /= (row*column)
        #
        #     for i in range(3):
        #         wholemean[i] += eachimmean[i]
        #
        # l = len(self.trainimgs)
        # for i in range(3):
        #     wholemean[i] /= l
        # #
        # print(wholemean)

        immatrix = np.array(self.trainimgs)
        wholemean = np.mean(immatrix, axis=(0, 1, 2))

        return wholemean

    def getimgsize(self):
        row = len(self.trainimgs[0])
        column = len(self.trainimgs[0][0])
        channel = len(self.trainimgs[0][0][0])
        return (row, column, channel)

    def getlabeldim(self):
        return len(self.trainlabels[0])

if __name__ == '__main__':
    dc = datacontainer(0.7)
    print(dc.getTrainMean())
    print(dc.getimgsize())