# from utils.network import Net
# obj = Net()
#
# dataset = dict()
# dataset["data"]= '/media/risverm/in_ssdv2/01_Datasets/data_arjun_assignment/data.bin'
# dataset["label"] = '/media/risverm/in_ssdv2/01_Datasets/data_arjun_assignment/labels.bin'
# dataset["test"] = '/media/risverm/in_ssdv2/01_Datasets/data_arjun_assignment/test.bin'
# learning_rate = 0.000000001
# epoch = 150
# batch_size = 32
# obj.start_Gpu_training(dataset, batch_size, learning_rate, epoch)

from utils.dataloader import Dataloader
import sklearn
import numpy as np

# dataset = dict()
# dataset["data"]= '/media/risverm/in_ssdv2/01_Datasets/data_arjun_assignment/data.bin'
# dataset["label"] = '/media/risverm/in_ssdv2/01_Datasets/data_arjun_assignment/labels.bin'
# dataset["test"] = '/media/risverm/in_ssdv2/01_Datasets/data_arjun_assignment/test.bin'
# data = Dataloader.loaddata(dataset["data"])
# testData = Dataloader.loaddata(dataset["test"])
# labels = Dataloader.loaddata(dataset["label"])
#
# mean, var, stdv = Dataloader.image_mvsd(data)
# print(mean)
# print(var)
# print(stdv)

X = np.array([[1., 0.], [2., 1.], [0., 0.]])
y = np.array([0, 1, 2])
print(X.shape)
train, label = sklearn.utils.shuffle(X, y, random_state=1)
print(train.shape)
print(label)