import numpy as np
import torchfile
class Dataloader:

    def __init__(self):
        pass
    @staticmethod
    def loadMNIST(prefix, folder):
        intType = np.dtype('int32').newbyteorder('>')
        nMetaDataBytes = 4 * intType.itemsize

        data = np.fromfile(folder + "/" + prefix + '-images.idx3-ubyte', dtype='ubyte')
        magicBytes, nImages, width, height = np.frombuffer(data[:nMetaDataBytes].tobytes(), intType)
        data = data[nMetaDataBytes:].astype(dtype='float32').reshape([nImages, width, height])

        labels = np.fromfile(folder + "/" + prefix + '-labels.idx1-ubyte',
                             dtype='ubyte')[2 * intType.itemsize:]

    @staticmethod
    def loaddata(fileName):
        data = torchfile.load(fileName)
        return data

    @staticmethod
    def image_mvsd(image_data):
        nddims = image_data.ndim
        batchsize = image_data.shape[0]
        mean = []
        var = []
        stdv = []
        mean_0, var_0, stdv_0 = 0.0, 0.0, 0.0
        mean_1, var_1, stdv_1 = 0.0, 0.0, 0.0
        mean_2, var_2, stdv_2 = 0.0, 0.0, 0.0
        for b in range(batchsize):
            image = image_data[b]
            if nddims == 4:
                mean_0 += (image.mean(axis=0))
                mean_1 += (image.mean(axis=1))
                mean_2 += (image.mean(axis=2))
                var_0 += image.var(axis=0)
                var_1 += image.var(axis=1)
                var_2 += image.var(axis=2)
                stdv_0 += image.std(axis=0)
                stdv_1 += image.std(axis=1)
                stdv_2 += image.std(axis=2)


            elif nddims == 3:
                mean_0 += (image.mean())
                var_0 += image.var()
                stdv_0 += image.std()


            else:
                print("Image should be with batch dimension")
        if nddims == 4:
            mean_0 /= float(batchsize)
            mean_1 /= float(batchsize)
            mean_2 /= float(batchsize)
            var_0 /= float(batchsize)
            var_1 /= float(batchsize)
            var_2 /= float(batchsize)
            stdv_0 /= float(batchsize)
            stdv_1 /= float(batchsize)
            stdv_2 /= float(batchsize)
            mean.append(mean_0)
            mean.append(mean_1)
            mean.append(mean_2)
            var.append(var_0)
            var.append(var_1)
            var.append(var_2)
            stdv.append(stdv_0)
            stdv.append(stdv_1)
            stdv.append(stdv_2)

        elif nddims == 3:
            mean_0 /= float(batchsize)

            var_0 /= float(batchsize)

            stdv_0 /= float(batchsize)
            mean.append(mean_0)
            var.append(var_0)
            stdv.append(stdv_0)

        return mean, var, stdv

    @staticmethod
    def toHotEncoding(data):
        hotEncoding = np.zeros([len(data),
                                np.max(data) + 1])
        hotEncoding[np.arange(len(hotEncoding)), data] = 1
        return hotEncoding


