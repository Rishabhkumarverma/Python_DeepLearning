from helper import Optimizer
from layer import FullyConnected, Softmax, SoftMaxLoss, Relu
from dataloader import Dataloader
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle

class Net:
    def __init__(self):
        self.parameter = dict()
        self.grad = dict()
        self.gradInput = dict()
        self.Fc1 = FullyConnected()
        self.Fc2 = FullyConnected()
        self.Fc3 = FullyConnected()
        self.Relu1 = Relu()
        self.Relu2 = Relu()
        self.Relu3 = Relu()
        self.softmaxloss = SoftMaxLoss()
        self.optimizer = Optimizer()
    def initWeigth(self):

        # self.parameter['w1'] = np.zeros(shape=(28*28, 32), dtype=np.float32)
        # self.parameter['w2'] = np.zeros(shape=(32, 10), dtype=np.float32)
        # self.parameter['b1'] = np.zeros(shape=(32), dtype=np.float32)
        # self.parameter['b2'] = np.zeros(shape=(10), dtype=np.float32)
        self.parameter['w1'] = np.random.rand(28 * 28, 32) * np.sqrt(1/(28*28+32))
        self.parameter['w2'] = np.random.rand(32, 16) * np.sqrt(1/(32+16))
        self.parameter['w3'] = np.random.rand(16, 10) * np.sqrt(1/(16+10))
        self.parameter['b1'] = np.random.rand(32)
        self.parameter['b2'] = np.random.rand(16)
        self.parameter['b3'] = np.random.rand(10)


        # self.parameter['w1'] = np.random.uniform(low=0.5, high=15, size = (28 * 28, 32))
        # self.parameter['w2'] = np.random.uniform(low=0.3, high=15, size = (32, 16))
        # self.parameter['w3'] = np.random.uniform(low=0.1, high=2, size = (16, 10))
        # self.parameter['b1'] = np.random.uniform(low=0, high=1, size = (32) )
        # self.parameter['b2'] = np.random.uniform(low=0., high=1, size = (16) )
        # self.parameter['b3'] = np.random.uniform(low=0., high=1, size = (10) )



    def ForwardPass(self, image):
        fc1, fca2 = self.Fc1.forward(input=image, weight=self.parameter["w1"],
                                     bias=self.parameter["b1"])  # in = 1x1024, out = 1x32
        relu1, _ = self.Relu1.forward(fc1)
        fc2, _ = self.Fc2.forward(input=relu1, weight=self.parameter["w2"],
                                  bias=self.parameter["b2"])  # in = 1x32 , out = 1x10
        relu2, _ = self.Relu2.forward(fc2)
        fc3, _ = self.Fc3.forward(input=relu2, weight=self.parameter["w3"],
                                  bias=self.parameter["b3"])  # in = 1x32 , out = 1x10
        relu3, _ = self.Relu3.forward(fc3)
        out = Softmax.forward(relu3)
        return out

    def BackwordPass(self, prediction):
        dout = self.softmaxloss.backward(prediction)
        dout = self.Relu3.backward(dout)
        delta3, dw3, db3 = self.Fc3.backward(dout=dout)
        relu_delta2 = self.Relu2.backward(delta3)
        delta2, dw2, db2 = self.Fc2.backward(dout=relu_delta2)
        relu_delta1 = self.Relu1.backward(delta2)
        delta1, dw1, db1 = self.Fc1.backward(relu_delta1)
        self.grad['dw1'] = dw1
        self.grad['dw2'] = dw2
        self.grad['dw3'] = dw3
        self.grad['db1'] = db1
        self.grad['db2'] = db2
        self.grad['db3'] = db3

    def Train(self, image, label):
        data = np.array(image, dtype=np.float32)
        data = data.flatten()
        input_data = np.reshape(data, (1, data.size))
        label = np.reshape(label, (1, label.size))
        output = self.ForwardPass(input_data)
        # print(output)
        loss = self.softmaxloss.forward(output, label)
        # print(loss)
        #print(loss)
        self.BackwordPass(output)
        return loss, output


    def Test(self,testImages, testLabelsHotEncoding):
        numberOfImages = testImages.shape[0]
        # numberOfImages = 10
        avgloss = 0
        avgacc = 0
        count = 0
        for iter in range(numberOfImages):
            image = testImages[iter, :, :]
            labels = testLabelsHotEncoding[iter, :]
            data = np.array(image, dtype=np.float32)
            data = data.flatten()
            input_data = np.reshape(data, (1, data.size))
            labels = np.reshape(labels, (1, labels.size))
            output = self.ForwardPass(input_data)
            pred = np.argmax(output[0])
            gt = np.argmax(labels[0])
            loss = self.softmaxloss.forward(output, labels)
            if pred == gt:
                count+=1
                #print("True")
            avgloss += loss
        avgacc = float(count) /  float(numberOfImages)
        avgloss = float(avgloss) / float(numberOfImages)
        print("Test Accuracy: ", avgacc)
        print("Test Loss: ", avgloss)
        return avgloss, avgacc

    def start_training_mnist(self,data_folder, batch_size, learning_rate, NumberOfEpoch, display=1000):
        self.initWeigth()
        self.optimizer.initADAM(3, 3)
        trainingImages, trainingLabels = Dataloader.loadMNIST('train', data_folder)
        testImages, testLabels = Dataloader.loadMNIST('t10k', data_folder)
        trainLabelsHotEncoding = Dataloader.toHotEncoding(trainingLabels)
        testLabelsHotEncoding = Dataloader.toHotEncoding(testLabels)
        numberOfImages = trainingImages.shape[0]
        # numberOfImages = 10
        pEpochTrainLoss = []
        pEpochTrainAccuracy = []
        pEpochTestLoss = []
        pEpochTestAccuracy = []
        print ("Training started")
        t = 0
        for epoch in range(NumberOfEpoch):
            avgLoss = 0
            trainAcc = 0.0
            count = 0.0
            countacc = 0.0
            pIterLoss = []
            print("##############EPOCH : {}##################".format(epoch))
            for iter in range(numberOfImages):
                t+=1
                image = trainingImages[iter, :, :]
                labels = trainLabelsHotEncoding[iter, :]
                loss, output = self.Train(image, labels)
                self.parameter = self.optimizer.ADAM(self.parameter, self.grad, learning_rate, t)
                self.parameter = self.optimizer.l2_regularization(self.parameter, 0.001)
                output = output[0]
                pred = np.argmax(output)
                gt = np.argmax(labels)
                if pred == gt:
                    count += 1.0
                    countacc += 1.0
                    #print("True")
                # self.parameter = self.optimizer.SGD(self.parameter, self.grad, learning_rate)

                pIterLoss.append(loss)
                avgLoss += loss
                if iter % display == 0:
                    print("Train Accuracy {} with prob : {}".format((countacc/float(display)), output[pred]))
                    print("Train Loss: ", loss)
                    countacc = 0.0
                    loss, acc = self.Test(testImages, testLabelsHotEncoding)
            trainAcc = ( float(count) / float(numberOfImages) )
            print("##################Overall Accuracy & Loss Calculation")
            print("TrainAccuracy: ", trainAcc)
            print("TrainLoss: ", ( float(avgLoss)/float(numberOfImages)))
            avgtestloss, avgtestacc = self.Test(testImages, testLabelsHotEncoding)
            totaloss = float(avgLoss) / float(numberOfImages)
            pEpochTrainLoss.append(totaloss)
            pEpochTrainAccuracy.append(trainAcc)
            pEpochTestLoss.append(avgtestloss)
            pEpochTestAccuracy.append(avgtestacc)

            x_axis = np.linspace(0, epoch, len(pEpochTrainLoss), endpoint=True)
            plt.semilogy(x_axis, pEpochTrainLoss)
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.draw()
            file = open("Weightparameter1.pkl", "wb")
            file.write(pickle.dumps(self.parameter))
            file.close()
            fill2 = open("parameter.pkl", "wb")
            fill2.write(pickle.dumps((pEpochTrainAccuracy, pEpochTrainLoss, pEpochTestAccuracy, pEpochTestLoss)))
            fill2.close()


    def start_training(self,datafolder, batchsize, learning_rate, number_of_epoch):
        pass











