from utils.helper import Optimizer
from utils.layer import FullyConnected, Softmax, SoftMaxLoss, Relu, Dropout, BatchNorm1
from utils.dataloader import Dataloader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
from tf_utils.Network import run_trainingepoch
import json
import pickle


class Net:
    def __init__(self):
        self.parameter = dict()
        self.grad = dict()
        self.gradInput = dict()
        self.gamma = dict()
        self.beta = dict()

        self.Fc1 = FullyConnected()
        self.Fc2 = FullyConnected()
        self.Fc3 = FullyConnected()
        self.Fc4 = FullyConnected()
        self.Fc5 = FullyConnected()

        self.bn1 = BatchNorm1()
        self.bn2 = BatchNorm1()
        self.bn3 = BatchNorm1()
        self.bn4 = BatchNorm1()

        self.Relu1 = Relu()
        self.Relu2 = Relu()
        self.Relu3 = Relu()
        self.Relu4 = Relu()

        self.dropout1 = Dropout()
        self.dropout2 = Dropout()
        self.softmaxloss = SoftMaxLoss()
        self.optimizer = Optimizer()
        # self.mean = 96.69698605419005
        # self.std_v = 51.426420731308994
        self.mean = 128.0
        self.std_v = 128.0

    def initWeigth(self, saveParameter=None):

        # self.parameter['w1'] = np.zeros(shape=(28*28, 32), dtype=np.float32)
        # self.parameter['w2'] = np.zeros(shape=(32, 10), dtype=np.float32)
        # self.parameter['b1'] = np.zeros(shape=(32), dtype=np.float32)
        # self.parameter['b2'] = np.zeros(shape=(10), dtype=np.float32)

        # self.parameter['w1'] = np.random.randn(108 * 108, 32) * np.sqrt(1/(108*108+32))
        # self.parameter['w2'] = np.random.randn(32, 10) * np.sqrt(1/(32+10))
        # self.parameter['w3'] = np.random.randn(10, 6) * np.sqrt(1/(10+6))
        # self.parameter['b1'] = np.zeros(32)
        # self.parameter['b2'] = np.zeros(10)
        # self.parameter['b3'] = np.zeros(6)

        # self.parameter['w1'] = np.random.uniform(low=0.5, high=15, size = (108 * 108, 32))
        # self.parameter['w2'] = np.random.uniform(low=0.3, high=15, size = (32, 10))
        # self.parameter['w3'] = np.random.uniform(low=0.1, high=2, size = (10, 6))
        # self.parameter['b1'] = np.random.uniform(low=0, high=1, size = (32) )
        # self.parameter['b2'] = np.random.uniform(low=0., high=1, size = (10) )
        # self.parameter['b3'] = np.random.uniform(low=0., high=1, size = (6) )
        fc1 = 256
        fc2 = 128
        fc3 = 32
        fc4 = 16
        label = 6

        if saveParameter is None:
            # self.parameter['w1'] = np.random.uniform(low=-1, high=1, size=(108 * 108, fc1)) * np.sqrt(6. / (108 * 108 + fc1))
            # self.parameter['w2'] = np.random.uniform(low=-1, high=1, size=(fc1, fc2)) * np.sqrt(6. / (fc1 + fc2))
            # self.parameter['w3'] = np.random.uniform(low=-1, high=1, size=(fc2, label)) * np.sqrt(6. / (fc2 + label))
            self.parameter['w1'] = np.random.randn(108 * 108, fc1) * np.sqrt(6. / (108 * 108 + fc1))
            self.parameter['w2'] = np.random.randn(fc1, fc2) * np.sqrt(6. / (fc1 + fc2))
            self.parameter['w3'] = np.random.randn(fc2, fc3) * np.sqrt(6. / (fc2 + fc3))
            self.parameter['w4'] = np.random.randn(fc3, fc4) * np.sqrt(6. / (fc3 + fc4))
            self.parameter['w5'] = np.random.randn(fc4, label) * np.sqrt(6. / (fc4 + label))
            # self.parameter['b1'] = np.random.randn(fc1)
            # self.parameter['b2'] = np.random.randn(fc2)
            # self.parameter['b3'] = np.random.randn(label)
            self.parameter['b1'] = np.zeros(fc1)
            self.parameter['b2'] = np.zeros(fc2)
            self.parameter['b3'] = np.zeros(fc3)
            self.parameter['b4'] = np.zeros(fc4)
            self.parameter['b5'] = np.zeros(label)
            self.gamma["bn1"] = 0.9
            self.beta["bn1"] = 0.1
            self.gamma["bn2"] = 0.9
            self.beta["bn2"] = 0.1
            self.gamma["bn3"] = 0.9
            self.beta["bn3"] = 0.1
            self.gamma["bn4"] = 0.9
            self.beta["bn4"] = 0.1

        else:
            parameter = pickle.load(open(saveParameter, "rb"))
            self.parameter = parameter

    def ForwardPass(self, image, dropout_param1, dropout_param2):
        mode = dropout_param2["mode"]

        fc1 = self.Fc1.forward(input=image, weight=self.parameter["w1"],
                               bias=self.parameter["b1"])  # in = 1x1024, out = 1x32
        bn1 = self.bn1.forward(fc1, self.gamma["bn1"], self.beta["bn1"], mode=mode)
        # drp1 = self.dropout1.forward(bn1, dropout_param1)
        drp1 = bn1
        relu1 = self.Relu1.forward(drp1)
        fc2 = self.Fc2.forward(input=relu1, weight=self.parameter["w2"],
                               bias=self.parameter["b2"])  # in = 1x32 , out = 1x10
        bn2 = self.bn2.forward(fc2, self.gamma["bn2"], self.beta["bn2"], mode=mode)

        # drp2 = self.dropout2.forward(bn2, dropout_param2)
        drp2 = bn2
        relu2 = self.Relu2.forward(drp2)
        fc3 = self.Fc3.forward(input=relu2, weight=self.parameter["w3"],
                               bias=self.parameter["b3"])  # in = 1x32 , out = 1x10
        bn3 = self.bn3.forward(fc3, self.gamma["bn3"], self.beta["bn3"], mode=mode)
        relu3 = self.Relu3.forward(bn3)
        fc4 = self.Fc4.forward(input=relu3, weight=self.parameter["w4"],
                               bias=self.parameter["b4"])
        bn4 = self.bn4.forward(fc4, self.gamma["bn4"], self.beta["bn4"], mode=mode)
        relu4 = self.Relu4.forward(bn4)
        fc5 = self.Fc5.forward(input=relu4, weight=self.parameter["w5"],
                               bias=self.parameter["b5"])
        # relu3, _ = self.Relu3.forward(fc3)
        out = Softmax.forward(fc5)
        return out

    def ForwardPassold(self, image, dropout_param1, dropout_param2):

        fc1 = self.Fc1.forward(input=image, weight=self.parameter["w1"],
                               bias=self.parameter["b1"])  # in = 1x1024, out = 1x32
        drp1 = self.dropout1.forward(fc1, dropout_param1)
        relu1 = self.Relu1.forward(drp1)
        fc2 = self.Fc2.forward(input=relu1, weight=self.parameter["w2"],
                               bias=self.parameter["b2"])  # in = 1x32 , out = 1x10
        drp2 = self.dropout2.forward(fc2, dropout_param2)
        relu2 = self.Relu2.forward(drp2)
        fc3 = self.Fc3.forward(input=relu2, weight=self.parameter["w3"],
                               bias=self.parameter["b3"])  # in = 1x32 , out = 1x10
        relu3 = self.Relu3.forward(fc3)
        fc4 = self.Fc4.forward(input=relu3, weight=self.parameter["w4"],
                               bias=self.parameter["b4"])
        relu4 = self.Relu4.forward(fc4)
        fc5 = self.Fc5.forward(input=relu4, weight=self.parameter["w5"],
                               bias=self.parameter["b5"])
        # relu3, _ = self.Relu3.forward(fc3)
        out = Softmax.forward(fc5)
        return out

    def ForwardPass2(self, image, dropout_param1, dropout_param2):

        fc1 = self.Fc1.forward(input=image, weight=self.parameter["w1"],
                               bias=self.parameter["b1"])  # in = 1x1024, out = 1x32
        relu1 = self.Relu1.forward(fc1)
        drp1 = self.dropout1.forward(relu1, dropout_param1)

        fc2 = self.Fc2.forward(input=drp1, weight=self.parameter["w2"],
                               bias=self.parameter["b2"])  # in = 1x32 , out = 1x10
        relu2 = self.Relu2.forward(fc2)
        drp2 = self.dropout2.forward(relu2, dropout_param2)
        fc3 = self.Fc3.forward(input=drp2, weight=self.parameter["w3"],
                               bias=self.parameter["b3"])  # in = 1x32 , out = 1x10

        relu3 = self.Relu3.forward(fc3)
        fc4 = self.Fc4.forward(input=relu3, weight=self.parameter["w4"],
                               bias=self.parameter["b4"])
        relu4 = self.Relu4.forward(fc4)
        fc5 = self.Fc5.forward(input=relu4, weight=self.parameter["w5"],
                               bias=self.parameter["b5"])
        # relu3, _ = self.Relu3.forward(fc3)
        out = Softmax.forward(fc5)
        return out

    def BackwordPass(self, prediction):
        dout = self.softmaxloss.backward(prediction)
        # dout = self.Relu3.backward(dout)
        delta5, dw5, db5 = self.Fc5.backward(dout=dout)
        relu_delta4 = self.Relu4.backward(delta5)
        bn_delta4, self.gamma["bn4"], self.beta["bn4"] = self.bn4.backward(relu_delta4)

        delta4, dw4, db4 = self.Fc4.backward(bn_delta4)

        relu_delta3 = self.Relu3.backward(delta4)
        bn_delta3, self.gamma["bn3"], self.beta["bn3"] = self.bn3.backward(relu_delta3)
        delta3, dw3, db3 = self.Fc3.backward(bn_delta3)

        relu_delta2 = self.Relu2.backward(delta3)
        bn_delta2, self.gamma["bn2"], self.beta["bn2"] = self.bn2.backward(relu_delta2)

        # drp1_delta = self.dropout2.backward(bn_delta2)
        drp1_delta = bn_delta2
        delta2, dw2, db2 = self.Fc2.backward(dout=drp1_delta)

        relu_delta1 = self.Relu1.backward(delta2)
        bn_delta1, self.gamma["bn1"], self.beta["bn1"] = self.bn1.backward(relu_delta1)
        # drp2_delta = self.dropout1.backward(relu_delta1)
        drp2_delta = bn_delta1
        delta1, dw1, db1 = self.Fc1.backward(drp2_delta)
        self.grad['dw1'] = dw1
        self.grad['dw2'] = dw2
        self.grad['dw3'] = dw3
        self.grad['dw4'] = dw4
        self.grad['dw5'] = dw5
        self.grad['db1'] = db1
        self.grad['db2'] = db2
        self.grad['db3'] = db3
        self.grad['db4'] = db4
        self.grad['db5'] = db5

    def BackwordPassold(self, prediction):
        dout = self.softmaxloss.backward(prediction)
        # dout = self.Relu3.backward(dout)
        delta5, dw5, db5 = self.Fc5.backward(dout=dout)
        relu_delta4 = self.Relu4.backward(delta5)
        delta4, dw4, db4 = self.Fc4.backward(relu_delta4)
        relu_delta3 = self.Relu3.backward(delta4)
        delta3, dw3, db3 = self.Fc3.backward(relu_delta3)
        relu_delta2 = self.Relu2.backward(delta3)
        drp1_delta = self.dropout2.backward(relu_delta2)
        delta2, dw2, db2 = self.Fc2.backward(dout=drp1_delta)
        relu_delta1 = self.Relu1.backward(delta2)
        drp2_delta = self.dropout1.backward(relu_delta1)
        delta1, dw1, db1 = self.Fc1.backward(drp2_delta)
        self.grad['dw1'] = dw1
        self.grad['dw2'] = dw2
        self.grad['dw3'] = dw3
        self.grad['dw4'] = dw4
        self.grad['dw5'] = dw5
        self.grad['db1'] = db1
        self.grad['db2'] = db2
        self.grad['db3'] = db3
        self.grad['db4'] = db4
        self.grad['db5'] = db5

    def BackwordPass2(self, prediction):
        dout = self.softmaxloss.backward(prediction)
        # dout = self.Relu3.backward(dout)
        delta5, dw5, db5 = self.Fc5.backward(dout=dout)
        relu_delta4 = self.Relu4.backward(delta5)
        delta4, dw4, db4 = self.Fc4.backward(relu_delta4)
        relu_delta3 = self.Relu3.backward(delta4)
        delta3, dw3, db3 = self.Fc3.backward(relu_delta3)
        drp1_delta = self.dropout2.backward(delta3)

        relu_delta2 = self.Relu2.backward(drp1_delta)
        delta2, dw2, db2 = self.Fc2.backward(dout=relu_delta2)

        drp2_delta = self.dropout1.backward(delta2)
        relu_delta1 = self.Relu1.backward(drp2_delta)
        delta1, dw1, db1 = self.Fc1.backward(relu_delta1)
        self.grad['dw1'] = dw1
        self.grad['dw2'] = dw2
        self.grad['dw3'] = dw3
        self.grad['dw4'] = dw4
        self.grad['dw5'] = dw5
        self.grad['db1'] = db1
        self.grad['db2'] = db2
        self.grad['db3'] = db3
        self.grad['db4'] = db4
        self.grad['db5'] = db5

    def Train(self, image, label):
        dropout_param1 = dict()
        dropout_param1["mode"] = "train"
        dropout_param1["p"] = 0.95
        dropout_param2 = dict()
        dropout_param2["mode"] = "train"
        dropout_param2["p"] = 0.90

        data = np.array(image, dtype=np.float32)
        # data = data.flatten()
        # input_data = np.reshape(data, (1, data.size))
        # label = np.reshape(label, (1, label.size))
        output = self.ForwardPass(data, dropout_param1, dropout_param2)
        # print(output)
        loss = self.softmaxloss.forward(output, label)

        # print(loss)
        # print(loss)
        self.BackwordPass(output)
        return loss, output

    def Test(self, testImages, testLabelsHotEncoding):
        dropout_param1 = dict()
        dropout_param1["mode"] = "test"
        dropout_param1["p"] = 0.95
        dropout_param2 = dict()
        dropout_param2["mode"] = "test"
        dropout_param2["p"] = 0.90
        numberOfImages = testImages.shape[0]
        # numberOfImages = 10
        avgloss = 0
        avgacc = 0
        count = 0
        assignmentOut = dict()
        for iter in range(numberOfImages):
            image = testImages[iter, :, :]
            labels = testLabelsHotEncoding[iter, :]
            data = np.array(image, dtype=np.float32)
            # data = np.divide(data, 255)
            data = np.subtract(data, self.mean)
            data = np.divide(data, self.std_v)
            data = data.flatten()
            input_data = np.reshape(data, (1, data.size))
            labels = np.reshape(labels, (1, labels.size))
            output = self.ForwardPass(input_data, dropout_param1, dropout_param2)
            pred = np.argmax(output[0])
            gt = np.argmax(labels[0])
            loss = self.softmaxloss.forward(output, labels)
            if pred == gt:
                count += 1
                # print("True")
            avgloss += loss
            imagestrID = str(iter)
            assignmentOut[imagestrID] = gt
        avgacc = float(count) / float(numberOfImages)
        avgloss = float(avgloss) / float(numberOfImages)
        print("Test Accuracy: ", avgacc)
        print("Test Loss: ", avgloss)
        return avgloss, avgacc

    def doTest(self, testImages):
        dropout_param1 = dict()
        dropout_param1["mode"] = "test"
        dropout_param1["p"] = 0.90
        dropout_param2 = dict()
        dropout_param2["mode"] = "test"
        dropout_param2["p"] = 0.95
        numberOfImages = testImages.shape[0]

        assignmentOut = dict()
        assignmentOut["id"] = "label"
        for iter in range(numberOfImages):
            image = testImages[iter, :, :]
            data = np.array(image, dtype=np.float32)
            # data = np.divide(data, 255)
            data = np.subtract(data, self.mean)
            data = np.divide(data, self.std_v)
            data = data.flatten()
            input_data = np.reshape(data, (1, data.size))

            output = self.ForwardPass(input_data, dropout_param1, dropout_param2)
            pred = np.argmax(output[0])

            imagestrID = str(iter)
            assignmentOut[imagestrID] = pred
        return assignmentOut

    def start_training_mnist(self, data_folder, batch_size, learning_rate, NumberOfEpoch, display=1000):
        self.initWeigth()
        self.optimizer.initADAM(5, 5)
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
        print("Training started")
        t = 0
        for epoch in range(NumberOfEpoch):
            avgLoss = 0
            trainAcc = 0.0
            count = 0.0
            countacc = 0.0
            pIterLoss = []
            print("##############EPOCH : {}##################".format(epoch))
            for iter in range(numberOfImages):
                t += 1
                image = trainingImages[iter, :, :]
                labels = trainLabelsHotEncoding[iter, :]
                loss, output = self.Train(image, labels)
                output = output[0]
                pred = np.argmax(output)
                gt = np.argmax(labels)
                if pred == gt:
                    count += 1.0
                    countacc += 1.0
                    # print("True")
                # self.parameter = self.optimizer.SGD(self.parameter, self.grad, learning_rate)

                pIterLoss.append(loss)
                avgLoss += loss
                if iter % display == 0:
                    print("Train Accuracy {} with prob : {}".format((countacc / float(display)), output[pred]))
                    print("Train Loss: ", loss)
                    countacc = 0.0
                    loss, acc = self.Test(testImages, testLabelsHotEncoding)

                self.parameter = self.optimizer.ADAM(self.parameter, self.grad, learning_rate, t)
                self.parameter = self.optimizer.l2_regularization(self.parameter, 0.001)
            trainAcc = (float(count) / float(numberOfImages))
            print("##################Overall Accuracy & Loss Calculation")
            print("TrainAccuracy: ", trainAcc)
            print("TrainLoss: ", (float(avgLoss) / float(numberOfImages)))
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
            file = open("Assignment_test_2.pkl", "wb")
            file.write(pickle.dumps(self.parameter))
            file.close()
            fill2 = open("Assignment_parameter.pkl", "wb")
            fill2.write(pickle.dumps((pEpochTrainAccuracy, pEpochTrainLoss, pEpochTestAccuracy, pEpochTestLoss)))
            fill2.close()

    def start_training(self, datafolder, batchsize, learning_rate, number_of_epoch, display=1000):
        # self.initWeigth(saveParameter="Assignment_weight_3.pkl")
        self.initWeigth()
        self.optimizer.initADAM(5, 5)
        data = Dataloader.loaddata(datafolder["data"])
        testData = Dataloader.loaddata(datafolder["test"])
        labels = Dataloader.loaddata(datafolder["label"])
        data = np.array(data)
        train, val, temptrainlabel, tempvallabel = train_test_split(data, labels, test_size=0.2)
        # train = data[0:25000, :, :]
        # val = data[25001:29160, :, :]
        # temptrainlabel = labels[0:25000]
        # tempvallabel = labels[25001:29160]
        trainlabel = Dataloader.toHotEncoding(temptrainlabel)
        vallabel = Dataloader.toHotEncoding(tempvallabel)

        t = 0
        # numberOfImages = 10
        pEpochTrainLoss = []
        pEpochTrainAccuracy = []
        pEpochTestLoss = []
        pEpochTestAccuracy = []
        for epoch in range(number_of_epoch):
            train, temptrainlabel = sklearn.utils.shuffle(train, temptrainlabel, random_state=1)
            trainlabel = Dataloader.toHotEncoding(temptrainlabel)

            # if epoch > 20:
            #     learning_rate = 0.0001
            if epoch > 70:
                learning_rate = 0.00001
            if epoch > 130:
                learning_rate = 0.000001
            if epoch > 175:
                learning_rate = 0.0000001

            avgLoss = 0
            trainAcc = 0.0
            count = 0.0
            countacc = 0.0
            pIterLoss = []
            total_train_image = train.shape[0]
            iter = 0
            countiter = 0.0
            countitertemp = 0.0
            loss_iter = 0.0
            # for iter in range(total_train_image - batchsize):
            t += 1
            while iter < (total_train_image - batchsize):

                randomSelect = iter
                # randomSelect = np.random.randint(0 ,(total_train_image - batchsize))
                image = train[randomSelect:randomSelect + batchsize, :, :]
                labels = trainlabel[randomSelect:randomSelect + batchsize, :]
                image = np.array(image, dtype=np.float32)
                image = np.subtract(image, self.mean)
                image = np.divide(image, self.std_v)
                input_data = np.reshape(image, (batchsize, 108 * 108))
                input_data, labels = sklearn.utils.shuffle(input_data, labels, random_state=1)
                # label = np.reshape(label, (1, label.size))
                loss, outputs = self.Train(input_data, labels)
                # self.parameter = self.optimizer.SGD(self.parameter, self.grad, learning_rate)
                self.grad, reg_loss = self.optimizer.l2_regularization(self.parameter, self.grad, 0)
                loss += reg_loss
                for outiter in range(batchsize):
                    # output = output[0]
                    pred = np.argmax(outputs[outiter, :])
                    gt = np.argmax(labels[outiter, :])
                    if pred == gt:
                        count += 1.0
                        countacc += 1.0
                    countiter += 1.0
                    countitertemp += 1.0
                    # print("True")
                    # self.parameter = self.optimizer.SGD(self.parameter, self.grad, learning_rate)

                pIterLoss.append(loss)
                avgLoss += loss
                if iter % display == 0:
                    print("Preiction: ", outputs[0, :])
                    print(
                        "Train Accuracy {} with prob : {}".format((countacc / float(countitertemp)), outputs[0, pred]))
                    print("Train Loss: ", loss)
                    countacc = 0.0
                    countitertemp = 0.0
                    loss, acc = self.Test(val, vallabel)
                    # if acc > 0.55:
                    #     assignmentOut = self.doTest(testData)
                    #     fileName = "result_" + str(acc) + "_.csv"
                    #     with open(fileName, 'w') as f:
                    #         for key in assignmentOut.keys():
                    #
                    # f.write("%s,%s\n" % (key, assignmentOut[key]))
                self.parameter = self.optimizer.ADAM(self.parameter, self.grad, learning_rate, t)
                iter += batchsize
                loss_iter += 1.0

            trainAcc = (float(count) / float(countiter))
            print("##################Overall Accuracy & Loss Calculation")
            print(iter, ":TrainAccuracy: ", trainAcc)
            print(iter, ":TrainLoss: ", (float(avgLoss) / float(loss_iter)))
            avgtestloss, avgtestacc = self.Test(val, vallabel)
            totaloss = float(avgLoss) / float(total_train_image)
            pEpochTrainLoss.append(totaloss)
            pEpochTrainAccuracy.append(trainAcc)
            pEpochTestLoss.append(avgtestloss)
            pEpochTestAccuracy.append(avgtestacc)
            # fileName = "Assignment_weight_" + str(trainAcc) + "_" + str(avgtestacc) + ".pkl"
            file = open("Assignment_weight_4.pkl", "wb")
            file.write(pickle.dumps(self.parameter))
            file.close()
            fill2 = open("Assignment_parameter.pkl", "wb")
            fill2.write(pickle.dumps((pEpochTrainAccuracy, pEpochTrainLoss, pEpochTestAccuracy, pEpochTestLoss)))
            fill2.close()
            print("############################################")
            if avgtestacc > 0.55:
                assignmentOut = self.doTest(testData)
                fileName = "result_ov_" + str(avgtestacc) + "_.csv"
                with open(fileName, 'w') as f:
                    for key in assignmentOut.keys():
                        f.write("%s,%s\n" % (key, assignmentOut[key]))

    def start_Gpu_training(self, datafolder, batchsize, learning_rate, number_of_epoch, display=1000):
        data = Dataloader.loaddata(datafolder["data"])
        testData = Dataloader.loaddata(datafolder["test"])
        labels = Dataloader.loaddata(datafolder["label"])
        data = np.array(data)
        train, val, temptrainlabel, tempvallabel = train_test_split(data, labels, test_size=0.3)

        # train = data[0:25000, :, :]
        # val = data[25001:29160, :, :]
        # temptrainlabel = labels[0:25000]
        # tempvallabel = labels[25001:29160]
        trainlabel = Dataloader.toHotEncoding(temptrainlabel)
        vallabel = Dataloader.toHotEncoding(tempvallabel)
        run_trainingepoch(number_of_epoch, train, trainlabel, val, vallabel, testData, batchsize, train.shape[0])
