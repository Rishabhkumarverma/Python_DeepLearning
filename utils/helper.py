import numpy as np


class Optimizer:

    def __init__(self):
        self.Vdw = dict()
        self.Sdw = dict()
        self.Vdb = dict()
        self.Sdb = dict()
        self.Vdwc = dict()
        self.Vdbc = dict()
        self.Sdbc = dict()
        self.Sdwc = dict()

    def initADAM(self, numberofweight, numberofbias):

        for l in range(numberofweight):
            self.Vdw["dw" + str(l + 1)] = 0
            self.Sdw["dw" + str(l + 1)] = 0
            self.Vdb["db" + str(l + 1)] = 0
            self.Sdb["db" + str(l + 1)] = 0

            self.Vdwc["dw" + str(l + 1)] = 0
            self.Vdbc["db" + str(l + 1)] = 0
            self.Sdbc["db" + str(l + 1)] = 0
            self.Sdwc["dw" + str(l + 1)] = 0

    def d_abs(self, x):
        mask = (x >= 0) * 1.0
        mask2 = (x < 0) * -1.0
        return mask + mask2

    def l2_regularization(self, parameter, grad, lambda_v):
        L = len(parameter) // 2
        total_sum = 0
        for l in range(L):
            grad["dw" + str(l + 1)] = grad["dw" + str(l + 1)] + 2*lambda_v*parameter["w" + str(l + 1)]
            total_sum += np.abs(np.square(parameter["w" + str(l + 1)])).sum()
        reg_loss = 0.5 * lambda_v * total_sum
        return grad, reg_loss




    def SGD(self, parameter, grads, learning_rate):
        L = len(parameter) // 2
        # update rule for each parameter
        for l in range(L):
            parameter["w" + str(l + 1)] = parameter["w" + str(l + 1)] - learning_rate * \
                                          grads["dw" + str(l + 1)]
            parameter["b" + str(l + 1)] = parameter["b" + str(l + 1)] - learning_rate * \
                                          grads["db" + str(l + 1)]
        return parameter

    def ADAM(self, parameter, grads, learning_rate, t):
        beta1 = 0.9
        beta2 = 0.999
        eps_stable = 1e-8
        L = len(parameter) // 2
        for l in range(L):
            dw = grads["dw" + str(l + 1)]
            db = grads["db" + str(l + 1)]

            # Update Vdw and Vdb like momentum.

            self.Vdw["dw" + str(l + 1)] = np.add(np.multiply(beta1, self.Vdw["dw" + str(l + 1)]),
                                                 np.multiply((1.0 - beta1), dw))
            self.Vdb["db" + str(l + 1)] = np.add(np.multiply(beta1, self.Vdb["db" + str(l + 1)]),
                                                 np.multiply((1.0 - beta1), db))
            # self.Vdw = beta1 * self.Vdw + (1 - beta1) * dw
            # self.Vdb = beta1 * self.Vdb + (1 - beta1) * db

            # Update Sdw and Sdb like Rmsprop

            self.Sdw["dw" + str(l + 1)] = np.add(np.multiply(beta2, self.Sdw["dw" + str(l + 1)]),
                                                 np.multiply((1.0 - beta2), np.square(dw)))
            self.Sdb["db" + str(l + 1)] = np.add(np.multiply(beta2, self.Sdb["db" + str(l + 1)]),
                                                 np.multiply((1.0 - beta2), np.square(db)))

            # self.Sdw = beta2 * self.Sdw + (1 - beta2) * np.square(dw)
            # self.Sdb = beta2 * self.Sdb + (1 - beta2) * np.square(db)
            # In Adam optimization implementation, we do implement bias correction.
            self.Vdwc["dw" + str(l + 1)] = np.divide(self.Vdw["dw" + str(l + 1)], (1.0 - np.power(beta1, t)) + eps_stable)
            self.Vdbc["db" + str(l + 1)] = np.divide(self.Vdb["db" + str(l + 1)], (1.0 - np.power(beta1, t)) + eps_stable)

            # self.Vdwc = self.Vdw / (1 - beta1 ** t)
            # self.Vdbc = self.Vdb / (1 - beta1 ** t)

            self.Sdwc["dw" + str(l + 1)] = np.divide(self.Sdw["dw" + str(l + 1)], (1.0 - np.power(beta2, t)) + eps_stable)
            self.Sdbc["db" + str(l + 1)] = np.divide(self.Sdb["db" + str(l + 1)], (1.0 - np.power(beta2, t)) + eps_stable)

            # self.Sdwc = self.Sdw / (1 - beta2 ** t)
            # self.Sdbc = self.Sdb / (1 - beta2 ** t)
            # Update parameters W and b
            parameter["w" + str(l + 1)] = np.subtract(parameter["w" + str(l + 1)], np.divide(np.multiply(learning_rate,
                                                                                                         self.Vdwc[
                                                                                                             "dw" + str(
                                                                                                                 l + 1)]),
                                                                                             np.sqrt(np.add(self.Sdwc[
                                                                                                                "dw" + str(
                                                                                                                    l + 1)],
                                                                                                            eps_stable))))

            parameter["b" + str(l + 1)] = np.subtract(parameter["b" + str(l + 1)], np.divide(np.multiply(learning_rate,
                                                                                                         self.Vdbc[
                                                                                                             "db" + str(
                                                                                                                 l + 1)]),
                                                                                             np.sqrt(np.add(self.Sdbc[
                                                                                                                "db" + str(
                                                                                                                    l + 1)],
                                                                                                            eps_stable))))

            # parameter["w" + str(l + 1)] = parameter["w" + str(l + 1)] - ( learning_rate * (
            #             self.Vdwc) ) /  np.sqrt(self.Sdwc + eps_stable)
            # parameter["b" + str(l + 1)] = parameter["b" + str(l + 1)] - ( learning_rate * (
            #         self.Vdbc ) ) / np.sqrt(self.Sdbc + eps_stable)
        return parameter
