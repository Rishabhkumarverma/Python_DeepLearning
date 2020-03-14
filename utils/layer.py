import numpy as np
from utils.im2col import *

class FullyConnected:

    def __init__(self):
        self.cache = None

    def forward(self, input, weight, bias):

        """
        :param input: Input Tensor (N,D_i), N : miniBatch , D_i: dimension of input tensor
        :param weight: Weight of the Layer (D_i,D_o)
        :param bias: Bias (D_o)
        :return:
        """
        # input = np.array(input)
        # weight = np.array(weight)
        # bias = np.array(bias)
        N = input.shape[0]
        reshaped_input = input.reshape(N, -1)
        output = np.dot(reshaped_input, weight) + bias.T
        cache = (input, weight, bias)
        self.cache = cache
        return output

    def backward(self, dout, cache=None):
        """
        :param dout:
        :param cache:
        :return:
        """
        input, weight, bias = None, None, None
        dx, dw, db = None, None, None
        bias = None
        if self.cache is not None:
            input, weight, bias = self.cache

        elif cache is not None:
            input, weight, bias = cache
        else:
            print("required (input, weight, bias) cache data not found!")
            exit(-1)
        # input = np.array(input)
        # weight = np.array(weight)
        # bias = np.array(bias)

        N = input.shape[0]
        dx = np.dot(dout, weight.T)
        dx = dx.reshape(input.shape)

        reshaped_input = input.reshape(N, -1)
        dw = reshaped_input.T.dot(dout)  # transpose the input vector then dot product with dout

        db = np.sum(dout, axis=0)

        return dx, dw, db


class Relu:

    def __init__(self):
        self.cache = None
        self.r_range = -0.1

    def forward(self, input):
        """
        :param input:
        :return:
        """
        output = np.maximum(self.r_range , input)
        cache = input
        self.cache = cache
        return output

    def backward(self, dout, cache=None):
        dx, input = None, None
        if self.cache is not None:
            input = self.cache

        elif cache is not None:
            input = cache
        else:
            print("required (input) cache data not found!")
            exit(-1)

        grad_relu = input >= self.r_range

        dx = dout * grad_relu

        return dx


class Softmax:

    @staticmethod
    def forward(input):
        # print (input)
        exps = np.exp(input - np.max(input, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
        # # e_x = np.exp(x - np.max(x))
        # output =  ( np.exp(input) / np.sum(np.exp(input), axis=1) )  # only difference
        # # output = np.multiply(output, [-1])
        # return output


    @staticmethod
    def backward():
        pass


class SoftMaxLoss:

    def __init__(self):
        self.cache = None
        self.N = 1
        self.labels = None

    def forward(self, input, labels, epsilon=1e-12):

        predictions = np.clip(input, epsilon, 1. - epsilon)
        self.N = input.shape[0]
        loss = - ( np.sum(labels * np.log(predictions + 1e-9)) / self.N )
        # loss = - np.mean(labels * np.log(input), axis=1,keepdims=True)
        self.labels = labels
        return loss

    def backward(self, data):

        dout = data.copy()
        # dout /= self.N
        # for b in range(self.N):
        #     scale_factor = 1 / float(np.count_nonzero(self.labels[b,:]))
        #     for c in range(len(self.labels[b,:])):  # For each class
        #         if self.labels[b, c] != 0.0:  # If positive class
        #             dout[b, c] = scale_factor * (dout[b, c] - 1) + (1 - scale_factor) * dout[
        #                 b, c]  # Gradient for classes with positive labels considering scale factor
        dout = data - self.labels
        dout = dout / self.N
        return dout

class Dropout :
    def __init__(self):
        self.dropoutparam = None
        self.mask = None
        self.N = 1

    def forward(self, input, droput_param):
        p, mode = droput_param['p'], droput_param['mode']
        if 'seed' in droput_param:
            np.random.seed(droput_param['seed'])

        # inititlaztion of the output  and mask
        mask = None
        out = None
        self.N = input.shape[0]
        if mode == 'train':
            data = np.random.rand(*input.shape)
            mask = (data < p) / p
            out = input*mask
        elif mode == "test":
            mask = None
            out = input

        self.dropoutparam = droput_param
        self.mask = mask
        out = out.astype(input.dtype, copy=False)
        return out
    def backward(self, dout):

        mode = self.dropoutparam['mode']
        mask = self.mask
        dx = None

        if mode == 'train':
            dx = dout * mask
        elif mode == 'test':
            dx = dout
        return dx

# vanilla implementation
class BatchNorm :

    def __init__(self):
        self.cache = None
        pass

    def forward(self, input, gamma, beta, eps=1e-12):

        N, C, H,  W = input.shape
        D = (C, H, W)
        # step1: calculate mean
        mu = (1. / N ) * np.sum(input, axis=0)

        # step2: subtract mean vector of every trainings example
        xmu = input - mu

        # step3: following the lower branch - calculation denominator
        sq = xmu ** 2

        # step4: calculate variance
        var = (1. / N) * np.sum(sq, axis=0)

        # step5: add eps for numerical stability, then sqrt
        sqrtvar = np.sqrt(var + eps)

        # step6: invert sqrtwar
        ivar = 1. / sqrtvar

        # step7: execute normalization
        xhat = xmu * ivar

        # step8: Nor the two transformation steps
        gammax = gamma * xhat

        # step9
        out = gammax + beta

        # store intermediate
        self.cache = (xhat, gamma, xmu, ivar, sqrtvar, var, eps)
        return out

    def backward(self, dout):
        # unfold the variables stored in cache
        xhat, gamma, xmu, ivar, sqrtvar, var, eps = self.cache
        # get the dimensions of the input/output
        N, C, H, W = dout.shape
        D = C*H*W

        # step9
        dbeta = np.sum(dout, axis=0)
        dgammax = dout  # not necessary, but more understandable

        # step8
        dgamma = np.sum(dgammax * xhat, axis=0)
        dxhat = dgammax * gamma

        # step7
        divar = np.sum(dxhat * xmu, axis=0)
        dxmu1 = dxhat * ivar

        # step6
        dsqrtvar = -1. / (sqrtvar ** 2) * divar

        # step5
        dvar = 0.5 * 1. / np.sqrt(var + eps) * dsqrtvar

        # step4
        dsq = 1. / N * np.ones((C, H, W)) * dvar
        # dsq = 1. / N * np.ones((N, D)) * dvar

        # step3
        dxmu2 = 2 * xmu * dsq

        # step2
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)

        # step1
        dx2 = 1. / N * np.ones((C,H,W)) * dmu
        # dx2 = 1. / N * np.ones((N, D)) * dmu

        # step0
        dx = dx1 + dx2

        return dx, dgamma, dbeta


class BatchNorm1 :

    def __init__(self):
        self.cache = None
        self.bn2_mean = 0
        self.bn2_var = 0
        pass

    def forward(self, input, gamma, beta, eps=1e-12, mode="train"):

        length = len(input.shape)
        if length == 2:
            N, D = input.shape
        else:
            N, C, H, W = input.shape

        if mode == "train":
            mu = np.mean(input, axis=0)
            var = np.var(input, axis=0)
            self.bn2_mean = 0.9 * self.bn2_mean + 0.1* mu
            self.bn2_var = 0.9 * self.bn2_var + 0.1 * var

        if mode == "test":
            mu = self.bn2_mean
            var = self.bn2_var


        X_norm = (input - mu) / np.sqrt(var + 1e-8)
        out = gamma * X_norm + beta
        # store intermediate
        self.cache = (input, X_norm, mu, var, gamma, beta)
        return out

    def backward(self, dout):
        # unfold the variables stored in cache
        X, X_norm, mu, var, gamma, beta = self.cache
        # get the dimensions of the input/output
        length = len(dout.shape)
        if length == 2:
            N, D = dout.shape
        else:
            N, C, H, W = dout.shape
            D = C*H*W

        X_mu = X - mu
        std_inv = 1. / np.sqrt(var + 1e-8)

        dX_norm = dout * gamma
        dvar = np.sum(dX_norm * X_mu, axis=0) * -.5 * std_inv ** 3
        dmu = np.sum(dX_norm * -std_inv, axis=0) + dvar * np.mean(-2. * X_mu, axis=0)

        dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)
        dgamma = np.sum(dout * X_norm, axis=0)
        dbeta = np.sum(dout, axis=0)

        return dX, dgamma, dbeta

"""

conv layer will accept an input in X: DxCxHxW dimension, input filter W: NFxCxHFxHW, and bias b: Fx1, where:

    D is the number of input
    C is the number of image channel
    H is the height of image
    W is the width of the image
    NF is the number of filter in the filter map W
    HF is the height of the filter, and finally
    HW is the width of the filter.


"""
class Conv :
    def __init__(self):
        self.cache = None

    def forward(self, X, W, b, stride=1, padding=1):

        n_filters, d_filter, h_filter, w_filter = W.shape
        n_x, d_x, h_x, w_x = X.shape
        h_out = (h_x - h_filter + 2 * padding) / stride + 1
        w_out = (w_x - w_filter + 2 * padding) / stride + 1

        # if not h_out.is_integer() or not w_out.is_integer():
            # raise Exception('Invalid output dimension!')

        h_out, w_out = int(h_out), int(w_out)

        X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
        W_col = W.reshape(n_filters, -1)
        # tout = np.dot(W_col, X_col)
        # out = tout + b
        out = W_col @ X_col + b
        out = out.reshape(n_filters, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)

        self.cache = (X, W, b, stride, padding, X_col)

        return out

    def backward(self, dout):
        X, W, b, stride, padding, X_col = self.cache
        n_filter, d_filter, h_filter, w_filter = W.shape

        db = np.sum(dout, axis=(0, 2, 3))
        db = db.reshape(n_filter, -1)

        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)
        dW = dout_reshaped @ X_col.T
        dW = dW.reshape(W.shape)

        W_reshape = W.reshape(n_filter, -1)
        dX_col = W_reshape.T @ dout_reshaped
        dX = col2im_indices(dX_col, X.shape, h_filter, w_filter, padding=padding, stride=stride)

        return dX, dW, db














