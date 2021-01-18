import numpy as np




def batchnorm_forward(x, gamma, beta, bn_param):

    mode = bn_param["mode"]
    eps = 1e-5
    momentum = 0.9

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None

    if mode == 'train':

        # TODO: 实现训练过程中batch normalization层的前向传播
        # ******请在此段内填写代码(开始)******#

        x_mean = np.mean(x, axis=0)
        x_sigma = np.var(x, axis=0)

        x_hat = (x - x_mean) / np.sqrt(x_sigma + eps)
        out = gamma * x_hat + beta

        cache = (x, x_hat, gamma, x_mean, x_sigma, eps)

        running_mean = momentum * running_mean + (1 - momentum) * x_mean
        running_var = momentum * running_var + (1 - momentum) * x_sigma

        # ******请在此段内填写代码(结束)******#

    elif mode == 'test':

        # TODO: 实现测试过程中batch normalization层的前向传播
        # ******请在此段内填写代码(开始)******#

        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta

        # ******请在此段内填写代码(结束)******#
    else:
        raise ValueError('ValueError: Batch Normalization mode "%s", only accept for "train" or "test". ' % mode)

    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var


    return out, cache



def batchnorm_backward(dout, cache):

    dx, dgamma, dbeta = None, None, None
    # TODO: 实现batch normalization层的反向传播，将结果保存在dx, dgamma, dbeta三个变量中

    # ******请在此段内填写代码(开始)******#

    x, x_hat, gamma, x_mean, x_sigma, eps = cache

    N, D = dout.shape

    dgamma = np.sum(x_hat * dout, axis=0)
    dbeta = np.sum(dout, axis=0)

    dx_hat = dout * gamma

    dvar = np.sum(dx_hat * (x - x_mean) * (-0.5) * np.power(x_sigma + eps, -1.5), axis=0)
    dmean = np.sum(dx_hat * -1 / np.sqrt(x_sigma + eps), axis=0) + dvar * np.mean(-2 * (x - x_mean), axis=0)
    dx = 1 / np.sqrt(x_sigma + eps) * dx_hat + dvar * 2.0 / N * (x - x_mean) + 1.0 / N * dmean

    # ******请在此段内填写代码(结束)******#


    return dx, dgamma, dbeta