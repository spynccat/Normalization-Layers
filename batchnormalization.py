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

        pass

        # ******请在此段内填写代码(结束)******#

    elif mode == 'test':

        # TODO: 实现测试过程中batch normalization层的前向传播
        # ******请在此段内填写代码(开始)******#

        pass

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

    pass

    # ******请在此段内填写代码(结束)******#


    return dx, dgamma, dbeta