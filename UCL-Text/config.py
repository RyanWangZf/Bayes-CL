# -*- coding: utf-8 -*-

class DefaultConfig(object):
    # model name
    model = "small"

    # data name
    data_name = "R8"

    # training log output dir
    result_dir = "./out"

    # random seed
    seed = 2020

    # execute on cuda devices or not
    use_gpu = True

    # hyperparameters
    batch_size = 32
    num_epoch = 50
    lr = 1e-3
    weight_decay = 0.0
    
    # if >0, then label noise is imposed on training data
    noise_ratio = 0.0

    # log name
    print_opt = "DEF"

    # mode for bayesian cl: snr, mix, epis, alea
    bnn = "snr"

    # num of baby steps , in [2, 5]
    baby_step = 2

    # mode for computing uncertainty v.s. acc, in ["ucl", "ucltl"]
    ucl = "ucl"

    # curriculum learning for self-paced learning
    spl = "spl"

    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)

        print('*************************************************')
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print("{} => {}".format(k, getattr(self, k)))

        print('*************************************************')


opt = DefaultConfig()
