import numpy as np

def xavier_init(fan_in, fan_out):
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

def he_init(fan_in, fan_out):
    std = np.sqrt(2 / fan_in)
    return np.random.randn(fan_in, fan_out) * std
