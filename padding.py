import numpy as np

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return np.pad(x[:max_len], (149,0), 'constant', constant_values=(0,0))
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    padded_x = np.pad(padded_x, (149,0), 'constant', constant_values=(0,0))
    return padded_x	


def pad_(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    padded_x = np.pad(x, (0, max_len-x_len), 'constant', constant_values=(0, 0))
    return padded_x