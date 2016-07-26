import numpy as np

def RMSLE(targets,predictions):
    err = np.log(np.array(targets) + 1) - np.log(np.array(predictions) + 1)
    return np.sqrt(sum(err ** 2) / len(err))
