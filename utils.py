
from sklearn import svm, linear_model, ensemble
import csp
import numpy as np
import scipy as sp
def classification(x, y, clf='lsvm'):
    if clf == 'lsvm': model = svm.SVC(kernel='linear')
    return ''



if __name__ == '__main__':
    raw = np.load('data/comp_iva/epo.npz', allow_pickle=True)
    data = raw['data']
    x = data[0]['x']
    y = data[0]['y']


    x0 = []
    x1 = []
    x2 = []
    x3 = []
    for i in range(len(y)):
        if y[i] == 0:
            x0.append(x[i])
        elif y[i] == 1:
            x1.append(x[i])
        elif y[i] == 2:
            x2.append(x[i])
        elif y[i] == 3:
            x3.append(x[i])


    


    abc = csp.CSP([x0, x1, x2, x3])
    pass


