import pickle

import numpy as np
from sklearn.model_selection import cross_validate

from kernel import HOG, KernelMultiClassSVC, PolynomialKernel, load_raw_data
from kernel.utils import aggregate, preds_to_csv, vote

X, y, Xte = load_raw_data('./data')

hog = HOG(pixels_per_cell=(4, 4), cells_per_block=(7, 7))
hog.set_logger('debug')

X = hog.transform(X)
Xte = hog.transform(Xte)
k = PolynomialKernel(degree=5)
model = KernelMultiClassSVC(C=1, kernel=k)

res = cross_validate(model, X, y, cv=3, verbose=10, scoring='accuracy',
                     n_jobs=-1, return_estimator=True)

print('Validation scores', res['test_score'])

# with open('res.pkl', 'wb') as f:
#     pickle.dump(res, f)
models = res['estimator']
best = np.argmax(res['test_score'])
del res

preds = aggregate(models, Xte)
# preds = vote(model, Xte, best)
preds_to_csv(preds, 'result.csv')
