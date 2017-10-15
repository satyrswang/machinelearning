#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn import datasets,cross_validation
from sklearn import svm
# import quandl


digits = datasets.load_digits()
#print(type(digits),digits.data,digits.target)


clf = svm.SVC(gamma=0.001, C=100) #two para
X,y = digits.data[:-10], digits.target[:-10] 

clf.fit(X,y)
print(clf.predict(digits.data[-5]))

plt.imshow(digits.images[-5], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

clf2 = svm.SVC(gamma=0.01, C=100)
clf3 = svm.SVC(gamma=0.0001, C=100)
clf4 = svm.SVR(kernel = 'poly')

# X = np.array(df.drop(['label'],1)) #丢掉label列
# y = np.array(df['label'])

# X= preprocessing.scale(X) #

# X_t,X_test,y_t,y_test  =cross_validation.train_test_split(X,y,test_size = 0.2)
# clf = LinearRegression(n_jobs = -1)
# accuracy = clf.score(X_test,y_test)














