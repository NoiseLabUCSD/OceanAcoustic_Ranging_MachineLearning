
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Training Data
X_train = np.loadtxt( './data/DataSet01/SBCEx16_training_input.txt')
Y_train = np.loadtxt( './data/DataSet01/training_labels.txt')
Range_train = np.loadtxt( './data/DataSet01/Mapping_range_labels.txt')

# Test Data
X_test = np.loadtxt( './data/DataSet01/SBCEx16_test_input.txt')
Y_test = np.loadtxt( './data/DataSet01/test_Ranges.txt')

C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X_train, Y_train)
print('svc linear done!')
rbf_svc = svm.SVC(kernel='rbf', gamma=1/149.0, C=C).fit(X_train, Y_train)
print('rbf_svc done!')
poly_svc = svm.SVC(kernel='poly', gamma=1/149.0,degree=1, C=C).fit(X_train, Y_train)
print('poly_svc done!')
lin_svc = svm.LinearSVC(C=C).fit(X_train, Y_train)
print('lin_svc done!')

# title for the plots
titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel (gama=1/149)',
          'SVC with polynomial (degree 1) kernel']

fig=plt.figure(figsize=(5.0,4.0))
for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(X_test)
    print(Z)
    Z_out = Range_train[Z.astype(np.int32)]

    plt.plot(Z_out,"o",markersize=2,markeredgewidth=0.5,markeredgecolor='b',markerfacecolor='none')
    plt.plot(Y_test,'r',linewidth=1.0)

    plt.xlabel('Time [index]',fontsize=8)
    plt.ylabel('Range (m)',fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tick_params(axis='both', which='minor', labelsize=8)
    plt.title(titles[i],fontsize=8)


plt.show()
fig.savefig('Fig_DataSet01.jpg', dpi=300)
