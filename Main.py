from sklearn.metrics import f1_score, confusion_matrix, classification_report
from libsvm.python.libsvm import svm, svmutil,commonutil

base_dir_train = '../data/train/'
category_train =['airplane', 'bird', 'cat', 'frog', 'horse', 'ship']

base_dir_test = '../data/test/'
category_test = ['airplane', 'bird', 'cat', 'frog', 'horse', 'ship']

# libsvm constants
LINEAR = 0
POLYNOMIAL = 1
RBF = 2
CHI_SQAURED = 5


def get_param(kernel_type=LINEAR):
    param = svm.svm_parameter("-q")
    param.probability = 1
    if kernel_type == LINEAR:
        param.kernel_type = LINEAR
        param.C = 1

    elif kernel_type == POLYNOMIAL:
        param.kernel_type = POLYNOMIAL
        param.C = .01
        param.gamma = .00000001
        param.degree = 3

    elif kernel_type == RBF:
        param.kernel_type = RBF
        param.C = .01
        param.gamma = .00000001
    else:
        param.kernel_type = CHI_SQAURED

    return param


def classifier(kernelType=LINEAR):

    if kernelType == LINEAR:
        param = get_param(0)
    elif kernelType == POLYNOMIAL:
        param = get_param(1)
    elif kernelType == RBF:
        param = get_param(2)
    else:
        param = get_param(5)

    y_train, x_train = svmutil.svm_read_problem("D:/data/htrain.txt")
    y_test, x_test = svmutil.svm_read_problem("D:/data/htest.txt")
    print(x_test[1])

    #prob = svm.svm_problem(y_train, x_train)
    print("problem read")
    model = svmutil.svm_train(y_train,x_train, ' -t 1 -c 0.5 -g 0.0078125 ')
    print("traind")
    y_hat, p_acc, p_val = svmutil.svm_predict(y_test, x_test, model, "-q")


    print("Accuracy:", p_acc[0])
    f1 = f1_score(y_test, y_hat,average='micro')

    print("F1 score:", f1)
    print(classification_report(y_test, y_hat, target_names=category_test))
    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_hat)

    print("Confusion matrix:")
    print(cm)


if __name__ == "__main__":
  classifier(LINEAR)

