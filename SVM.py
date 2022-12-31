import os

from libsvm.python.libsvm import svm, svmutil
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report
# from python.libsvm import svm, svmutil
import glob

base_dir_train = '../data/train/'
category_train = ['airplane', 'bird', 'cat', 'frog', 'horse', 'ship']

base_dir_test = '../data/test/'
category_test = ['airplane', 'bird', 'cat', 'frog', 'horse', 'ship']

# libsvm constants
LINEAR = 0
POLYNOMIAL = 1
RBF = 2
CHI_SQAURED = 5
le = LabelEncoder()


def main():
    x_train, y_train = get_data_images(base_dir_train, category_train)
    x_test, y_test = get_data_images(base_dir_test, category_test)

    y_train = le.fit_transform(y_train)
    # features selection
    y_test = le.fit_transform(y_test)

    # for development and validation
    # (x_train, x_test, y_train, y_test) = train_test_split(x_test, y_test, test_size=0.25, random_state=42)
    with open("train.txt", "w") as f:

        for i in range(len(x_train)):

            f.write(str(y_train[i]) + " ")

            for j, val in enumerate(x_train[i]):
                f.write(str(j + 1) + ":" + str(val) + " ")
            f.write("\n")

    with open("test.txt", "w") as f:

        for i in range(len(x_test)):

            f.write(str(y_test[i]) + " ")

            for j, val in enumerate(x_test[i]):
                f.write(str(j + 1) + ":" + str(val) + " ")
            f.write("\n")
    classifier(5)


# Linear :  python grid.py -log2c -1,2,1 -log2g 1,1,1 -t 0 D:\SVMProject\train.txt


def get_data_images(base_dir, category):
    X = []
    Y = []

    for sub in category:
        files = glob.glob(os.path.join(base_dir, sub, '*.jpg'))
        # get label from folder name
        label = sub

        for file in files:
            img = cv2.imread(file)
            # Set size of image
            img = cv2.resize(img, (32, 32))
            # Convert image to HSV
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Convert the image  to  vector
            x = img.flatten()

            X.append(x)
            Y.append(label)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def get_param(kernel_type=LINEAR):
    param = svm.svm_parameter("-q")
    param.probability = 1
    if kernel_type == LINEAR:
        param.kernel_type = LINEAR
        param.C = .01


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

    y_train, x_train = svmutil.svm_read_problem("train.txt")
    y_test, x_test = svmutil.svm_read_problem("test.txt")
    prob = svm.svm_problem(y_train, x_train)
    model = svmutil.svm_train(prob, param)

    y_hat, p_acc, p_val = svmutil.svm_predict(y_test, x_test, model, "-q -b 1")

    print("Accuracy:", p_acc[0])
    f1 = f1_score(y_test, y_hat, average='micro')

    print("F1 score:", f1)
    print(classification_report(y_test, y_hat, target_names=le.classes_))
    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_hat)

    print("Confusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()
