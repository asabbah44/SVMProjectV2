import os
from sklearn.preprocessing import StandardScaler
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
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
    print(x_train[1])

    #for normaliztion btween -1 and 1
    x_train=StandardScaler().fit_transform(x_train)
    x_test = StandardScaler().fit_transform(x_test)


    print(x_test.shape)
    print(y_test.shape)
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)

    # for development and validation
    # (x_train, x_test, y_train, y_test) = train_test_split(x_test, y_test, test_size=0.25, random_state=42)

    # To generate the data set with Libsvm format
    festuerExtract = False
    if festuerExtract:
        with open("D:/data/ntrain.txt", "w") as f:

            for i in range(len(x_train)):

                f.write(str(y_train[i]))
                for j, val in enumerate(x_train[i]):
                    f.write(" " + str(j + 1) + ":" + str(val))
                f.write("\n")

        with open("D:/data/ntest.txt", "w") as f:

            for i in range(len(x_test)):

                f.write(str(y_test[i]))

                for j, val in enumerate(x_test[i]):
                   f.write(" " + str(j + 1) + ":" + str(val))
                f.write("\n")

 # Extract data from imagees
def get_data_images(base_dir, category):
    X = []
    Y = []
    His=[]
    for sub in category:
        files = glob.glob(os.path.join(base_dir, sub, '*.jpg'))
        # get label from folder name
        label = sub

        for file in files:
            His=[]
            img = cv2.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h1 = cv2.calcHist([img], [0], None, [16], [0, 256]).ravel()
            h2 = cv2.calcHist([img], [1], None, [16], [0, 256]).ravel()
            h3 = cv2.calcHist([img], [2], None, [16], [0, 256]).ravel()
            His.extend(h1)
            His.extend(h2)
            His.extend(h3)

            #print(His)

            X.append(His)

            Y.append(label)

    X = np.array(X)

    Y = np.array(Y)

    return X, Y


if __name__ == "__main__":
    main()
