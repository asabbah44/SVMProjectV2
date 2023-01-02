# import necessary packages
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os
import glob
import cv2
import numpy as np

from sklearn.model_selection import GridSearchCV

# For server
base_dir_train = '../data/train/'
category_train =['airplane', 'bird', 'cat', 'frog', 'horse', 'ship']

base_dir_test = '../data/test/'
category_test = ['airplane', 'bird', 'cat', 'frog', 'horse', 'ship']


# base_dir_train = 'Data/train/'
# category_train =['airplane', 'bird', 'cat', 'frog', 'horse', 'ship']
#
# base_dir_test = 'Data/test/'
# category_test = ['airplane', 'bird', 'cat', 'frog', 'horse', 'ship']

def praperData(base_dir, category):
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

# def praperData(base_dir, category):
#     X = []
#     Y = []
#     for sub in category:
#         files = glob.glob(os.path.join(base_dir, sub, '*.jpg'))
#         # get label from folder name
#         label = sub
#         # Loop over the files
#         for file in files:
#             img = cv2.imread(file)
#             # Set size of image
#             img = cv2.resize(img, (32, 32))
#             # Convert image to HSV
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#             # Convert the image  to  vector
#             x = img.flatten()
#
#             X.append(x)
#             Y.append(label)
#
#     X = np.array(X)
#     Y = np.array(Y)
#     return X, Y


# encode the labels as integer


dataTrain, lablesTrain = praperData(base_dir_train, category_train)

dataTest, lablesTest = praperData(base_dir_test, category_test)

le = LabelEncoder()
lablesTrain = le.fit_transform(lablesTrain)

print(lablesTrain.shape)

le = LabelEncoder()
lablesTest = le.fit_transform(lablesTest)

print(dataTrain.shape)
print(lablesTest.shape)
(trainX, testX, trainY, testY) = dataTrain,dataTest, lablesTrain, lablesTest

get_best_parameters = False
if get_best_parameters:
    grid_params = {'n_neighbors': [5, 7, 9, 11, 13, 15, 17, 19, 21,23,25],
                   'weights': ['uniform', 'distance'],
                   'metric': ['minkowski', 'euclidean', 'manhattan']}

    gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose=1, cv=3, n_jobs=-1)
    g_res = gs.fit(trainX, trainY)
    print("Best score ", g_res.best_score_)

    print("Best hyperparameters  ", g_res.best_params_)


startTrain= True
if startTrain:
    # model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    model = KNeighborsClassifier(n_neighbors=21, weights='uniform', metric='manhattan')
    model.fit(trainX, np.ravel(trainY))
    y_hat = model.predict(trainX)
    y_knn = model.predict(testX)
    print('Training set accuracy: ', metrics.accuracy_score(trainY, y_hat))
    print('Test set accuracy: ', metrics.accuracy_score(testY, y_knn))

    print(confusion_matrix(testY, y_knn))

    print(classification_report(testY, y_knn, target_names=le.classes_))
    # print(classification_report(testY, model.predict(testX), target_names=le.classes_))