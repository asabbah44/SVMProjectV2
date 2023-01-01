import glob
import os

import cv2
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder

base_dir_train = '../data/train/'
category_train = ['airplane', 'bird', 'cat', 'frog', 'horse', 'ship']
def main():
    x_train, y_train = get_data_images(base_dir_train, category_train)

    X_train, X_validate, Y_train, Y_validate = train_test_split(x_train, y_train, test_size=0.10,shuffle=True)

    le = LabelEncoder()
    Y_validate=le.fit_transform(Y_validate)
    Create_validate=True
    if Create_validate:
        with open("Validate.txt", "w") as f:

            for i in range(len(X_validate)):

                f.write(str(Y_validate[i]) + " ")

                for j, val in enumerate(X_validate[i]):
                    f.write(str(j + 1) + ":" + str(val) + " ")
                f.write("\n")


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


if __name__ == "__main__":
    main()

# cv2.imshow('Original image', image)
# cv2.imshow('HSV image', hsvImage)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Load the dataset and split it into training and test sets
# X, y = load_dataset()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# # Convert the data to the format required by LibSVM
# X_train_svm = svmutil.svm_read_problem(X_train, y_train)
# X_test_svm = svmutil.svm_read_problem(X_test, y_test)
#
# # Set the SVM parameters
# param = svmutil.svm_parameter('-t 0 -c 1')
#
# # Train the SVM model
# model = svmutil.svm_train(X_train_svm, param)
#
# # Make predictions on the test data
# _, y_pred, _ = svmutil.svm_predict(y_test, X_test_svm, model)
#
# # Calculate the accuracy of the model
# accuracy = sum(y_pred == y_test) / len(y_test)
# print("Accuracy:", accuracy)
