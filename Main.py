import os
import cv2
import matplotlib.pyplot as plt
from python.libsvm import svmutil

image = cv2.imread('train/cat/0000.jpg')
hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
for i, col in enumerate(['b', 'g', 'r']):
    hist = cv2.calcHist([hsvImage], [i], None, [32], [0, 32])
    plt.plot(hist, color=col)
    plt.xlim([0, 32])

plt.show()
print(hist.shape)



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
