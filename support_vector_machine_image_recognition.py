import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

numberImages = datasets.load_digits()

supportVectorClassifier = svm.SVC(gamma = 0.0001,C=100)

x = numberImages.data[:-5]
y = numberImages.target[:5]
supportVectorClassifier.fit(x,y) 

predictedImage = numberImages.data[-4]

print("Assume the image is: ",supportVectorClassifier.predict(predictedImage))

plt.imshow(numberImages.imahes[-4], cmap = plt.cm.gray_r, interpolation = "nearest")

plt.show()