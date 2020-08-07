import numpy as np
import pandas as pd
import seaborn as sb

sb.set_style("dark")
import matplotlib.pyplot as plt
from time import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# %pylab inline

data_dir = '/content/drive/My Drive/ML_apps/'
dataset = pd.read_csv(data_dir + 'MNIST_train.csv')
classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']  # used for display

labels = dataset['label'].values.astype('int32')  # labels
dataset = dataset.drop(['label'], axis=1).values.astype('float32') / 255


def display_train_examples(num_samples, data, labels):
    for y, column in enumerate(classes):
        indicies = np.nonzero([i == y for i in labels])
        indicies = np.random.choice(indicies[0], num_samples, replace=False)
        for i, idx in enumerate(indicies):
            plt_index = i * len(classes) + y + 1
            plt.subplot(num_samples, len(classes), plt_index)
            plt.imshow(data[idx].reshape((28, 28)))
            plt.axis("off")
            if i == 0:
                plt.title(column)

    plt.show()
    return None


x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.33, random_state=42)

knn_model = KNeighborsClassifier(n_neighbors=1)  # setting k=1 based on what was previously seen on this dataset
rf_model = RandomForestClassifier()  # default Random Forest Classifier from sklearn

print("Fitting kNN Model...")
start_time = time.time()
knn_model.fit(x_train, y_train)
end_time = time.time()
print("Time to fit kNN: %f" % (end_time - start_time))

print("Fitting Random Forest Model...")
start_time = time.time()
rf_model.fit(x_train, y_train)
end_time = time.time()
print("Time to Random Forest: %f" % (end_time - start_time))

print("kNN Predicting test dataset...")
start_time = time.time()
knn_pred = knn_model.predict(x_test)
end_time = time.time()
knn_total_time = end_time - start_time
print("kNN took %f seconds" % knn_total_time)

print("Random Forest Prediction test dataset")
start_time = time.time()
rf_pred = rf_model.predict(x_test)
end_time = time.time()
rf_total_time = end_time - start_time
print("Random Forest took %f seconds" % rf_total_time)

print("Generating Confusion Matricies")
knn_cm = confusion_matrix(y_test, knn_pred)
rf_cm = confusion_matrix(y_test, rf_pred)

print("kNN Confution Matrix")
sb.heatmap(knn_cm, annot=True)
plt.show()
print("kNN Classification Report")
print(classification_report(y_test, knn_pred, target_names=classes))

print("Random Forest Confustion Matrix")
sb.heatmap(rf_cm, annot=True)
plt.show()
print("Random Forest Classification Report")
print(classification_report(y_test, rf_pred, target_names=classes))
