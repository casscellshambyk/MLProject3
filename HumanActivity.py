import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from time import time
from sklearn.preprocessing import StandardScaler
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.metrics import homogeneity_score, completeness_score, \
v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
# %matplotlib inline

data_dir = '/content/drive/My Drive/ML_apps/'
train_dataset = pd.read_csv(data_dir + 'HA_train.csv')

Y_data = train_dataset['activity']
x_data = train_dataset.drop(['rn', 'activity'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x_data, Y_data, test_size=0.33, random_state=42)

print(train_dataset.sample(5))
# sb.pairplot(train_dataset.drop(['rn'], axis=1), hue='activity')
# plt.ioff()

ks = range(1, 10)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(x_train)
    inertias.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.style.use('bmh')
plt.plot(ks, inertias, '-o')
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(ks)
plt.show()

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

pca = PCA(random_state=123)
pca.fit(x_train)

features = range(pca.n_components_)


plt.figure(figsize=(8,4))
plt.bar(features[:15], pca.explained_variance_[:15], color='lightskyblue')
plt.xlabel('PCA feature')
plt.ylabel('Variance')
plt.xticks(features[:15])
plt.show()

pca = PCA(n_components=1, random_state=123)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.fit_transform(x_test)

print("Fitting model...")
k_model = KMeans(n_clusters=2, random_state=123, n_init=30)
start_time = time()
k_model.fit(x_train_pca)
end_time = time()
print("Model fitting took %f seconds" % (end_time - start_time))

print("Starting predictions...")
start_time = time()
Y_pred = k_model.predict(pca.fit_transform(x_test_pca))
end_time = time()
print("Predictions took %f seconds" % (end_time - start_time))

df = pd.DataFrame({'cluster_labels': Y_pred, 'actual_labels': y_test.to_list()})

ct = pd.crosstab(df['cluster_labels'], df['actual_labels'])
display(ct)

print('% 9s' % 'inertia  homo    compl   v-meas   ARI     AMI     silhouette')
print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
  %(k_model.inertia_,
  homogeneity_score(y_test, Y_pred),
  completeness_score(y_test, Y_pred),
  v_measure_score(y_test, Y_pred),
  adjusted_rand_score(y_test, Y_pred),
  adjusted_mutual_info_score(y_test, Y_pred),
  silhouette_score(x_test_pca, Y_pred, metric='euclidean')))
