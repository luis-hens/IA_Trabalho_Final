### Importando as bibliotecas ###

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score

### Leitura dos dados com pandas e remoção da coluna date ###

seattle = pd.read_csv('seattle-weather.csv')
seattle = seattle.drop('date', axis=1)

### Separação dos dados e conversão de weather para numéricos ###

seattle['weather'] = seattle['weather'].map({'drizzle': 0, 'fog': 1, 'rain': 2, 'snow': 3, 'sun': 4})
X = seattle.iloc[:, :-1].values
Y = seattle.iloc[:, -1].values

### Aplicação do K-Means

SKMeans = KMeans(n_clusters=5)
YPredict = SKMeans.fit_predict(X)
YResult = np.concatenate((YPredict.reshape(len(YPredict), 1), Y.reshape(len(Y), 1)), 1)

### Imprimindo os resultados ###

print('Teste do K-Means')
cm = confusion_matrix(Y, YPredict)
print(cm)
print(accuracy_score(Y, YPredict))