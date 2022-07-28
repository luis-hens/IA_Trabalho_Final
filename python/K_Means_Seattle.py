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

for i in range(30):
    SKMeans = KMeans(n_clusters=5, random_state=i)
    YPredict = SKMeans.fit_predict(X)
    YResult = np.concatenate((YPredict.reshape(len(YPredict),1), Y.reshape(len(Y),1)),1)
    SKMeans.fit(X)
    seattle['k-means'] = SKMeans.labels_

    ### Imprimindo tabela e os resultados (Note que o resultado pode estar incorreto dado que nem sempre o K-Means
    # Acertará o  número correto do cluster). A matriz de confusão foi omitida para melhor legibilidade dos dados ###

    print('Teste do K-Means {}'.format(i+1))
    cm = confusion_matrix(Y, YPredict)
    # print(cm)
    print(accuracy_score(Y, YPredict))
    
print("\nDataset")
print(seattle)