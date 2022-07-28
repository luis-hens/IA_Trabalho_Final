### Importando as bibliotecas ###

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score

### Leitura dos dados com pandas ###

college = pd.read_csv('go-to-college.csv')

### Convertendo Strings para Inteiros no dataset ###

college['type_school'] = college['type_school'].map({'Academic': 0, 'Vocational': 1})
college['school_accreditation'] = college['school_accreditation'].map({'A': 0, 'B': 1})
college['gender'] = college['gender'].map({'Male': 0, 'Female': 1})
college['interest'] = college['interest'].map({'Not Interested': 0, 'Less Interested': 1, 'Interested': 2, 'Uncertain': 3, 'Very Interested': 4})
college['residence'] = college['residence'].map({'Urban': 0, 'Rural': 1})
college['parent_was_in_college'] = college['parent_was_in_college'].map({False: 0, True: 1})
college['will_go_to_college'] = college['will_go_to_college'].map({False: 1, True: 0})

### Separando as colunas usadas na classificação do dado de entrada na universidade ###

X = college.iloc[:, :-1].values
Y = college.iloc[:, -1].values

### Aplicação do K-Means

for i in range(30):
    SKMeans = KMeans(n_clusters=2, random_state=i)
    YPredict = SKMeans.fit_predict(X)
    YResult = np.concatenate((YPredict.reshape(len(YPredict),1), Y.reshape(len(Y),1)),1)
    SKMeans.fit(X)
    college['k-means'] = SKMeans.labels_

    ### Imprimindo tabela e os resultados (Note que o resultado pode estar incorreto dado que nem sempre o K-Means
    # Acertará o  número correto do cluster). A matriz de confusão foi omitida para melhor legibilidade dos dados ###

    print('Teste do K-Means {}'.format(i+1))
    cm = confusion_matrix(Y, YPredict)
    # print(cm)
    print(accuracy_score(Y, YPredict))
    
print("\nDataset")
print(college)
