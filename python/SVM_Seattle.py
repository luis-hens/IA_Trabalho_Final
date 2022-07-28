### Importando as bibliotecas ###

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import sklearn.model_selection as ms
from sklearn.preprocessing import StandardScaler

### Leitura dos dados com pandas e remoção da coluna date ###

seattle = pd.read_csv('seattle-weather.csv')
seattle = seattle.drop('date', axis=1)

### Separação dos dados e conversão de weather para numéricos ###

seattle['weather'] = seattle['weather'].map(
    {'drizzle': 0, 'fog': 1, 'rain': 2, 'snow': 3, 'sun': 4})
X = seattle.iloc[:, :-1].values
Y = seattle.iloc[:, -1].values

### Separando as variáveis com 80% dos dados para treinamento e 20% para teste ###

XTrain, XTest, YTrain, YTest = ms.train_test_split(X, Y, test_size = 1/5, random_state = 0)

### Normalizando os dados para evitar que outliers gerem previsões erradas ###

XTrainScaler = StandardScaler()
XTestScaler = StandardScaler()

XTrain = XTrainScaler.fit_transform(XTrain)
XTest = XTestScaler.fit_transform(XTest)

### Aplicando SVM com as funções kernel: rbf, poly e linear ###

SeattleClassify_rbf = SVC(kernel='rbf')
SeattleClassify_rbf.fit(XTrain, YTrain)

SeattleClassify_poly = SVC(kernel='poly')
SeattleClassify_poly.fit(XTrain, YTrain)

SeattleClassify_linear = SVC(kernel='linear')
SeattleClassify_linear.fit(XTrain, YTrain)

YPredict_rbf = SeattleClassify_rbf.predict(XTest)
YPredict_poly = SeattleClassify_poly.predict(XTest)
YPredict_linear = SeattleClassify_linear.predict(XTest)

YResult_rbf = np.concatenate((YPredict_rbf.reshape(len(YPredict_rbf),1), YTest.reshape(len(YTest),1)),1)
YResult_poly = np.concatenate((YPredict_poly.reshape(len(YPredict_poly),1), YTest.reshape(len(YTest),1)),1)
YResult_linear = np.concatenate((YPredict_linear.reshape(len(YPredict_linear),1), YTest.reshape(len(YTest),1)),1)

### Imprimindo os resultados ###

print('Teste do SVM para kernel rbf')
cm = confusion_matrix(YTest, YPredict_rbf)
print(cm)
print(accuracy_score(YTest, YPredict_rbf))

print('\nTeste do SVM para kernel poly')
cm = confusion_matrix(YTest, YPredict_poly)
print(cm)
print(accuracy_score(YTest, YPredict_poly))

print('\nTeste do SVM para kernel linear')
cm = confusion_matrix(YTest, YPredict_linear)
print(cm)
print(accuracy_score(YTest, YPredict_linear))