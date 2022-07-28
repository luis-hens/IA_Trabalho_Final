### Importando as bibliotecas ###

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import sklearn.model_selection as ms
from sklearn.preprocessing import StandardScaler

### Leitura dos dados com pandas ###

college = pd.read_csv('go-to-college.csv')

### Convertendo Strings para Inteiros no dataset ###

college['type_school'] = college['type_school'].map({'Academic': 0, 'Vocational': 1})
college['school_accreditation'] = college['school_accreditation'].map({'A': 0, 'B': 1})
college['gender'] = college['gender'].map({'Male': 0, 'Female': 1})
college['interest'] = college['interest'].map({'Not Interested': 0, 'Less Interested': 1, 'Interested': 2, 'Uncertain': 3, 'Very Interested': 4})
college['residence'] = college['residence'].map({'Urban': 0, 'Rural': 1})
college['parent_was_in_college'] = college['parent_was_in_college'].map({False: 0, True: 1})
college['will_go_to_college'] = college['will_go_to_college'].map({False: 0, True: 1})

### Separando as colunas usadas na classificação do dado de entrada na universidade ###

X = college.iloc[:, :-1].values
Y = college.iloc[:, -1].values

### Separando as variáveis com 80% dos dados para treinamento e 20% para teste ###

XTrain, XTest, YTrain, YTest = ms.train_test_split(X, Y, test_size = 1/5, random_state = 0)

### Normalizando os dados para evitar que outliers gerem previsões erradas ###

XTrainScaler = StandardScaler()
XTestScaler = StandardScaler()

XTrain = XTrainScaler.fit_transform(XTrain)
XTest = XTestScaler.fit_transform(XTest)

### Aplicando SVM com as funções kernel: rbf, poly e linear

CollegeClassify_rbf = SVC(kernel='rbf')
CollegeClassify_rbf.fit(XTrain, YTrain)

CollegeClassify_poly = SVC(kernel='poly')
CollegeClassify_poly.fit(XTrain, YTrain)

CollegeClassify_linear = SVC(kernel='linear')
CollegeClassify_linear.fit(XTrain, YTrain)

YPredict_rbf = CollegeClassify_rbf.predict(XTest)
YPredict_poly = CollegeClassify_poly.predict(XTest)
YPredict_linear = CollegeClassify_linear.predict(XTest)

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