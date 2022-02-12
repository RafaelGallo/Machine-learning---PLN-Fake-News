# Bibliotecas do python

import pandas as pd # Biblioteca para carregar dataset
import numpy as np # Manipulação de alegbra linear

# Bibliotecas para visualização de dados
import matplotlib.pyplot as plt
import seaborn as sns

# Biblotecas tirar alertar de mensagens
import warnings 
warnings.filterwarnings("ignore")

# Bibliotecas NLTK
import re
import nltk
import re 
import html 
import string

nltk.download('punkt')
nltk.download("stopwords")
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Carregando base de dados
# Base de dados
data_1 = pd.read_csv("data\Constraint_Train.csv")

# Exibindo os 5 primeiras linhas com o comando head()
data_1.head()

# Exibindo os 5 últimos linhas com o comando tail()
data_1.tail()

# Exibindo quantidades de linhas e colunas 
data_1.shape

# Exibindo os tipos de dados
data_1.dtypes

# Total de colunas e linhas - data_test
print("Números de linhas: {}" .format(data_1.shape[0]))
print("Números de colunas: {}" .format(data_1.shape[1]))

# Exibindo valores ausentes e valores únicos
print("\nMissing values :  ", data_1.isnull().sum().values.sum())
print("\nUnique values :  \n",data_1.nunique())

# stopwords e pontuação 
pont = string.punctuation
pont = stopwords.words("english")
print(pont)

# Label encoder
dummyTrain = pd.get_dummies(data_1["Previsão"]) 
print(dummyTrain)

# Defininfo dummy para label fake 0 real 1
data_1["Previsão"] = dummyTrain["real"]
data_1.head() 

# Defenindo base de treino e teste train e test

train = data_1["Texto"]
test = data_1["Previsão"]

word_lemmatizer = WordNetLemmatizer()

# Lemmatization dos dados
def Lemmatization(inst):
    pal = []
    for x in inst.split():
        pal.append(word_lemmatizer.lemmatize(x))
    return (" ".join(pal))

# Preprocessing base de dados
def Preprocessing(inst):
    inst = re.sub(r"http\S+", "", inst).lower().replace('.','').replace(';','').replace('-','').replace(':','').replace(')','').replace('"','')
    stopwords = set(nltk.corpus.stopwords.words('english'))
    pal = [i for i in inst.split() if not i in stopwords]
    return (" ".join(pal))

# Negações dos textos
def neg(text):
    neg = ["não", "not"]
    neg_dect = False
    result = []
    pal = text.split()

    for x in pal:
        x = x.lower()
        if neg_dect == True:
            x = x + "_NEG"
        if x in neg:
            neg_dect = True
        result.append(x)
    return ("".join(result))

# stopwords dos textos seperando
def stopwords(inst):
    stopwords = set(nltk.corpus.stopwords.words("english"))
    pal = [i for i in inst.split() if not i in stopwords]
    return (" ".join(pal))

# stem - stemmer
def stem(inst):
    stem = nltk.stem.RSLPStemmer()
    pal = []
    for x in inst.split():
        pal.append(stemmer.stem(x))
    return (" ".join(pal))

# Limpeza dos dados recomendo instancia de http 
def dados_limp(inst):
    inst = re.sub(r"http\S+", "", instancia).lower().replace('.','').replace(';','').replace('-','').replace(':','').replace(')','')
    return (inst)

# Dados limpados da coluna texto
train = [Preprocessing(i) for i in train]
train[:1000]

# Treino e teste do modelo
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train, test, test_size = 0.3, random_state = 0)

# Sklearn Tfidf Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df=0.7)
tf_train = tfidf_vectorizer.fit_transform(x_train)
tf_test = tfidf_vectorizer.transform(x_test)

#################### Modelo machine learning ####################
# Modelo Passive Aggressive Classifier
from sklearn.linear_model import PassiveAggressiveClassifier

# Nome do algoritmo M.L
model_passive_aggressive = PassiveAggressiveClassifier(max_iter=50)

# Treinamento do modelo
model_passive_aggressive_fit = model_passive_aggressive.fit(tf_train, y_train)

# Score do modelo
model_passive_aggressive_score = model_passive_aggressive.score(tf_train, y_train)

# Previsão do modelo
model_passive_aggressive_predict = model_passive_aggressive.predict(tf_test)

print("modelo passive aggressive: %.2f" % (model_passive_aggressive_score * 100))

# Accuracy do modelo - Passive aggressive classifier
from sklearn.metrics import accuracy_score

accuracy_passive_aggressive = accuracy_score(y_test, model_passive_aggressive_predict)
print("Accuracy - Passive aggressive classifier: %.2f" % (accuracy_passive_aggressive * 100))

# Confusion matrix
from sklearn.metrics import confusion_matrix
matrix_confusion = confusion_matrix(y_test, model_passive_aggressive_predict)

matrix_confusion = confusion_matrix(y_test, model_passive_aggressive_predict)
ax = plt.subplot()
sns.heatmap(matrix_confusion, annot=True, ax = ax, fmt = ".1f", cmap="plasma"); 
ax.set_title('Confusion Matrix - Passive Aggressive'); 
ax.xaxis.set_ticklabels(["Fake", "Real"]); ax.yaxis.set_ticklabels(["Fake", "Real"]);

# Classification report
from sklearn.metrics import classification_report
classification = classification_report(y_test, model_passive_aggressive_predict)
print(classification)

# Métricas do modelo ML

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

precision = precision_score(y_test, model_passive_aggressive_predict)
Recall = recall_score(y_test, model_passive_aggressive_predict)
Accuracy = accuracy_score(y_test, model_passive_aggressive_predict)
F1_Score = f1_score(y_test, model_passive_aggressive_predict)

precisao = pd.DataFrame({
    
    "Metricas" : ["precision",
                 "Recall", 
                  "Accuracy", 
                  "F1_Score"],
    
    "Resultado": [precision,
                Recall, 
                Accuracy, 
                F1_Score]})

precisao.sort_values(by = "Resultado", ascending = False)

## Salvando modelo M.L PLN

import pickle
 
with open('model_passive_aggressive_predict.pkl', 'wb') as file:
    pickle.dump(model_passive_aggressive_predict, file)