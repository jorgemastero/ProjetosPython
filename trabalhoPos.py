import nltk  # para remoção das stopwords
import re  # para remoção dos números
from nltk.corpus import stopwords  # para remoção das stopwords
from nltk.stem import RSLPStemmer  # para Stemming
import matplotlib.pyplot as plt
from googletrans import Translator
import numpy as np
import skfuzzy as fuzz

nltk.download('stopwords')
nltk.download('rslp')

##referenciando o local do arquivo

caminho = r"C:\\Users\\jorge\\Desktop\\documentos\\Avaliação de produto.txt"

# Processo para tratar o texto

with open(caminho, 'r', encoding='UTF-8') as arquivo:
    texto = arquivo.read()

translator = Translator()
texto1 = texto
traducao = translator.translate(texto1, src='pt', dest='en')

texto_traduzido = traducao.text

# Removendo as stopwords
def remover_stopwords(texto_traduzido):
    stopwords_pt = set(stopwords.words('english')).union({"report", "assessment", "useful", "0", "1", "2", "3", "4", "5"})
    palavras = texto_traduzido.split()
    palavras_sem_stopwords = [palavra for palavra in palavras if palavra.lower() not in stopwords_pt and not re.match(r'^\d+﹩', palavra)]
    return ' '.join(palavras_sem_stopwords)

def stemming(texto_traduzido):
    stemmer = RSLPStemmer()
    palavras = texto_traduzido.split()
    palavras_stemmed = [stemmer.stem(palavra) for palavra in palavras]
    return ' '.join(palavras_stemmed)


texto_sem_stopwords = remover_stopwords(texto_traduzido)
texto_stemmed = stemming(texto_sem_stopwords)

print(texto_stemmed)