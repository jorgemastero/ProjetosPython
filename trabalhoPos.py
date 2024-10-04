import nltk  # para remoção das stopwords
import re  # para remoção dos números
from nltk.corpus import stopwords  # para remoção das stopwords
from nltk.stem import RSLPStemmer  # para Stemming
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from googletrans import Translator
import numpy as np
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('rslp')
nltk.download('vader_lexicon')

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

texto_tokenizado = word_tokenize(texto_stemmed)
frases = texto_tokenizado

# Definir as funções de pertinência fuzzy com ajuste para capturar melhor os extremos
def pertinencia_positivo(x):
    return np.maximum(0, x)  # Pertinência positiva quando a pontuação é acima de 0

def pertinencia_negativo(x):
    return np.maximum(0, -x)  # Pertinência negativa quando a pontuação é abaixo de 0

def pertinencia_neutro(x):
    return 1 - np.abs(x)  # Pertinência neutra quando a pontuação está próxima de 0

# Inicializar o analizador de sentimento do VADER

sia = SentimentIntensityAnalyzer()

# Calcular a pontuação de sentimento para cada frase
pontuacoes_sentimento = [sia.polarity_scores(frase)['compound'] for frase in frases]

# Classificar o sentimento de cada frase
sentimentos = []
palavras_positivas = []
palavras_negativas = []
for i, pontuacao in enumerate(pontuacoes_sentimento):
    positivo = pertinencia_positivo(pontuacao)
    negativo = pertinencia_negativo(pontuacao)
    neutro = pertinencia_neutro(pontuacao)

    # Comparar os valores fuzzy e determinar o sentimento predominante
    if positivo > negativo and positivo > neutro:
        sentimentos.append("Positivo")
        palavras_positivas.extend(frases[i].split())
    elif negativo > positivo and negativo > neutro:
        sentimentos.append("Negativo")
        palavras_negativas.extend(frases[i].split())
    else:
        sentimentos.append("Neutro")

# Exibir os sentimentos classificados

# Criar uma nuvem de palavras para as palavras positivas
wordcloud_positivas = WordCloud(width=800, height=400, max_words=100).generate(' '.join(palavras_positivas))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_positivas, interpolation='bilinear')
plt.axis('off')
plt.title('Palavras Positivas')
plt.show()

# Criar uma nuvem de palavras para as palavras negativas
wordcloud_negativas = WordCloud(width=800, height=400, max_words=100).generate(' '.join(palavras_negativas))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_negativas, interpolation='bilinear')
plt.axis('off')
plt.title('Palavras Negativas')
plt.show()

# Criar um gráfico de barras para os sentimentos
plt.figure(figsize=(10, 5))
plt.bar(['Positivo', 'Negativo', 'Neutro'], [sentimentos.count('Positivo'), sentimentos.count('Negativo'), sentimentos.count('Neutro')])
plt.xlabel('Sentimento')
plt.ylabel('Frequência')
plt.title('Distribuição dos Sentimentos')
plt.show()