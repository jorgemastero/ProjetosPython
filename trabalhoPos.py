import pandas as pd
import numpy as np
import re
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import skfuzzy.control as ctrl
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter
import warnings
from bs4 import BeautifulSoup
import requests

# Suprimir o FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# Conjunto de dados é fundamental para realizar tarefas de tokenização na sua análise de linguagem natural
nltk.download('punkt') 
nltk.download('stopwords') 
#WordNet é como um dicionário eletrônico que contém informações sobre o significado das palavras em inglês, suas relações semânticas (sinônimos, antônimos, hiperônimos, hipônimos) e exemplos de uso.
nltk.download('wordnet')  
nltk.download('vader_lexicon')

def site_de_busca():
    print('Escolha onde você deseja realizar a busca: Amazon ou Mercado livre: ')
    buscador_de_preco = input("")
    if buscador_de_preco == "Amazon":
        url = 'https://www.amazon.com.br/Fritadeira-Elétrica-Start-Elgin-Litros/dp/B0CFG1JZGY?th=1'
        try:
            requisicao = requests.get(url)
            site = BeautifulSoup(requisicao.text,'html.parser')

            #CAPTURAR INFORMAÇÕES: TÍTULO DA PÁGINA E PRINCIPAIS AVALIAÇÕES DO PRODUTO

            title = site.find('title')
            review = site.find_all('div',class_='a-expander-content reviewText review-text-content a-expander-partial-collapse-content')

            print(f'Análise para: {title.text}\n')
            print('Principais Avaiações do produto:\n')
            print('******************************************************************************************************************************************************')
            cont = 1
            avaliacoes = []
            for i in review:
                print(f'Review: {cont}')
                print(i.text)
                cont+=1
                print('----------------------------------------------------------------------------------------------------------------------------------------------')
                avaliacoes.append(i.text)

        except Exception as e:
            print(f"Erro: {e}")
            return []
    elif buscador_de_preco == "Mercado livre":
        url = 'https://www.mercadolivre.com.br/fritadeira-air-fryer-eletrica-start-fry-35l-110v-elgin/p/MLB28328271?pdp_filters=item_id:MLB4237860714'
        try:
            requisicao = requests.get(url)
            site = BeautifulSoup(requisicao.text,'html.parser')

            #CAPTURAR INFORMAÇÕES: TÍTULO DA PÁGINA E PRINCIPAIS AVALIAÇÕES DO PRODUTO

            title = site.find('title')
            review = site.find_all('p',class_='ui-review-capability-comments__comment__content ui-review-capability-comments__comment__content')

            print(f'Análise para: {title.text}\n')
            print('Principais Avaiações do produto:\n')
            print('******************************************************************************************************************************************************')
            cont = 1
            avaliacoes = []
            for i in review:
                print(f'Review: {cont}')
                print(i.text)
                cont+=1
                print('----------------------------------------------------------------------------------------------------------------------------------------------')
                avaliacoes.append(i.text)

        except Exception as e:
            print(f"Erro: {e}")
            return []
    else:
        return "Buscador incorreto, por favor digite novamente"
    
    return site_de_busca()

resultado = site_de_busca()

def clean_text(resultado):
    if isinstance(resultado, str): # Testa para ver se o objeto texto é uma string
        # Texto fica todo minúsculo
        text = resultado.lower()

        # Remove caracteres especiais
        text = re.sub(r'[^\w\s]', '', resultado)

        # Tokenizar o texto
        tokens = word_tokenize(resultado)

        # Removendo palavras irrelevantes
        stop_words = set(stopwords.words('english'))
        tokens = [words for word in tokens if word not in stop_words]

        # Iniciando o stemmer(PorterStemmer) e o lematizer (WordNetLematizer)
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        # Aplicando stemming para palavras que não precisam de lematização
        processed_tokens = []
        for word in tokens: 
            # Lematiza se a palavra for um verbo (ou outra classe gramatical)
            if lemmatizer.lemmatize(word) != word:
                processed_tokens.append(lemmatizer.lemmatize(word))
            else:
                #se não tiver lemmatização, aplicar stemming
                processed_tokens.append(stemmer.stem(word))

        #retorna o texto processado
        return ' '.join(processed_tokens) # o .join une todos os elementos de uma lista em uma unica string separado pelo elemento dado entre as aspas
    return ' '

def load_and_preprocess_data(resultado):
    df = load_and_preprocess_data(resultado)
    df.describe()
    print(f"DataFrame carregado com {len(df)} avaliações.")