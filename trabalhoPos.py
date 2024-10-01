import nltk #para remoção das stopwords
import re #para remoção dos números
from nltk.corpus import stopwords #para remoção das stopwords
from nltk.stem import RSLPStemmer #para Stemming
from googletrans import Translator
from nltk.sentiment.vader import SentimentIntensityAnalyzer


nltk.download('stopwords')
nltk.download('rslp')
nltk.download('vader_lexicon')


##referenciando o local do arquivo

caminho = r"C:\\Users\\jorge\\Desktop\\documentos\\Avaliação de produto.txt"

#Processo para tratar o texto 

with open('Avaliação de produto.txt', 'r', encoding='UTF-8') as arquivo:
    linhas = arquivo.readlines()
    for linha in linhas:
        linha = linha.strip()
        
        # Removendo as stopwords
        def remover_stopwords(linha):
            stopwords_pt = set(stopwords.words('portuguese')).union({"denunciar", "avaliação", "útil" "0", "1", "2", "3", "4", "5"})
            palavras = linha.split()
            palavras_sem_stopwords = [palavra for palavra in palavras if palavra.lower() not in stopwords_pt and not re.match(r'^\d+$', palavra)]
            return ' '.join(palavras_sem_stopwords)

        # Aplicando Stemming
        def stemming(linha):
            stemmer = RSLPStemmer()
            palavras = linha.split()
            palavras_stemmed = [stemmer.stem(palavra) for palavra in palavras]
            return ' '.join(palavras_stemmed)
        
        texto_sem_stopwords = remover_stopwords(linha)
        texto_stemmed = stemming(texto_sem_stopwords)

        # Transformar o texto para o inglês para que seja mais facil de realizar a analise de sentimentos 

        translator = Translator()
            
        def traduzindo_texto(texto_stemmed, destino='en'):
            traducao = translator.translate(texto_stemmed, dest=destino)
            return traducao.text
            
                    translator = Translator()
            
        def traduzindo_texto(texto_stemmed, destino='en'):
            traducao = translator.translate(texto_stemmed, dest=destino)
            return traducao.text
            
        print(linha)