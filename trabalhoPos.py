import nltk  # para remoção das stopwords
import re  # para remoção dos números
from nltk.corpus import stopwords  # para remoção das stopwords
from nltk.stem import RSLPStemmer  # para Stemming
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt


nltk.download('stopwords')
nltk.download('rslp')
nltk.download('vader_lexicon')

##referenciando o local do arquivo

caminho = r"C:\\Users\\jorge\\Desktop\\documentos\\Avaliação de produto.txt"

# Processo para tratar o texto

with open(caminho, 'r', encoding='UTF-8') as arquivo:
    texto = arquivo.read()


        # Removendo as stopwords
    def remover_stopwords(linha):
            stopwords_pt = set(stopwords.words('portuguese')).union(
                {"denunciar", "avaliação", "útil" "0", "1", "2", "3", "4", "5"})
            palavras = linha.split()
            palavras_sem_stopwords = [palavra for palavra in palavras if
                                      palavra.lower() not in stopwords_pt and not re.match(r'^\d+$', palavra)]
            return ' '.join(palavras_sem_stopwords)


    # Aplicando Stemming
    def stemming(linha):
            stemmer = RSLPStemmer()
            palavras = linha.split()
            palavras_stemmed = [stemmer.stem(palavra) for palavra in palavras]
            return ' '.join(palavras_stemmed)


texto_sem_stopwords = remover_stopwords(texto)
texto_stemmed = stemming(texto_sem_stopwords)

        

    # Realizando a analise de sentimentos
def analise_de_sentimentos(texto_stemmed):
            sia = SentimentIntensityAnalyzer()
            sentimentos = sia.polarity_scores(texto_stemmed)
            return sentimentos

sentimentos = analise_de_sentimentos(texto_stemmed)

## Expressando gráficamente a análise de sentimentos

plt.bar(sentimentos.keys(), sentimentos.values())
plt.xlabel('Sentimento')
plt.ylabel('Pontuação')
plt.title('Análise de Sentimentos')
plt.show()

