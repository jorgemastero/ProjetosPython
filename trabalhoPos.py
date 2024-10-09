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

def buscar_avaliacoes(url):
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
        return avaliacoes
    
    except Exception as e:
        print(f"Erro: {e}")
        return []

def limpar_texto(texto):
    if isinstance(texto, str): 
        texto = texto.lower()
        texto = re.sub(r'[^\w\s]', '', texto)
        tokens = word_tokenize(texto)
        stop_words = set(stopwords.words('portuguese'))
        tokens = [word for word in tokens if word not in stop_words]
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        processed_tokens = []
        for word in tokens: 
            if lemmatizer.lemmatize(word) != word:
                processed_tokens.append(lemmatizer.lemmatize(word))
            else:
                processed_tokens.append(stemmer.stem(word))
        return ' '.join(processed_tokens) 
    return ' '

def realizar_analise_lda(df, n_topics=10):
    print("Iniciando análise de tópicos com LDA...")
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(df['processed_text'])

    lda_model = LatentDirichletAllocation(n_components=n_topics, max_iter=20, random_state=42)
    lda_output = lda_model.fit_transform(tfidf_matrix)

    df['topic'] = lda_output.argmax(axis=1)

    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for idx, topic in enumerate(lda_model.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[-5:]]
        topics.append((f"Tópico {idx + 1}", ', '.join(topic_words)))

    print("Análise de tópicos concluída.")
    return df, topics

def analisar_sentimento_vader(df):
    print("Iniciando análise de sentimentos com VADER...")
    sia = SentimentIntensityAnalyzer()

    def get_sentimento(texto):
        return sia.polarity_scores(texto)['compound']

    def classificar_sentimento(polaridade):
        if polaridade > 0.1:
            return 'Positiva'
        elif polaridade < -0.1:
            return 'Negativa'
        return 'Neutra'

    df['sentimento_vader'] = df['reviewDescription'].apply(get_sentimento)
    df['sentimento_class'] = df['sentimento_vader'].apply(classificar_sentimento)
    print("Análise de sentimentos concluída.")
    return df

def plotar_distribuicao_sentimento(df, coluna_sentimento, titulo):
    sentiment_counts = df[coluna_sentimento].value_counts()
    plt.figure(figsize=(8, 6))
    sentiment_counts.plot(kind='bar', color=['green', 'gray', 'red'])
    plt.title(titulo)
    plt.ylabel('Número de Avaliações')
    plt.xlabel('Sentimento')
    plt.xticks(rotation=0)
    plt.show()

def main():
    print('Escolha onde você deseja realizar a busca: Amazon ou Mercado livre: ')
    buscador_de_preco = input("")
    if buscador_de_preco == "Amazon":
        url = 'https://www.amazon.com.br/Fritadeira-Elétrica-Start-Elgin-Litros/dp/B0CFG1JZGY?th=1'
    elif buscador_de_preco == "Mercado livre":
        url = 'https://www.mercadolivre.com.br/fritadeira-air-fryer-eletrica-start-fry-35l-110v-elgin/p/MLB28328271?pdp_filters=item_id:MLB4237860714'
    else:
        print("Buscador incorreto, por favor digite novamente")
        return

    avaliacoes = buscar_avaliacoes(url)
    if not avaliacoes:
        print("Não foi possível encontrar avaliações no site.")
        return

    texto_tratado = [limpar_texto(texto) for texto in avaliacoes]
    for i, texto in enumerate(texto_tratado):
        print(f"Texto tratado {i+1}: {texto}")

    df = pd.DataFrame(texto_tratado, columns=['reviewDescription'])
    df['processed_text'] = df['reviewDescription'].apply(limpar_texto)
    print(f"Pré-processamento concluído. {df.shape[0]} avaliações processadas.")

    df, topics = realizar_analise_lda(df)
    print("Tópicos encontrados:")
    for topic in topics:
        print(topic)

    df = analisar_sentimento_vader(df)
    plotar_distribuicao_sentimento(df, 'sentimento_class', 'Distribuição de Sentimentos')

    # Salvar o dataframe em um arquivo CSV
    df.to_csv('avaliacoes.csv', index=False)



# Seção 5: Nuvem de Palavras
def generate_wordcloud(df, column):
    """Gera uma nuvem de palavras a partir de uma coluna de texto e exibe as top 10 palavras presentes na nuvem de palavras."""
    print("Gerando nuvem de palavras...")
     # Junta todas as palavras da coluna em uma única string
    all_words = ' '.join(df[column])

    # Remover caracteres especiais e números
    all_words = re.sub(r'[^\w\s]', '', all_words)

    # Cria a nuvem de palavras
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

    # Extrai as palavras que aparecem na nuvem (sem stopwords ou caracteres irrelevantes)
    words_in_wordcloud = wordcloud.words_.keys()

    # Conta a frequência de cada palavra que aparece na nuvem
    words_list = [word for word in all_words.split() if word in words_in_wordcloud]
    word_freq = Counter(words_list)

    # Exibe as top 10 palavras mais frequentes que aparecem na nuvem
    top_10_words = word_freq.most_common(10)
    print("\nTop 10 palavras mais frequentes na nuvem de palavras:")
    for word, freq in top_10_words:
        print(f"{word}: {freq} ocorrências")

    # Plota a nuvem de palavras
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Nuvem de Palavras das Avaliações')
    plt.show()

    print("Nuvem de palavras gerada.")

    # Seção 6: Lógica Fuzzy
def fuzzy_logic(df):
    """Aplica a lógica fuzzy para calcular a qualidade das avaliações com base nas discrepâncias entre rating e sentimento."""
    print("Iniciando análise fuzzy...")

    # Definindo os antecedentes (inputs)
    rating = ctrl.Antecedent(np.arange(1, 6, 0.1), 'rating')  # Nota da avaliação (1 a 5)
    sentiment = ctrl.Antecedent(np.arange(-1, 1.1, 0.1), 'sentiment')  # Sentimento do comentário (-1 a 1)

    # Definindo a consequente (output)
    quality = ctrl.Consequent(np.arange(0, 11, 0.1), 'quality')  # Qualidade da revisão (0 a 10)

    # Definindo as variáveis fuzzy para a nota (rating)
    rating['low'] = fuzz.trimf(rating.universe, [1, 1, 3])
    rating['medium'] = fuzz.trimf(rating.universe, [2, 3, 4])
    rating['high'] = fuzz.trimf(rating.universe, [3, 5, 5])

    # Definindo as variáveis fuzzy para o sentimento
    sentiment['negative'] = fuzz.trimf(sentiment.universe, [-1, -1, 0])
    sentiment['neutral'] = fuzz.trimf(sentiment.universe, [-0.5, 0, 0.5])
    sentiment['positive'] = fuzz.trimf(sentiment.universe, [0, 1, 1])

    # Definindo as variáveis fuzzy para a qualidade (output)
    quality['poor'] = fuzz.trimf(quality.universe, [0, 0, 5])
    quality['average'] = fuzz.trimf(quality.universe, [0, 5, 10])
    quality['excellent'] = fuzz.trimf(quality.universe, [5, 10, 10])

    # Regras fuzzy para determinar a qualidade da revisão
    rule1 = ctrl.Rule(rating['low'] & sentiment['negative'], quality['excellent'])  # Nota baixa e sentimento negativo = qualidade excelente
    rule2 = ctrl.Rule(rating['low'] & sentiment['neutral'], quality['average'])  # Nota baixa e sentimento neutro = qualidade média
    rule3 = ctrl.Rule(rating['low'] & sentiment['positive'], quality['poor'])  # Nota baixa e sentimento positivo = qualidade ruim

    rule4 = ctrl.Rule(rating['medium'] & sentiment['negative'], quality['average'])  # Nota média e sentimento negativo = qualidade média
    rule5 = ctrl.Rule(rating['medium'] & sentiment['neutral'], quality['excellent'])  # Nota média e sentimento neutro = qualidade excelente
    rule6 = ctrl.Rule(rating['medium'] & sentiment['positive'], quality['average'])  # Nota média e sentimento positivo = qualidade média

    rule7 = ctrl.Rule(rating['high'] & sentiment['negative'], quality['poor'])  # Nota alta e sentimento negativo = qualidade ruim
    rule8 = ctrl.Rule(rating['high'] & sentiment['neutral'], quality['average'])  # Nota alta e sentimento neutro = qualidade média
    rule9 = ctrl.Rule(rating['high'] & sentiment['positive'], quality['excellent'])  # Nota alta e sentimento positivo = qualidade excelente

    # Sistema de controle fuzzy
    quality_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7,rule8,rule9])
    quality_sim = ctrl.ControlSystemSimulation(quality_ctrl)

    # Função para calcular a qualidade da revisão
    def calculate_review_quality(rating_value, sentiment_value=None):
        quality_sim.input['rating'] = rating_value
        quality_sim.input['sentiment'] = sentiment_value if sentiment_value is not None else 0
        quality_sim.compute()
        return quality_sim.output['quality']

    # Aplicar a lógica fuzzy ao DataFrame
    df['review_quality'] = df.apply(lambda row: calculate_review_quality(row['ratingScore'], row['sentiment_vader']), axis=1)

    print("Análise fuzzy concluída.")
    return df



print(df.head())  # Imprime as primeiras linhas do dataframe
fuzzy_logic(df)
# Salvar o dataframe em um arquivo CSV
df.to_csv('avaliacoes_fuzzy.csv', index=False)

if __name__ == "__main__":
    main()