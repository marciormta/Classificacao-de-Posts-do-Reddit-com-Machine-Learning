# Classificação de Textos do Reddit com Machine Learning

## Visão Geral

Este projeto tem como objetivo classificar textos extraídos do Reddit em cinco categorias específicas: `datascience`, `machinelearning`, `physics`, `astrology`, e `conspiracy`. Utilizando técnicas avançadas de Machine Learning, o projeto explora diferentes metodologias para entender e categorizar conteúdos, oferecendo insights sobre como diferentes tópicos são discutidos na comunidade do Reddit.

## Tecnologias Utilizadas

- **Python**: Linguagem principal do projeto.
- **PRAW (Python Reddit API Wrapper)**: Utilizado para acessar e extrair dados diretamente dos subreddits do Reddit.
- **Scikit-learn**: Usada para implementar algoritmos de classificação como **K-Nearest Neighbors, Random Forest e Logistic Regression**.
- **Numpy**: Essencial para manipulação de arrays e matrizes de dados.
- **Matplotlib & Seaborn**: Utilizadas para criar gráficos que ajudam a analisar os resultados dos modelos.

## Funcionalidades

- **Extração de Dados**: Scripts configurados para extrair posts de subreddits específicos, utilizando critérios como tamanho do texto para garantir a qualidade e relevância dos dados.
- **Pré-processamento de Dados**: Implementação de técnicas de limpeza e preparação de dados, como a tokenização e a aplicação de TF-IDF para conversão de textos em formatos numéricos adequados para análise.
- **Modelagem e Avaliação**: Uso de múltiplos modelos de machine learning para avaliar qual oferece a melhor precisão e performance na classificação dos textos.
- **Visualização de Resultados**: Gráficos e relatórios que facilitam a interpretação dos resultados dos modelos e a comparação de suas performances.


## Conteúdo
- [Instalação](#instalação)

## Instalação

1. Clone o repositório:
- git clone https://github.com/seu_usuario/nome_do_repositorio.git

2. Navegue até o diretório do projeto:
- cd nome_do_repositorio

4. Crie e ative um ambiente virtual (opcional, mas recomendado):
- python -m venv venv
- source venv/bin/activate

5. Instale as dependências do projeto:
- pip install -r requirements.txt

7. Gere credenciais da API do reddit através desse link:
- https://www.reddit.com/prefs/apps



