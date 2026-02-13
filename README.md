# 🧠 Breast Cancer Diagnosis - Machine Learning Project

---

## 🔍 Visão Geral do Projeto

Projeto completo de Machine Learning para diagnóstico preditivo de câncer de mama utilizando KNN e ajuste de hiperparâmetros com otimização de recall e ajuste de threshold.

---

## 🎯 Objetivo do Projeto

Desenvolver um modelo capaz de classificar tumores como:

- **Maligno (1)**
- **Benigno (0)**

Priorizando **Recall**, pois em contexto médico o erro mais crítico é classificar um tumor maligno como benigno (Falso Negativo).

---

## 🧠 Estratégia Técnica

O projeto foi estruturado em etapas profissionais:

### 1️⃣ Análise Exploratória de Dados (EDA)
- Distribuição das classes
- Análise de outliers (IQR)
- Correlação entre variáveis
- Identificação de desbalanceamento

### 2️⃣ Pipeline de Modelagem
- Padronização com `StandardScaler`
- Classificador `KNN`
- Pipeline estruturado com `scikit-learn`

### 3️⃣ Otimização de Hiperparâmetros
- `GridSearchCV`
- Métrica priorizada: **Recall**
- Validação cruzada estratificada

### 4️⃣ Ajuste de Threshold
Após o treinamento:

- Cálculo da Curva ROC
- Cálculo do AUC Score
- Aplicação do **Índice de Youden (TPR - FPR)**
- Definição do melhor threshold para equilibrar Recall e Precisão

---

## 📊 Métricas Avaliadas

- Recall (prioritária)
- Precisão
- F1-score
- AUC-ROC
- Matriz de Confusão

---

## 📂 Dataset Utilizado
- **Fonte:** Kaggle  
- **Nome:** Breast Cancer Wisconsis (Diagnostic) Dataset
- **Link:**  
  👉 https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data  

O dataset contém informações médicas sobre tumores como tamanho, concavidade, simetria, dimensão, etc.

---

## 🛠️ Tecnologias Utilizadas
Este projeto foi desenvolvido utilizando as seguintes ferramentas e bibliotecas:

- 🐍 **Python 3**
- 📊 **Pandas** — Manipulação e análise de dados  
- 🔢 **NumPy** — Operações numéricas  
- 📈 **Matplotlib** — Visualizações gráficas  
- 🎨 **Seaborn** — Visualizações estatísticas avançadas 
- 🧠 **Scikit-learn** - Pipeline, treino, implementação e ajuste do modelo KNN
- 📦 **Joblib** - Exportação do Modelo

---

## 🚀 Por que este projeto é relevante?
✅ Demonstra domínio do biblioteca Python **Scikit-Learn**  
✅ Aplica conceitos de Machine Learning em um **Cenário Real de Medicina**  
✅ Mostra capacidade de ajuste de hiperparâmetros e **Validação de um Modelo de Classificação**  

---

## 📌 Como executar o projeto:
1. Clone este repositório no Git
```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
```

2. Execute o arquivo requirements.txt no seu Terminal através do comando abaixo
```bash
pip install -r requirements.txt
```

3. Abra o arquivo main.py na sua IDE e pronto

---

## 👇 Gostou do meu projeto?
Considere dar uma estrela e me seguir aqui no Github e nas plataformas abaixo:
- LinkedIn: https://www.linkedin.com/in/matheus-mesquita-cintra-carvalho-a76509341/
- Kaggle: https://www.kaggle.com/mesquitam21

Abraços! 🚀
