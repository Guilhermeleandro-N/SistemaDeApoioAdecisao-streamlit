# MLP Adult Income

Projeto de classificação utilizando uma **Rede Neural Multilayer Perceptron (MLP)** para prever se a renda anual de uma pessoa é:

* `<=50K`
* `>50K`

O projeto utiliza o dataset **Adult Income**, pré-processamento com `pandas` e `scikit-learn`, uma implementação própria de MLP e uma interface interativa desenvolvida com **Streamlit**.

---

## 1. Estrutura do projeto

```text
.
├── app.py
├── train_model.py
│
├── datasets
│   ├── adult.py
│   └── xor.py
│
├── mlp
│   ├── activations.py
│   ├── checkpoint.py
│   ├── grid_search.py
│   ├── initializers.py
│   ├── __init__.py
│   ├── layers.py
│   ├── losses.py
│   ├── metrics.py
│   ├── network.py
│   └── optimizers.py
│
├── models
│   ├── adult_mlp.pkl
│   └── preprocessing.pkl
│
├── README.md
└── requirements.txt
```

### Principais arquivos

| Arquivo                    | Função                                              |
| -------------------------- | --------------------------------------------------- |
| `app.py`                   | Interface gráfica e interativa utilizando Streamlit |
| `train_model.py`           | Treinamento da rede neural e geração do modelo      |
| `datasets/adult.py`        | Carregamento e pré-processamento do Adult Income    |
| `mlp/network.py`           | Implementação da rede neural                        |
| `mlp/layers.py`            | Implementação das camadas Dense                     |
| `mlp/activations.py`       | Funções de ativação                                 |
| `mlp/losses.py`            | Funções de perda                                    |
| `mlp/optimizers.py`        | Algoritmos de otimização                            |
| `mlp/metrics.py`           | Métricas de avaliação                               |
| `mlp/checkpoint.py`        | Salvamento e carregamento do modelo                 |
| `models/adult_mlp.pkl`     | Modelo treinado                                     |
| `models/preprocessing.pkl` | Parâmetros utilizados no pré-processamento          |

---

# 2. Requisitos

O projeto foi desenvolvido utilizando Python 3.12.

É recomendado utilizar um ambiente virtual (`venv`) para instalar as dependências.

No Ubuntu, verifique a versão do Python:

```bash
python3 --version
```

---

# 3. Clonar o projeto

Caso o projeto esteja hospedado em um repositório Git, clone-o utilizando:

```bash
git clone URL_DO_REPOSITORIO
```

Entre na pasta do projeto:

```bash
cd NOME_DO_PROJETO
```

---

# 4. Criar o ambiente virtual

No Ubuntu:

```bash
python3 -m venv .venv
```

Ative o ambiente virtual:

```bash
source .venv/bin/activate
```

Após a ativação, o terminal deverá apresentar algo semelhante a:

```text
(.venv) usuario@computador:~/projeto$
```

---

# 5. Instalar as dependências

Com o ambiente virtual ativado:

```bash
pip install -r requirements.txt
```

Caso o `pip` esteja desatualizado, pode ser atualizado com:

```bash
pip install --upgrade pip
```

E então:

```bash
pip install -r requirements.txt
```

---

# 6. Executar a aplicação

O projeto possui um modelo já treinado em:

```text
models/adult_mlp.pkl
```

e o pré-processamento utilizado pelo modelo em:

```text
models/preprocessing.pkl
```

Portanto, **não é necessário treinar a rede novamente para utilizar a aplicação**.

Com o ambiente virtual ativado, execute:

```bash
streamlit run app.py
```

O Streamlit iniciará um servidor local e exibirá no terminal um endereço semelhante a:

```text
Local URL: http://localhost:8501
```

Abra esse endereço no navegador.

---

# 7. Utilização da aplicação

A aplicação permite inserir os dados de uma pessoa e realizar uma previsão utilizando a MLP previamente treinada.

O fluxo é:

```text
Dados informados pelo usuário
            ↓
Pré-processamento
            ↓
One-Hot Encoding
            ↓
Padronização
            ↓
Modelo MLP
            ↓
Probabilidade
            ↓
Classificação
```

A saída da rede utiliza uma função **Sigmoid**, produzindo um valor entre `0` e `1`.

A classificação é realizada utilizando o limite de `0.5`:

```text
probabilidade > 0.5 → >50K
probabilidade ≤ 0.5 → <=50K
```

---

# 8. Treinar o modelo novamente

O treinamento é separado da aplicação.

Para treinar uma nova versão da rede, execute:

```bash
python3 train_model.py
```

O processo realiza:

1. Carregamento do dataset Adult Income;
2. Tratamento dos valores ausentes;
3. Separação entre dados de treinamento e teste;
4. One-Hot Encoding;
5. Padronização das variáveis numéricas;
6. Construção da MLP;
7. Treinamento da rede;
8. Avaliação no conjunto de treinamento;
9. Avaliação no conjunto de teste;
10. Salvamento do modelo;
11. Salvamento do pré-processamento.

Ao final, serão gerados/atualizados:

```text
models/adult_mlp.pkl
models/preprocessing.pkl
```

Depois de treinar novamente, execute a aplicação:

```bash
streamlit run app.py
```

---

# 9. Arquitetura da MLP

A configuração utilizada no modelo treinado é:

```text
Entrada: 105 features
      ↓
Dense: 16 neurônios + ReLU
      ↓
Dense: 1 neurônio + Sigmoid
      ↓
Saída
```

Configurações utilizadas no treinamento:

| Parâmetro                   |   Valor |
| --------------------------- | ------: |
| Neurônios na camada oculta  |      16 |
| Função de ativação          |    ReLU |
| Função de ativação da saída | Sigmoid |
| Otimizador                  |    Adam |
| Learning rate               |   0.001 |
| Épocas                      |     100 |
| Função de perda             |     MSE |
| Número de features          |     105 |

---

# 10. Pré-processamento

O dataset possui variáveis numéricas e categóricas.

As variáveis numéricas utilizadas são:

```text
age
fnlwgt
educational-num
capital-gain
capital-loss
hours-per-week
```

Essas variáveis são padronizadas utilizando `StandardScaler`.

As variáveis categóricas são transformadas utilizando **One-Hot Encoding**.

Durante o treinamento, os parâmetros de pré-processamento são aprendidos apenas a partir do conjunto de treinamento.

Esses parâmetros são armazenados em:

```text
models/preprocessing.pkl
```

Isso permite que novos dados recebam exatamente o mesmo tratamento utilizado durante o treinamento.

---

# 11. Modelo salvo

O arquivo:

```text
models/adult_mlp.pkl
```

contém a rede neural já treinada.

A aplicação carrega esse arquivo para realizar as previsões, evitando a necessidade de executar o treinamento toda vez que o Streamlit é iniciado.

Portanto:

```text
train_model.py
       ↓
treina a MLP
       ↓
adult_mlp.pkl
       ↓
app.py
       ↓
faz previsões
```

---

# 12. Executar apenas a aplicação

Para utilização normal do projeto, basta:

```bash
source .venv/bin/activate
streamlit run app.py
```

Não é necessário executar:

```bash
python3 train_model.py
```

a cada execução da aplicação.

O treinamento deve ser realizado somente quando for necessário gerar uma nova versão do modelo.

---

# 13. Parar a aplicação

Para encerrar o Streamlit, volte ao terminal onde ele está sendo executado e pressione:

```text
Ctrl + C
```

---

# 14. Problemas comuns

## Ambiente virtual não ativado

Se aparecer erro relacionado a módulos não encontrados, verifique se o ambiente virtual está ativo:

```bash
source .venv/bin/activate
```

Depois instale as dependências:

```bash
pip install -r requirements.txt
```

---

## Streamlit não encontrado

Execute:

```bash
pip install streamlit
```

ou reinstale todas as dependências:

```bash
pip install -r requirements.txt
```

Depois:

```bash
streamlit run app.py
```

---

## Modelo não encontrado

Verifique se os arquivos existem:

```bash
ls models/
```

Deve aparecer:

```text
adult_mlp.pkl
preprocessing.pkl
```

Se os arquivos não existirem, execute o treinamento:

```bash
python3 train_model.py
```

---

# 15. Fluxo recomendado

### Primeira utilização

```bash
git clone URL_DO_REPOSITORIO
cd NOME_DO_PROJETO

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

streamlit run app.py
```

### Caso seja necessário treinar novamente

```bash
source .venv/bin/activate

python3 train_model.py

streamlit run app.py
```

---

## Tecnologias utilizadas

* Python
* NumPy
* Pandas
* Scikit-learn
* SciPy
* KaggleHub
* Streamlit
* Rede Neural MLP implementada manualmente
* Git/GitHub

---

## Objetivo do projeto

O projeto tem como objetivo demonstrar a implementação e utilização de uma **Rede Neural Multilayer Perceptron** para um problema de classificação binária, desde o pré-processamento dos dados e treinamento da rede até a utilização do modelo treinado por meio de uma aplicação interativa.
