import pandas as pd
import numpy as np
import kagglehub

from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURAÇÕES
# ============================================================

TARGET_COLUMN = "income"

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "educational-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]


# ============================================================
# CARREGAMENTO DO DATASET
# ============================================================

def load_raw_data():
    """
    Carrega o dataset Adult Income.

    Retorna
    -------
    pandas.DataFrame
        Dataset original.
    """

    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "wenruliu/adult-income-dataset",
        "adult.csv"
    )

    return df


# ============================================================
# TRATAMENTO DA VARIÁVEL TARGET
# ============================================================

def prepare_target(df):
    """
    Converte a variável income para valores binários.

    <=50K -> 0
    >50K  -> 1
    """

    y = df[TARGET_COLUMN].str.strip()

    y = y.map({
        "<=50K": 0,
        ">50K": 1
    })

    return y


# ============================================================
# TREINAMENTO
# ============================================================

def load_and_prepare_data():
    """
    Carrega e prepara os dados para treinamento e teste.

    O pré-processamento é aprendido somente a partir
    do conjunto de treinamento.

    Retorna
    -------
    X_train
    X_test
    y_train
    y_test
    preprocessing
    """

    # --------------------------------------------------------
    # 1. Carregar dataset
    # --------------------------------------------------------

    df = load_raw_data()

    print("Valores faltantes antes:")

    print(
        df.isnull().sum()
    )

    # Substitui '?' por NaN
    df = df.replace("?", pd.NA)

    print("\nValores faltantes após substituir '?':")

    print(
        df.isnull().sum()
    )

    # --------------------------------------------------------
    # 2. Separar X e y
    # --------------------------------------------------------

    X = df.drop(
        TARGET_COLUMN,
        axis=1
    ).copy()

    y = prepare_target(df)

    # --------------------------------------------------------
    # 3. Dividir treino e teste
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # 4. Tratamento dos valores ausentes
    # --------------------------------------------------------

    categorical_columns = [
        column
        for column in X_train.columns
        if column not in NUMERIC_COLUMNS
    ]

    # Guardamos a moda aprendida no treinamento.
    category_modes = {}

    for column in categorical_columns:

        mode = X_train[column].mode()[0]

        category_modes[column] = mode

        X_train.loc[:, column] = (
            X_train[column].fillna(mode)
        )

        X_test.loc[:, column] = (
            X_test[column].fillna(mode)
        )

    # Valores numéricos
    for column in NUMERIC_COLUMNS:

        median = X_train[column].median()

        X_train.loc[:, column] = (
            X_train[column].fillna(median)
        )

        X_test.loc[:, column] = (
            X_test[column].fillna(median)
        )

    # --------------------------------------------------------
    # 5. One-Hot Encoding
    # --------------------------------------------------------

    X_train = pd.get_dummies(
        X_train
    )

    X_test = pd.get_dummies(
        X_test
    )

    # --------------------------------------------------------
    # IMPORTANTE:
    # garante que teste tenha exatamente as mesmas
    # colunas do treinamento
    # --------------------------------------------------------

    feature_columns = X_train.columns.tolist()

    X_test = X_test.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # --------------------------------------------------------
    # 6. Padronização
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train[NUMERIC_COLUMNS] = X_train[NUMERIC_COLUMNS].astype(float)
    X_test[NUMERIC_COLUMNS] = X_test[NUMERIC_COLUMNS].astype(float)

    X_train[NUMERIC_COLUMNS] = scaler.fit_transform(
        X_train[NUMERIC_COLUMNS]
    )

    X_test[NUMERIC_COLUMNS] = scaler.transform(
        X_test[NUMERIC_COLUMNS]
    )

    # --------------------------------------------------------
    # 7. Conversão para float
    # --------------------------------------------------------

    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    # --------------------------------------------------------
    # 8. Conversão para NumPy
    # --------------------------------------------------------

    X_train = X_train.to_numpy()

    X_test = X_test.to_numpy()

    y_train = (
        y_train
        .to_numpy()
        .reshape(-1, 1)
    )

    y_test = (
        y_test
        .to_numpy()
        .reshape(-1, 1)
    )

    # --------------------------------------------------------
    # 9. Objeto de pré-processamento
    # --------------------------------------------------------

    preprocessing = {

        "feature_columns": feature_columns,

        "numeric_columns": NUMERIC_COLUMNS,

        "category_modes": category_modes,

        "scaler": scaler
    }

    # --------------------------------------------------------
    # Informações
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("DADOS PROCESSADOS")
    print("=" * 60)

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"X_test:  {X_test.shape}"
    )

    print(
        f"y_train: {y_train.shape}"
    )

    print(
        f"y_test:  {y_test.shape}"
    )

    print(
        f"\nNúmero de features: "
        f"{len(feature_columns)}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessing
    )


# ============================================================
# PRÉ-PROCESSAMENTO DE NOVOS DADOS
# ============================================================

def preprocess_new_data(
data,
preprocessing
    ):

    # --------------------------------------------------------
    # 1. Criar DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame([data])

    # --------------------------------------------------------
    # 2. Substituir valores "?" por NaN
    # --------------------------------------------------------

    df = df.replace("?", pd.NA)

    # --------------------------------------------------------
    # 3. Recuperar informações do treinamento
    # --------------------------------------------------------

    numeric_columns = preprocessing["numeric_columns"]

    category_modes = preprocessing["category_modes"]

    feature_columns = preprocessing["feature_columns"]

    scaler = preprocessing["scaler"]

    # --------------------------------------------------------
    # 4. Garantir que as colunas numéricas sejam float
    #
    # Isso é importante porque o StandardScaler retorna
    # valores float.
    # --------------------------------------------------------

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            ).astype(float)

    # --------------------------------------------------------
    # 5. Identificar colunas categóricas
    # --------------------------------------------------------

    categorical_columns = [
        column
        for column in df.columns
        if column not in numeric_columns
    ]

    # --------------------------------------------------------
    # 6. Preencher valores categóricos ausentes
    # --------------------------------------------------------

    for column in categorical_columns:

        if column in category_modes:

            df[column] = df[column].fillna(
                category_modes[column]
            )

    # --------------------------------------------------------
    # 7. Preencher valores numéricos ausentes
    #
    # Utilizamos a média aprendida pelo StandardScaler.
    # --------------------------------------------------------

    for column in numeric_columns:

        if column in df.columns:

            scaler_index = numeric_columns.index(
                column
            )

            mean = scaler.mean_[scaler_index]

            df[column] = df[column].fillna(
                mean
            )

    # --------------------------------------------------------
    # 8. One-Hot Encoding
    # --------------------------------------------------------

    df = pd.get_dummies(df)

    # --------------------------------------------------------
    # 9. Garantir exatamente as mesmas features
    # utilizadas durante o treinamento
    # --------------------------------------------------------

    df = df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # --------------------------------------------------------
    # 10. Garantir novamente que as colunas numéricas
    # sejam float
    # --------------------------------------------------------

    for column in numeric_columns:

        if column in df.columns:

            df[column] = df[column].astype(float)

    # --------------------------------------------------------
    # 11. Padronização
    # --------------------------------------------------------

    df[numeric_columns] = scaler.transform(
        df[numeric_columns]
    )

    # --------------------------------------------------------
    # 12. Garantir que tudo seja float
    # --------------------------------------------------------

    df = df.astype(float)

    # --------------------------------------------------------
    # 13. Converter para NumPy
    # --------------------------------------------------------

    return df.to_numpy()

