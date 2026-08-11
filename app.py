
import streamlit as st
import numpy as np

from mlp.checkpoint import ModelCheckpoint
from datasets.adult import preprocess_new_data


# ============================================================
# CONFIGURAÇÃO
# ============================================================

MODEL_PATH = "models/adult_mlp.pkl"
PREPROCESSING_PATH = "models/preprocessing.pkl"


# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Adult Income - MLP",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# CARREGAMENTO DO MODELO
# ============================================================

@st.cache_resource
def load_model():

    model = ModelCheckpoint.load(
        MODEL_PATH
    )

    return model


@st.cache_resource
def load_preprocessing():

    preprocessing = ModelCheckpoint.load(
        PREPROCESSING_PATH
    )

    return preprocessing


model = load_model()
preprocessing = load_preprocessing()


# ============================================================
# TÍTULO
# ============================================================

st.title("🧠 Predição de Renda — Adult Income")

st.markdown(
    """
    ### Rede Neural MLP

    Esta aplicação utiliza uma **Rede Neural Multilayer Perceptron (MLP)**
    previamente treinada com o dataset **Adult Income**.

    O modelo não é treinado durante a utilização da aplicação.
    Os dados informados abaixo são apenas pré-processados e enviados
    para a rede neural para realizar a predição.
    """
)


# ============================================================
# INFORMAÇÕES DO MODELO
# ============================================================

st.divider()

st.subheader("⚙️ Modelo utilizado")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Arquitetura",
        "105 → 16 → 1"
    )

with col2:

    st.metric(
        "Ativação",
        "ReLU + Sigmoid"
    )

with col3:

    st.metric(
        "Otimizador",
        "Adam"
    )

with col4:

    st.metric(
        "Accuracy",
        "84,54%"
    )


# ============================================================
# FORMULÁRIO
# ============================================================

st.divider()

st.subheader("👤 Dados da pessoa")

st.write(
    "Preencha os dados abaixo para realizar uma previsão."
)


with st.form("prediction_form"):

    # ========================================================
    # DADOS NUMÉRICOS
    # ========================================================

    st.markdown("#### Dados numéricos")

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Idade",
            min_value=17,
            max_value=100,
            value=30,
            step=1
        )

    with col2:

        fnlwgt = st.number_input(
            "FNLWGT",
            min_value=0,
            value=180000,
            step=1000,
            help="Peso final de amostragem utilizado pelo dataset."
        )

    with col3:

        educational_num = st.number_input(
            "Nível educacional",
            min_value=1,
            max_value=16,
            value=10,
            step=1
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        capital_gain = st.number_input(
            "Ganho de capital",
            min_value=0,
            value=0,
            step=100
        )

    with col2:

        capital_loss = st.number_input(
            "Perda de capital",
            min_value=0,
            value=0,
            step=100
        )

    with col3:

        hours_per_week = st.number_input(
            "Horas por semana",
            min_value=1,
            max_value=100,
            value=40,
            step=1
        )


    # ========================================================
    # DADOS CATEGÓRICOS
    # ========================================================

    st.markdown("#### Dados categóricos")

    col1, col2 = st.columns(2)

    with col1:

        workclass = st.selectbox(
            "Classe de trabalho",
            [
                "Private",
                "Self-emp-not-inc",
                "Self-emp-inc",
                "Federal-gov",
                "Local-gov",
                "State-gov",
                "Without-pay",
                "Never-worked"
            ]
        )

        education = st.selectbox(
            "Escolaridade",
            [
                "Bachelors",
                "Some-college",
                "11th",
                "HS-grad",
                "Masters",
                "9th",
                "Assoc-acdm",
                "Assoc-voc",
                "7th-8th",
                "Doctorate",
                "Prof-school",
                "5th-6th",
                "10th",
                "1st-4th",
                "Preschool",
                "12th"
            ]
        )

        marital_status = st.selectbox(
            "Estado civil",
            [
                "Married-civ-spouse",
                "Divorced",
                "Never-married",
                "Separated",
                "Widowed",
                "Married-spouse-absent",
                "Married-AF-spouse"
            ]
        )

        occupation = st.selectbox(
            "Ocupação",
            [
                "Tech-support",
                "Craft-repair",
                "Other-service",
                "Sales",
                "Exec-managerial",
                "Prof-specialty",
                "Handlers-cleaners",
                "Machine-op-inspct",
                "Adm-clerical",
                "Farming-fishing",
                "Transport-moving",
                "Priv-house-serv",
                "Protective-serv",
                "Armed-Forces"
            ]
        )

    with col2:

        relationship = st.selectbox(
            "Relacionamento familiar",
            [
                "Wife",
                "Own-child",
                "Husband",
                "Not-in-family",
                "Other-relative",
                "Unmarried"
            ]
        )

        race = st.selectbox(
            "Raça",
            [
                "White",
                "Black",
                "Asian-Pac-Islander",
                "Amer-Indian-Eskimo",
                "Other"
            ]
        )

        gender = st.selectbox(
            "Gênero",
            [
                "Male",
                "Female"
            ]
        )

        native_country = st.selectbox(
            "País de origem",
            [
                "United-States",
                "Cambodia",
                "England",
                "Puerto-Rico",
                "Canada",
                "Germany",
                "Outlying-US(Guam-USVI-etc)",
                "India",
                "Japan",
                "Greece",
                "South",
                "China",
                "Cuba",
                "Iran",
                "Honduras",
                "Philippines",
                "Italy",
                "Poland",
                "Jamaica",
                "Vietnam",
                "Mexico",
                "Portugal",
                "Ireland",
                "France",
                "Dominican-Republic",
                "Laos",
                "Ecuador",
                "Taiwan",
                "Haiti",
                "Columbia",
                "Hungary",
                "Guatemala",
                "Nicaragua",
                "Scotland",
                "Thailand",
                "Yugoslavia",
                "El-Salvador",
                "Trinadad&Tobago",
                "Peru",
                "Hong",
                "Holand-Netherlands"
            ]
        )


    # ========================================================
    # BOTÃO
    # ========================================================

    submitted = st.form_submit_button(
        "🔮 Realizar previsão",
        use_container_width=True
    )


# ============================================================
# PREDIÇÃO
# ============================================================

if submitted:

    data = {

        "age": age,

        "workclass": workclass,

        "fnlwgt": fnlwgt,

        "education": education,

        "educational-num": educational_num,

        "marital-status": marital_status,

        "occupation": occupation,

        "relationship": relationship,

        "race": race,

        "gender": gender,

        "capital-gain": capital_gain,

        "capital-loss": capital_loss,

        "hours-per-week": hours_per_week,

        "native-country": native_country
    }


    try:

        # ----------------------------------------------------
        # Pré-processamento
        # ----------------------------------------------------

        X = preprocess_new_data(
            data,
            preprocessing
        )


        # ----------------------------------------------------
        # Predição
        # ----------------------------------------------------

        prediction = model.forward(
            X
        )


        probability = float(
            prediction[0][0]
        )


        # ----------------------------------------------------
        # Classe
        # ----------------------------------------------------

        predicted_class = (
            1
            if probability >= 0.5
            else 0
        )


        # ====================================================
        # RESULTADO
        # ====================================================

        st.divider()

        st.subheader("📊 Resultado da previsão")


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Probabilidade de renda > 50K",
                f"{probability * 100:.2f}%"
            )


        with col2:

            if predicted_class == 1:

                st.metric(
                    "Classe prevista",
                    "> 50K"
                )

            else:

                st.metric(
                    "Classe prevista",
                    "<= 50K"
                )


        # ====================================================
        # BARRA DE PROBABILIDADE
        # ====================================================

        st.markdown(
            "### Probabilidade"
        )

        st.progress(
            probability
        )


        # ====================================================
        # INTERPRETAÇÃO
        # ====================================================

        if predicted_class == 1:

            st.success(
                f"""
                A rede neural classificou esta pessoa como
                **renda superior a 50K**, com probabilidade
                estimada de **{probability * 100:.2f}%**.
                """
            )

        else:

            st.info(
                f"""
                A rede neural classificou esta pessoa como
                **renda igual ou inferior a 50K**, com probabilidade
                estimada de **{probability * 100:.2f}%** para a classe >50K.
                """
            )


        # ====================================================
        # DISTRIBUIÇÃO DAS CLASSES
        # ====================================================

        st.markdown(
            "### Distribuição da previsão"
        )

        chart_data = {

            "<= 50K": 1 - probability,

            "> 50K": probability
        }

        st.bar_chart(
            chart_data
        )


        # ====================================================
        # DETALHES TÉCNICOS
        # ====================================================

        with st.expander(
            "🔧 Ver detalhes técnicos"
        ):

            st.write(
                "Entrada após pré-processamento:"
            )

            st.write(
                f"{X.shape[1]} features"
            )

            st.write(
                X
            )

            st.write(
                "Saída bruta da rede:"
            )

            st.write(
                prediction
            )


    except Exception as error:

        st.error(
            "Ocorreu um erro durante a previsão."
        )

        st.exception(error)


# ============================================================
# RODAPÉ
# ============================================================

st.divider()

st.caption(
    "Modelo MLP treinado previamente com o dataset Adult Income. "
    "A aplicação realiza apenas inferência."
)

