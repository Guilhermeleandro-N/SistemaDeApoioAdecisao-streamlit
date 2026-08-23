import streamlit as st
import numpy as np
import shap
import matplotlib.pyplot as plt

from mlp.checkpoint import ModelCheckpoint

from mlp.explainability import generate_shap_summary
from datasets.adult import (
    preprocess_new_data,
    load_and_prepare_data
)

# ============================================================
# CONFIGURAÇÃO
# ============================================================

MODEL_PATH = "models/adult_mlp.pkl"
PREPROCESSING_PATH = "models/preprocessing.pkl"


# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Predição de Renda - MLP",
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


@st.cache_data
def load_explanation_data():

    (
        X_train,
        X_test,
        y_train,
        y_test,
        _
    ) = load_and_prepare_data()

    return X_train, X_test

# ============================================================
# TÍTULO
# ============================================================

st.title("🧠 Predição de Renda — Adult Income")

st.markdown(
    """
    ### Rede Neural Perceptron Multicamadas (MLP)

    Esta aplicação utiliza uma **Rede Neural Perceptron Multicamadas (MLP)**
    previamente treinada com o conjunto de dados **Adult Income**.

    O modelo não é treinado durante a utilização da aplicação.
    Os dados informados abaixo são apenas pré-processados e enviados
    para a rede neural para realizar a previsão.
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
        "Acurácia",
        "84,54%"
    )

# ============================================================
# GUIA DOS CAMPOS
# ============================================================

st.divider()

st.subheader("📖 Guia dos campos")

st.write(
    """
    Consulte a tabela abaixo para entender o significado de cada
    campo e das opções utilizadas pelo dataset Adult Income.
    """
)

field_guide = [

    {
        "Campo": "age",
        "Significado": "Idade da pessoa.",
        "Como responder": "Informe a idade em anos. Ex.: 30 = pessoa com 30 anos.",
        "Exemplo": "30"
    },

    {
        "Campo": "workclass",
        "Significado": "Tipo de vínculo ou classe de trabalho.",
        "Como responder": "Private = empresa privada; Self-emp-not-inc = autônomo sem empresa constituída; Self-emp-inc = autônomo com empresa; Federal-gov = governo federal; Local-gov = governo local; State-gov = governo estadual; Without-pay = trabalho não remunerado; Never-worked = nunca trabalhou.",
        "Exemplo": "Private"
    },

    {
        "Campo": "fnlwgt",
        "Significado": "Peso final de amostragem utilizado pelo dataset.",
        "Como responder": "É um valor numérico utilizado para representar quantas pessoas da população possuem características semelhantes ao registro. Ex.: 180000 = peso de amostragem 180.000.",
        "Exemplo": "180000"
    },

    {
        "Campo": "education",
        "Significado": "Nível de escolaridade da pessoa.",
        "Como responder": "Preschool = pré-escola; 1st-4th = 1º ao 4º ano; 5th-6th = 5º ao 6º ano; 7th-8th = 7º ao 8º ano; 9th = 9º ano; 10th = 10º ano; 11th = 11º ano; 12th = 12º ano; HS-grad = ensino médio completo; Some-college = cursou faculdade, mas não concluiu; Assoc-voc = curso técnico; Assoc-acdm = curso superior de 2 anos; Bachelors = bacharelado; Masters = mestrado; Prof-school = escola profissionalizante; Doctorate = doutorado.",
        "Exemplo": "Bachelors"
    },

    {
        "Campo": "educational-num",
        "Significado": "Representação numérica do nível de escolaridade.",
        "Como responder": "1 = Preschool; 2 = 1st-4th; 3 = 5th-6th; 4 = 7th-8th; 5 = 9th; 6 = 10th; 7 = 11th; 8 = 12th; 9 = HS-grad; 10 = Some-college; 11 = Assoc-voc; 12 = Assoc-acdm; 13 = Bachelors; 14 = Masters; 15 = Prof-school; 16 = Doctorate.",
        "Exemplo": "13 = Bachelors"
    },

    {
        "Campo": "marital-status",
        "Significado": "Estado civil da pessoa.",
        "Como responder": "Married-civ-spouse = casado(a); Divorced = divorciado(a); Never-married = nunca casado(a); Separated = separado(a); Widowed = viúvo(a); Married-spouse-absent = casado(a), mas cônjuge ausente; Married-AF-spouse = casado(a) com cônjuge das Forças Armadas.",
        "Exemplo": "Never-married"
    },

    {
        "Campo": "occupation",
        "Significado": "Tipo de ocupação profissional.",
        "Como responder": "Selecione a ocupação correspondente à profissão ou atividade principal da pessoa.",
        "Exemplo": "Exec-managerial"
    },

    {
        "Campo": "relationship",
        "Significado": "Relação da pessoa dentro do núcleo familiar.",
        "Como responder": "Wife = esposa; Own-child = filho(a); Husband = marido; Not-in-family = não pertence ao núcleo familiar; Other-relative = outro parente; Unmarried = solteiro(a).",
        "Exemplo": "Husband"
    },

    {
        "Campo": "race",
        "Significado": "Categoria racial registrada no dataset.",
        "Como responder": "White = branca; Black = preta; Asian-Pac-Islander = asiática ou das ilhas do Pacífico; Amer-Indian-Eskimo = indígena ou esquimó; Other = outra categoria.",
        "Exemplo": "White"
    },

    {
        "Campo": "gender",
        "Significado": "Gênero registrado para a pessoa.",
        "Como responder": "Male = masculino; Female = feminino.",
        "Exemplo": "Male"
    },

    {
        "Campo": "capital-gain",
        "Significado": "Valor de ganho de capital.",
        "Como responder": "Informe o valor do ganho de capital. 0 = não possui ganho de capital registrado.",
        "Exemplo": "0"
    },

    {
        "Campo": "capital-loss",
        "Significado": "Valor de perda de capital.",
        "Como responder": "Informe o valor da perda de capital. 0 = não possui perda de capital registrada.",
        "Exemplo": "0"
    },

    {
        "Campo": "hours-per-week",
        "Significado": "Quantidade de horas trabalhadas por semana.",
        "Como responder": "Informe a quantidade de horas trabalhadas por semana. Ex.: 40 = trabalha 40 horas por semana.",
        "Exemplo": "40"
    },

    {
        "Campo": "native-country",
        "Significado": "País de origem ou nascimento.",
        "Como responder": "Selecione o país correspondente à origem ou nascimento da pessoa.",
        "Exemplo": "United-States"
    }
]

st.dataframe(
    field_guide,
    use_container_width=True,
    hide_index=True
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
            "Peso amostral (FNLWGT)",
            min_value=0,
            value=180000,
            step=1000,
            help="Peso final de amostragem utilizado pelo conjunto de dados."
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

        classe_trabalho_pt = {
            "Private": "Setor privado",
            "Self-emp-not-inc": "Autônomo não incorporado",
            "Self-emp-inc": "Autônomo incorporado",
            "Federal-gov": "Governo federal",
            "Local-gov": "Governo local",
            "State-gov": "Governo estadual",
            "Without-pay": "Sem remuneração",
            "Never-worked": "Nunca trabalhou"
        }

        workclass = st.selectbox(
            "Classe de trabalho",
            list(classe_trabalho_pt.keys()),
            format_func=lambda valor: classe_trabalho_pt[valor]
        )

        escolaridade_pt = {
            "Bachelors": "Bacharelado",
            "Some-college": "Ensino superior incompleto",
            "11th": "11º ano",
            "HS-grad": "Ensino médio completo",
            "Masters": "Mestrado",
            "9th": "9º ano",
            "Assoc-acdm": "Curso superior acadêmico",
            "Assoc-voc": "Curso técnico/profissionalizante",
            "7th-8th": "7º ao 8º ano",
            "Doctorate": "Doutorado",
            "Prof-school": "Formação profissional",
            "5th-6th": "5º ao 6º ano",
            "10th": "10º ano",
            "1st-4th": "1º ao 4º ano",
            "Preschool": "Pré-escola",
            "12th": "12º ano"
        }

        education = st.selectbox(
            "Escolaridade",
            list(escolaridade_pt.keys()),
            format_func=lambda valor: escolaridade_pt[valor]
        )

        estado_civil_pt = {
            "Married-civ-spouse": "Casado(a) com cônjuge civil",
            "Divorced": "Divorciado(a)",
            "Never-married": "Nunca casou",
            "Separated": "Separado(a)",
            "Widowed": "Viúvo(a)",
            "Married-spouse-absent": "Casado(a) com cônjuge ausente",
            "Married-AF-spouse": "Casado(a) com integrante das Forças Armadas"
        }

        marital_status = st.selectbox(
            "Estado civil",
            list(estado_civil_pt.keys()),
            format_func=lambda valor: estado_civil_pt[valor]
        )

        ocupacao_pt = {
            "Tech-support": "Suporte técnico",
            "Craft-repair": "Manutenção e reparos",
            "Other-service": "Outros serviços",
            "Sales": "Vendas",
            "Exec-managerial": "Executivo/Gerencial",
            "Prof-specialty": "Profissional especializado",
            "Handlers-cleaners": "Serviços gerais e limpeza",
            "Machine-op-inspct": "Operador de máquinas e inspeção",
            "Adm-clerical": "Administrativo",
            "Farming-fishing": "Agricultura e pesca",
            "Transport-moving": "Transporte",
            "Priv-house-serv": "Serviços domésticos",
            "Protective-serv": "Serviços de proteção",
            "Armed-Forces": "Forças Armadas"
        }

        occupation = st.selectbox(
            "Ocupação",
            list(ocupacao_pt.keys()),
            format_func=lambda valor: ocupacao_pt[valor]
        )

    with col2:

        relacionamento_pt = {
            "Wife": "Esposa",
            "Own-child": "Filho(a)",
            "Husband": "Marido",
            "Not-in-family": "Não pertence à família",
            "Other-relative": "Outro parente",
            "Unmarried": "Não casado(a)"
        }

        relationship = st.selectbox(
            "Relacionamento familiar",
            list(relacionamento_pt.keys()),
            format_func=lambda valor: relacionamento_pt[valor]
        )

        etnia_pt = {
            "White": "Branca",
            "Black": "Negra",
            "Asian-Pac-Islander": "Asiática ou das Ilhas do Pacífico",
            "Amer-Indian-Eskimo": "Indígena americana ou nativa do Alasca",
            "Other": "Outra"
        }

        race = st.selectbox(
            "Etnia",
            list(etnia_pt.keys()),
            format_func=lambda valor: etnia_pt[valor]
        )

        genero_pt = {
            "Male": "Masculino",
            "Female": "Feminino"
        }

        gender = st.selectbox(
            "Gênero",
            list(genero_pt.keys()),
            format_func=lambda valor: genero_pt[valor]
        )

        pais_pt = {
            "United-States": "Estados Unidos",
            "Cambodia": "Camboja",
            "England": "Inglaterra",
            "Puerto-Rico": "Porto Rico",
            "Canada": "Canadá",
            "Germany": "Alemanha",
            "Outlying-US(Guam-USVI-etc)": "Territórios dos EUA (Guam, Ilhas Virgens etc.)",
            "India": "Índia",
            "Japan": "Japão",
            "Greece": "Grécia",
            "South": "Coreia do Sul",
            "China": "China",
            "Cuba": "Cuba",
            "Iran": "Irã",
            "Honduras": "Honduras",
            "Philippines": "Filipinas",
            "Italy": "Itália",
            "Poland": "Polônia",
            "Jamaica": "Jamaica",
            "Vietnam": "Vietnã",
            "Mexico": "México",
            "Portugal": "Portugal",
            "Ireland": "Irlanda",
            "France": "França",
            "Dominican-Republic": "República Dominicana",
            "Laos": "Laos",
            "Ecuador": "Equador",
            "Taiwan": "Taiwan",
            "Haiti": "Haiti",
            "Columbia": "Colômbia",
            "Hungary": "Hungria",
            "Guatemala": "Guatemala",
            "Nicaragua": "Nicarágua",
            "Scotland": "Escócia",
            "Thailand": "Tailândia",
            "Yugoslavia": "Iugoslávia",
            "El-Salvador": "El Salvador",
            "Trinadad&Tobago": "Trinidad e Tobago",
            "Peru": "Peru",
            "Hong": "Hong Kong",
            "Holand-Netherlands": "Países Baixos"
        }

        native_country = st.selectbox(
            "País de origem",
            list(pais_pt.keys()),
            format_func=lambda valor: pais_pt[valor]
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
                "Probabilidade de renda anual acima de US$ 50 mil",
                f"{probability * 100:.2f}%"
            )


        with col2:

            if predicted_class == 1:

                st.metric(
                    "Classe prevista",
                    "Acima de US$ 50 mil"
                )

            else:

                st.metric(
                    "Classe prevista",
                    "Até US$ 50 mil"
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
                A rede neural classificou esta pessoa como tendo
                **renda anual superior a US$ 50 mil**, com probabilidade
                estimada de **{probability * 100:.2f}%**.
                """
            )

        else:

            st.info(
                f"""
                A rede neural classificou esta pessoa como tendo
                **renda anual igual ou inferior a US$ 50 mil**.
                A probabilidade estimada para uma renda acima de
                US$ 50 mil é de **{probability * 100:.2f}%**.
                """
            )


        # ====================================================
        # DISTRIBUIÇÃO DAS CLASSES
        # ====================================================

        st.markdown(
            "### Distribuição da previsão"
        )

        chart_data = {

            "Até US$ 50 mil": 1 - probability,

            "Acima de US$ 50 mil": probability
        }

        st.bar_chart(
            chart_data
        )

        # ====================================================
        # SHAP
        # ====================================================

        st.markdown(
            "### 🔎 Explicabilidade do modelo"
        )

        st.write(
            """
            O gráfico abaixo apresenta a contribuição das
            características utilizadas pela rede neural nas
            previsões realizadas.
            """
        )

        try:
            X_train, X_test = load_explanation_data()

            X_background = X_train[:50]

            X_explain = X_test[:100]

            feature_names = preprocessing[
                "feature_columns"
            ]

            fig = generate_shap_summary(
                model=model,
                X_background=X_background,
                X_explain=X_explain,
                feature_names=feature_names
            )

            col_esquerda, col_grafico, col_direita = st.columns([1, 2, 1])

            with col_grafico:
                st.pyplot(fig)

        except Exception as error:

            st.warning(
                "Não foi possível gerar a análise SHAP."
            )

            st.exception(error)
            
        # ====================================================
        # DETALHES TÉCNICOS
        # ====================================================

        with st.expander(
            "🔧 Ver detalhes técnicos"
        ):

            st.write(
                "Entrada após o pré-processamento:"
            )

            st.write(
                f"{X.shape[1]} atributos"
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
    "Modelo MLP previamente treinado com o conjunto de dados Adult Income. "
    "A aplicação realiza apenas previsões utilizando o modelo treinado."
)