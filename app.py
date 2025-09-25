import streamlit as st
import numpy as np
import zipfile
import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Configuração da Chave de API do Google ---
# O Streamlit Cloud armazena a chave de API de forma segura.
# Para rodar localmente, crie um arquivo .streamlit/secrets.toml
# e adicione a sua chave:
# [google_ai]
# google_api_key = "SUA_CHAVE_AQUI"
try:
    google_api_key = st.secrets["google_ai"]["google_api_key"]
except KeyError:
    st.error("Chave de API do Google não encontrada. Certifique-se de adicioná-la nos 'Secrets' da sua aplicação.")
    st.stop()

# --- Definição das Ferramentas (Tools) ---
@tool
def show_descriptive_stats(df):
    """
    Gera estatísticas descritivas para todas as colunas de um DataFrame.
    Retorna um dicionário com o resumo estatístico.
    """
    stats = df.describe(include='all')
    return {"status": "success", "data": stats.to_markdown(), "message": "Estatísticas descritivas geradas."}

@tool
def generate_histogram(df, column: str):
    """
    Gera um histograma para uma coluna numérica específica do DataFrame.
    A entrada deve ser o nome da coluna.
    """
    if column not in df.columns:
        return {"status": "error", "message": f"Erro: A coluna '{column}' não existe no DataFrame."}
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"status": "error", "message": f"Erro: A coluna '{column}' não é numérica. Por favor, forneça uma coluna numérica para gerar um histograma."}
    fig, ax = plt.subplots()
    df[column].hist(ax=ax)
    ax.set_title(f'Distribuição de {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequência')
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return {"status": "success", "image": buf, "message": f"Histograma para a coluna '{column}' gerado."}

@tool
def generate_correlation_heatmap(df):
    """
    Calcula a matriz de correlação entre as variáveis numéricas do DataFrame
    e gera um mapa de calor (heatmap) para visualização.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) < 2:
        return {"status": "error", "message": "Erro: O DataFrame não tem colunas numéricas suficientes para calcular a correlação."}
    correlation_matrix = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Mapa de Calor da Matriz de Correlação')
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return {"status": "success", "image": buf, "message": "Mapa de calor da correlação gerado."}

@tool
def generate_scatter_plot(df, x_col: str, y_col: str):
    """
    Gera um gráfico de dispersão (scatter plot) para visualizar a relação entre duas colunas numéricas.
    As entradas devem ser os nomes das colunas para os eixos X e Y.
    """
    if x_col not in df.columns or y_col not in df.columns:
        return {"status": "error", "message": f"Erro: Uma ou ambas as colunas ('{x_col}', '{y_col}') não existem no DataFrame."}
    fig, ax = plt.subplots()
    df.plot.scatter(x=x_col, y=y_col, ax=ax)
    ax.set_title(f'Gráfico de Dispersão: {x_col} vs {y_col}')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return {"status": "success", "image": buf, "message": f"Gráfico de dispersão para '{x_col}' vs '{y_col}' gerado."}

@tool
def detect_outliers_isolation_forest(df):
    """
    Detecta anomalias (outliers) no DataFrame usando o algoritmo Isolation Forest.
    A análise é aplicada às colunas V1 a V28, 'Time' e 'Amount' do dataset de fraudes.
    Retorna o número de anomalias detectadas e uma amostra dos outliers.
    """
    try:
        feature_cols = [col for col in df.columns if col.startswith('V')] + ['Time', 'Amount']
        df_features = df[feature_cols]
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_features)
        model = IsolationForest(contamination=0.01, random_state=42)
        df['anomaly_score'] = model.fit_predict(df_scaled)
        outliers = df[df['anomaly_score'] == -1]
        message = f"O algoritmo Isolation Forest detectou {len(outliers)} transações atípicas (outliers)."
        if not outliers.empty:
            message += "\nAmostra das transações detectadas como anomalias:\n" + outliers.head().to_markdown()
        return {"status": "success", "message": message}
    except Exception as e:
        return {"status": "error", "message": f"Erro ao detectar anomalias: {e}"}

@tool
def find_clusters_kmeans(df, n_clusters: int):
    """
    Realiza agrupamento (clustering) nos dados usando o algoritmo K-Means.
    A análise é aplicada às colunas V1 a V28, 'Time' e 'Amount' do dataset de fraudes.
    A entrada deve ser o número de clusters desejado.
    Retorna uma descrição dos clusters encontrados.
    """
    try:
        feature_cols = [col for col in df.columns if col.startswith('V')] + ['Time', 'Amount']
        df_features = df[feature_cols]
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        df['cluster'] = kmeans.fit_predict(df_scaled)
        cluster_summary = df.groupby('cluster').agg({
            'Amount': ['mean', 'min', 'max'],
            'Time': ['min', 'max']
        })
        message = f"O agrupamento K-Means com {n_clusters} clusters foi concluído."
        message += "\nCaracterísticas dos Clusters:\n" + cluster_summary.to_markdown()
        return {"status": "success", "message": message}
    except Exception as e:
        return {"status": "error", "message": f"Erro ao realizar o agrupamento com K-Means: {e}"}

# List of all tools
tools = [
    show_descriptive_stats,
    generate_histogram,
    generate_correlation_heatmap,
    generate_scatter_plot,
    detect_outliers_isolation_forest,
    find_clusters_kmeans
]

# System prompt
system_prompt = """
Você é um agente de análise de dados extremamente experiente e um programador Python de primeira linha. Sua tarefa é atuar como um analista de dados especialista e interagir com o usuário sobre um arquivo CSV.
Você tem acesso a um conjunto de ferramentas poderosas para realizar Análise Exploratória de Dados (EDA), geração de gráficos e detecção de padrões.
Siga estas instruções rigorosamente:
1. **Entenda a intenção:** O objetivo principal da sua interação é responder às perguntas do usuário sobre o dataset.
2. **Planeje e Execute:** Use as ferramentas disponíveis para executar a análise necessária. Você não deve escrever código Python diretamente na sua resposta final, mas sim usar as ferramentas que encapsulam essa lógica.
3. **Analise os Resultados:** Após executar uma ferramenta, analise a saída (ex: estatísticas, gráficos) para procurar por padrões, anomalias e relações.
4. **Sintetize as Conclusões:** O seu maior valor é a sua capacidade de ir além da mera execução de comandos. Sintetize todos os achados em uma conclusão narrativa e coerente.
5. **Gere Insights:** Ao invés de apenas listar fatos, explique o que eles significam no contexto do problema (ex: o que a assimetria de uma variável ou a alta correlação entre duas variáveis sugere).
6. **Memória:** Use o histórico da conversa para construir suas conclusões, referenciando achados de análises anteriores quando apropriado, para demonstrar raciocínio coeso.
Você só tem acesso às ferramentas que lhe foram fornecidas. Se a pergunta do usuário não puder ser respondida com suas ferramentas, informe-o de forma educada e sugira o que você pode fazer.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# --- Inicialização da Aplicação Streamlit ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'agent_executor' not in st.session_state:
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key, temperature=0.0)
    st.session_state.memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
    agent = create_tool_calling_agent(llm, tools, prompt)
    st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=st.session_state.memory)

st.title("Agente de Análise de Dados com IA")
st.markdown("""
    Bem-vindo ao seu assistente de Análise Exploratória de Dados (EDA) para a atividade extra do curso I2A2.
    Este agente é especialista em Python e LangChain, pronto para analisar seus dados.
    1.  Faça o upload do seu arquivo ZIP.
    2.  Faça sua pergunta! O agente irá usar suas ferramentas para carregar os dados, gerar gráficos e responder com conclusões.
    Exemplos de perguntas:
    * `Quais as estatísticas descritivas dos dados?`
    * `Gere um histograma para a coluna 'Amount'.`
    * `Gere um mapa de calor da correlação.`
    * `Use K-Means com 5 clusters e me diga as conclusões.`
    * `Faça uma detecção de anomalias com Isolation Forest.`
""")

# --- Manipulação de Arquivo e Sessão ---
uploaded_zip_file = st.file_uploader(
    "Faça o upload do arquivo ZIP com os dados de fraude de cartão de crédito",
    type="zip",
    key="zip_uploader"
)

# @st.cache_data(show_spinner=False)
def get_data_from_zip(uploaded_file):
    st.info("Descompactando o arquivo e carregando os dados...")
    try:
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
            if not csv_files:
                return {"status": "error", "message": "Nenhum arquivo CSV encontrado no arquivo ZIP."}
            file_name = csv_files[0]
            with zip_ref.open(file_name) as csv_file:
                df = pd.read_csv(io.BytesIO(csv_file.read()))
                return {"status": "success", "df": df, "message": f"Dados carregados com sucesso do arquivo '{file_name}'."}
    except zipfile.BadZipFile:
        return {"status": "error", "message": "O arquivo enviado não é um arquivo ZIP válido."}
    except Exception as e:
        return {"status": "error", "message": f"Ocorreu um erro ao processar o arquivo ZIP: {e}"}

if uploaded_zip_file and st.session_state.df is None:
    load_result = get_data_from_zip(uploaded_zip_file)
    if load_result["status"] == "success":
        st.session_state.df = load_result["df"]
        st.success("Dados carregados com sucesso! Aqui está uma pré-visualização:")
        st.dataframe(st.session_state.df.head())
    else:
        st.error(load_result["message"])
