from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# ---------- Lógica do RAG ----------
def carregar_pdf():  # lê o PDF e extrai o texto
    pdf_loader = PyPDFLoader("Kaiky_Cruz_AI_Engineer.pdf")
    return pdf_loader.load()


def separar_em_blocos(texto_pdf):
    separador = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return separador.split_documents(texto_pdf)


def criar_banco_vetores(lista_blocos):
    ferramenta_embedding = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    return FAISS.from_documents(lista_blocos, ferramenta_embedding)


def formatar_trechos(docs):  # junta só o texto dos trechos recuperados
    return "\n\n".join(doc.page_content for doc in docs)


# ---------- Lógica do agente ----------
PROMPT = """Você é um recrutador técnico sênior, especialista em currículos para a área de tecnologia no Brasil.

Sua tarefa é avaliar a compatibilidade entre o currículo de um candidato e uma vaga. Use SOMENTE os trechos do currículo e a descrição da vaga fornecidos abaixo. Seja direto e honesto: não elogie por educação, e justifique cada ponto com o que está escrito nos trechos.

Regras:
- Nunca invente experiências, tecnologias, cargos, anos de experiência ou números.
- Os trechos do currículo são parciais. Se um requisito não aparece neles, escreva "não encontrei nos trechos analisados" e não conclua que o candidato não tem essa habilidade.
- Só classifique um requisito como "Atendido" se houver um trecho que o comprove. Cite esse trecho de forma curta.
- Experiência em tecnologia parecida, mas não igual, conta como "Parcial" e deve ser explicada.
- Não faça suposições sobre o que o candidato "provavelmente sabe".
- Responda em português do Brasil.

Trechos do currículo:
{contexto}

Descrição da vaga:
{vaga}

Entregue a resposta neste formato:

1. Resumo da vaga: em 2 ou 3 linhas, o que a empresa busca (cargo, senioridade e principais tecnologias).
2. Requisitos atendidos: lista, cada item com o trecho do currículo que comprova.
3. Requisitos parcialmente atendidos: lista, explicando o que há de parecido e o que falta.
4. Requisitos ausentes ou não encontrados: lista, indicando se pode ser limitação dos trechos analisados.
5. Compatibilidade estimada: Baixa, Média ou Alta, com uma justificativa de uma frase. Lembre que é uma estimativa.
6. Três ações para aumentar a aderência: ajustes no currículo (o que destacar, o que reescrever) sem inventar fatos.

Resposta:"""


def criar_chain_agente(banco_vetores):
    prompt_template = ChatPromptTemplate.from_template(PROMPT)
    buscador_contexto = banco_vetores.as_retriever(search_kwargs={"k": 6})
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

    chain = (
        {
            "contexto": buscador_contexto | formatar_trechos,
            "vaga": RunnablePassthrough(),
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return chain


@st.cache_resource  # prepara tudo uma vez só, e não a cada interação
def carregar_chain():
    texto_pdf = carregar_pdf()
    lista_blocos = separar_em_blocos(texto_pdf)
    banco_vetores = criar_banco_vetores(lista_blocos)
    return criar_chain_agente(banco_vetores)


# ---------- Interface ----------
st.title("Assistente Analista de Currículo x Vaga")

chain = carregar_chain()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Cole a descrição da vaga (responsabilidades e requisitos)...")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analisando..."):
            answer = chain.invoke(prompt)
            st.write(answer)

    st.session_state["messages"].append({"role": "assistant", "content": answer})