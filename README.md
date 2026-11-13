# rag_agent_groq

Agente em **Streamlit** que analisa a compatibilidade entre um **currículo (PDF)** e uma **descrição de vaga** colada pelo usuário, usando **RAG** (Retrieval-Augmented Generation).

## Como funciona

1. O currículo em PDF é carregado e lido com `PyPDFLoader`.
2. O texto é dividido em chunks (`RecursiveCharacterTextSplitter`, chunk de 1000 e overlap de 200).
3. Cada chunk vira um embedding (`OllamaEmbeddings`, modelo `qwen3-embedding:0.6b`) e é indexado no **FAISS**.
4. Dada a vaga, o agente recupera os trechos mais relevantes do currículo e envia ao LLM (`ChatGroq`, modelo `openai/gpt-oss-120b`), que gera a análise de aderência.

## Stack

- Python, Streamlit
- LangChain (`langchain`, `langchain-community`, `langchain-groq`, `langchain-ollama`, `langchain-text-splitters`)
- FAISS (`faiss-cpu`) como vector store
- Groq (LLM) e Ollama (embeddings locais)
- `pypdf` e `python-dotenv`

## Como rodar

Pré-requisitos: Python 3.10+, [Ollama](https://ollama.com) instalado e uma chave de API da [Groq](https://console.groq.com).

```bash
git clone https://github.com/kaikycruz/rag_agent_groq.git
cd rag_agent_groq

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

ollama pull qwen3-embedding:0.6b
```

Crie um arquivo `.env` na raiz com a chave da Groq:

```
GROQ_API_KEY=sua_chave_aqui
```

Inicie o app:

```bash
streamlit run agent.py
```

## Próximos passos

- Extrair os requisitos da vaga automaticamente e verificar o currículo requisito por requisito.
