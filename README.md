# Yooga · Análise de Consumo

Interface de chat para consultar dados de consumo de clientes em restaurantes usando linguagem natural. A IA traduz perguntas em português para código Pandas — os cálculos são exatos, não aproximações.

## Como funciona

1. Importe um arquivo `.txt` tabulado por tab com os dados dos pedidos
2. Faça perguntas em português no chat
3. A LLM gera código Pandas, executa e exibe o resultado (tabela, número ou gráfico)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Edite o `.env` com sua chave da OpenAI:

```
OPENAI_API_KEY=sk-sua-chave-aqui
```

## Rodando

```bash
streamlit run app.py
```

Acesse em `http://localhost:8501`.

## Estrutura

```
app.py            → Interface Streamlit (chat + sidebar com métricas)
llm.py            → Abstração da LLM (OpenAI GPT-4o por padrão)
requirements.txt  → Dependências
.env.example      → Template de variáveis de ambiente
```

## Deploy

O projeto roda no [Streamlit Community Cloud](https://share.streamlit.io). Basta conectar o repositório e configurar a `OPENAI_API_KEY` nos secrets.
