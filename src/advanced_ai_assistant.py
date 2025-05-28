import os
from dotenv import load_dotenv

from langchain.schema import SystemMessage
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain


class AdvancedLangChainAgent:
    """
    Classe avançada para criação de um agente inteligente usando LangChain e GPT-4 Turbo.
    """

    def __init__(self, url: str, model_name: str = "gpt-4-turbo", temperature: float = 0.2):
        """
        Inicializa o agente com URL de conteúdo, modelo de linguagem e configuração de temperatura.

        Args:
            url (str): URL para carregar dados externos.
            model_name (str): Nome do modelo GPT para geração de texto.
            temperature (float): Configuração de temperatura para controle da aleatoriedade das respostas.
        """
        load_dotenv()
        user_agent = os.getenv("USER_AGENT") or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        os.environ["USER_AGENT"] = user_agent

        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        loader = WebBaseLoader(url)
        loader.requests_kwargs = {"headers": {"User-Agent": user_agent}}
        documents = loader.load()
        if not documents:
            raise ValueError("Nenhum documento foi carregado. Verifique a URL ou o User-Agent.")

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever()

        self.conversational_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
        )

        self.tools = [
            Tool(
                name="AI Knowledge Base",
                func=self.conversational_chain.run,
                description="Útil para responder perguntas complexas sobre inteligência artificial.",
            ),
        ]

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="Você é um assistente especialista em Inteligência Artificial e Machine Learning. Responda sempre com detalhes técnicos."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            memory=self.memory,
            prompt=self.prompt
        )

    def run_query(self, query: str) -> str:
        """
        Executa uma consulta ao agente inteligente.

        Args:
            query (str): A consulta ou pergunta para o agente responder.

        Returns:
            str: Resposta do agente baseado na consulta.
        """
        return self.agent.run(query)


# Exemplo de execução avançada
if __name__ == "__main__":
    agent = AdvancedLangChainAgent(url="https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial")
    while True:
        query = input("Faça uma pergunta avançada sobre IA (ou 'sair'): ")
        if query.lower() == 'sair':
            break
        resposta = agent.run_query(query)
        print(f"\nResposta avançada: {resposta}\n")
