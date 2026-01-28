from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from prompts.prompts import Prompts
from services.ai_service import AIService
from langchain_core.messages import HumanMessage, AIMessage
from operator import itemgetter


class RagChain:
    def __init__(self, files):
        self.prompts = Prompts()
        self.ai_service = AIService(files)

        self.llm = self.ai_service.load_llm()
        self.retriever = self.ai_service.create_retriever()
        self.vectorstore = self.ai_service.create_vectorstore()

        self.history_aware_retriever = self._create_history_aware_retriever()
        self.qa_chain_runnable = self._create_qa_chain()
        self.evaluation_chain_runnable = self._create_evaluation_chain()
        self.quiz_chain_runnable = self._create_quiz_chain()

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    
    def _create_history_aware_retriever(self):
        return (
            {
                "input": itemgetter("input"), 
                "chat_history": itemgetter("chat_history"),  
            }
            | self.prompts.contextualize_q_prompt()
            | self.llm
            | RunnableLambda(lambda x: x.content)
            | self.retriever
        )

    def _create_qa_chain(self):
        return (
            {
                "context": self.history_aware_retriever | RunnableLambda(self.format_docs), 
                "input": itemgetter("input"),  
                "chat_history": itemgetter("chat_history"),  
            }
            | self.prompts.qa_prompt()
            | self.llm
        )
        
    def _create_quiz_chain(self):
        return (
             {
        "context": RunnablePassthrough() | self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "lambda_mult": 0.5}
        ) | RunnableLambda(self.format_docs)
    }
    | self.prompts.quiz_prompt()
    | self.llm
        )

    def _create_evaluation_chain(self):
        return (
            self.prompts.evaluation_prompt()
            | self.llm
        )

    def ask(self, question, chat_history):
        response = self.qa_chain_runnable.invoke({
            "input": question,
            "chat_history": chat_history
        })
        return response.content

    def evaluate(self, context, question, user_answer):
        response = self.evaluation_chain_runnable.invoke({
            "context": context,
            "question": question,
            "answer": user_answer
        })
        return response.content