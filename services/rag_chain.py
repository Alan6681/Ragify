from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from prompts.prompts import Prompts
from services.ai_service import AIService


class RagChain:
    def __init__(self, files):
        self.prompts = Prompts()
        self.ai_service = AIService(files)

        self.llm = self.ai_service.load_llm()
        self.retriever = self.ai_service.create_retriever()

        self.history_aware_retriever = self._create_history_aware_retriever()
        self.qa_chain_runnable = self._create_qa_chain()
        self.evaluation_chain_runnable = self._create_evaluation_chain()

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    
    def _create_history_aware_retriever(self):
        return (
            {
                "input": RunnablePassthrough(),
                "chat_history": RunnablePassthrough(),
            }
            | self.prompts.contextualize_q_prompt()
            | self.llm
            | RunnableLambda(lambda x: x.content)
            | self.retriever
        )

    def _create_qa_chain(self):
        return (
            {
                "context": self.history_aware_retriever
                | RunnableLambda(self.format_docs),
                "input": RunnablePassthrough(),
                "chat_history": RunnablePassthrough(),
            }
            | self.prompts.qa_prompt()
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
