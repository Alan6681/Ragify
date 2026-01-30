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
        self.prompts.quiz_prompt()
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
    
    
    def quiz_generator(self, topic=None, num_questions=1):
        """
        Generate multiple diverse quiz questions (iOS-compatible)
        """
        import random
        
        # Retrieve documents
        if topic:
            docs = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": min(20, num_questions * 3),
                    "fetch_k": min(50, num_questions * 5),
                    "lambda_mult": 0.4
                }
            ).invoke(topic)
        else:
            docs = self.vectorstore.similarity_search("", k=min(20, num_questions * 3))
        
        # Shuffle for variety
        random.shuffle(docs)
        
        questions_and_contexts = []
        docs_per_question = max(2, len(docs) // num_questions)
        
        # Generate questions one at a time (more reliable)
        for i in range(num_questions):
            start_idx = i * docs_per_question
            end_idx = start_idx + docs_per_question
            question_docs = docs[start_idx:end_idx]
            
            if not question_docs:
                continue
            
            context = self.format_docs(question_docs)
            
            # Simple prompt without complex formatting
            response = self.quiz_chain_runnable.invoke({
                "context": context
            })
            
            # Just extract the text directly (no regex parsing)
            question_text = response.content.strip()
            
            # Remove "Question:" prefix if present
            if question_text.lower().startswith("question:"):
                question_text = question_text[9:].strip()
            
            questions_and_contexts.append((question_text, context))
        
        return questions_and_contexts

    def _fallback_quiz_generation(self, topic, num_questions, docs=None):
        """Fallback: generate questions individually"""
        import random
        
        # If docs not provided, retrieve them
        if docs is None:
            if topic:
                docs = self.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": num_questions * 4,
                        "fetch_k": num_questions * 10,
                        "lambda_mult": 0.3
                    }
                ).invoke(topic)
            else:
                docs = self.vectorstore.similarity_search("", k=num_questions * 4)
        
        random.shuffle(docs)
        questions_and_contexts = []
        docs_per_question = max(2, len(docs) // num_questions)
        
        for i in range(num_questions):
            start_idx = i * docs_per_question
            end_idx = min(start_idx + docs_per_question, len(docs))
            question_docs = docs[start_idx:end_idx]
            
            if question_docs:
                context = self.format_docs(question_docs)
                response = self.quiz_chain_runnable.invoke({"context": context})
                questions_and_contexts.append((response.content, context))
        
        return questions_and_contexts

    def evaluate(self, context, question, user_answer):
        response = self.evaluation_chain_runnable.invoke({
            "context": context,
            "question": question,
            "answer": user_answer
        })
        return response.content