from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class Prompts:
    def __init__(self):
        pass

    def contextualize_q_system_prompt(self):
        return (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
        )
    
    def contextualize_q_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", self.contextualize_q_system_prompt()),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
    

    def qa_system_prompt(self):
        return (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"

        )
    
    def qa_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", self.qa_system_prompt()),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
    
    def evalution_system_prompt(self):
        return (
        "You are an impartial evaluator.\n"
        "Your task is to assess a user's answer based ONLY on the provided reference context.\n\n"

        "Evaluation rules:\n"
        "- Use ONLY the reference context. Do not use outside knowledge.\n"
        "- If the user's answer matches the reference context in meaning, mark it as CORRECT.\n"
        "- If the answer is partially correct but missing key points, mark it as PARTIALLY_CORRECT.\n"
        "- If the answer is incorrect or unsupported by the context, mark it as INCORRECT.\n"
        "- Be strict but fair.\n\n"

        "Output format (must be followed exactly):\n"
        "Verdict: <CORRECT | PARTIALLY_CORRECT | INCORRECT>\n"
        "Score: <1 | 0.5 | 0>\n"
        "Feedback: <brief explanation based on the context>\n"
        )
    
    def evaluation_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", self.evalution_system_prompt()),
            ("human", 
              "Reference Context:\n{context}\n\n"
              "Question:\n{question}\n\n"
              "User Answer:\n{answer}")
        ])

    