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
    
    def quiz_system_prompt(self):
        return (

        "You are a quiz generator.\n"
        "Your task is to ask the user a single quiz question based ONLY on the provided reference context.\n\n"

        "Quiz rules:\n"
        "- Use ONLY the reference context. Do not use outside knowledge.\n"
        "- Ask exactly ONE clear and specific question.\n"
        "- The question must be answerable from the context.\n"
        "- Do NOT include the answer or hints.\n"
        "- Do NOT ask multiple questions at once.\n"
        "- Avoid ambiguous or opinion-based questions.\n\n"

        "Question style:\n"
        "- Prefer factual or conceptual questions.\n"
        "- Keep the wording concise and clear.\n"
        "- Do not reference the context explicitly.\n\n"

        "Output format (must be followed exactly):\n"
        "Question: <quiz question>\n"

        )
    
    def quiz_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", self.quiz_system_prompt()),
            (
            "human",
            "Reference context:\n"
            "{context}\n\n"
            "Generate one quiz question based on the reference context."
        )
        ])
    
    def evaluation_system_prompt(self):
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
            ("system", self.evaluation_system_prompt()),
            (
                "human",
                "Context: {context}\n\n"
                "Question: {question}\n\n"
                "Student's Answer: {answer}\n\n"
                "Evaluate the answer and respond in this EXACT JSON format:\n"
                "{{\"correct\": true/false, \"feedback\": \"your detailed feedback here\", \"score\": 0-100}}\n\n"
                "Be strict but fair in your evaluation."
            )
        ])
    def multi_question_prompt(self, full_context, num_questions):
        return  ChatPromptTemplate.from_messages([
                ("system", 
                "You are a quiz generator. Generate diverse, non-repetitive quiz questions "
                "based on the provided context. Each question should focus on different "
                "aspects or facts from the context."),
                ("human",
                f"Reference context:\n{full_context}\n\n"
                f"Generate exactly {num_questions} DIFFERENT quiz questions based on this context.\n\n"
                "Rules:\n"
                "- Each question must be unique and focus on different information\n"
                "- Questions should test different concepts/facts\n"
                "- Vary the difficulty and question types\n"
                "- Do NOT repeat or rephrase the same question\n\n"
                "Format each question on a new line starting with 'Q#: ' like this:\n"
                "Q1: [first question]\n"
                "Q2: [second question]\n"
                "Q3: [third question]\n")
            ])
            

        