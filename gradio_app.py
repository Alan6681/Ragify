import gradio as gr
from services.rag_chain import RagChain
from langchain_core.messages import HumanMessage, AIMessage
import time
import json

# Global state
class AppState:
    def __init__(self):
        self.rag_chain = None
        self.chat_history = []
        self.quiz_questions = []
        self.current_question_index = 0
        self.quiz_answers = []
        self.quiz_scores = []
        self.quiz_evaluations = []
        self.answer_times = []
        self.question_start_time = None

state = AppState()

# ==================== Functions ====================
def process_pdfs(files):
    if not files:
        return "⚠️ Please upload at least one PDF file"
    
    try:
        uploaded_files = [open(f.name, 'rb') for f in files]
        state.rag_chain = RagChain(uploaded_files)
        state.chat_history = []
        
        for f in uploaded_files:
            f.close()
        
        return f"✅ Successfully processed {len(files)} PDF file(s)!"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def chat_with_pdf(message, history):
    if not state.rag_chain:
        return history + [["❌ Please upload a PDF first!", None]]
    
    if not message or not message.strip():
        return history
    
    try:
        state.chat_history.append(HumanMessage(content=message))
        response = state.rag_chain.ask(question=message, chat_history=state.chat_history)
        state.chat_history.append(AIMessage(content=response))
        
        return history + [[message, response]]
    except Exception as e:
        return history + [[message, f"❌ Error: {str(e)}"]]

def generate_quiz(topic, num_questions):
    if not state.rag_chain:
        return "❌ Please upload a PDF first!", gr.update(visible=False)
    
    if not topic.strip():
        return "⚠️ Please enter a topic", gr.update(visible=False)
    
    try:
        questions_and_contexts = state.rag_chain.quiz_generator(
            topic=topic.strip(),
            num_questions=int(num_questions)
        )
        
        if not questions_and_contexts:
            return "❌ Failed to generate questions", gr.update(visible=False)
        
        state.quiz_questions = questions_and_contexts
        state.current_question_index = 0
        state.quiz_answers = []
        state.quiz_scores = []
        state.quiz_evaluations = []
        state.answer_times = []
        state.question_start_time = time.time()
        
        return (
            f"✅ Generated {len(questions_and_contexts)} questions!",
            gr.update(visible=True)
        )
    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update(visible=False)

def get_current_question():
    if not state.quiz_questions or state.current_question_index >= len(state.quiz_questions):
        return "No question available", "0/0"
    
    question = state.quiz_questions[state.current_question_index][0]
    progress = f"{state.current_question_index + 1}/{len(state.quiz_questions)}"
    return question, progress

def submit_answer(answer):
    if not state.quiz_questions:
        return "Please generate a quiz first!", gr.update(), gr.update()
    
    if not answer.strip():
        return "⚠️ Please enter an answer", gr.update(), gr.update()
    
    idx = state.current_question_index
    
    if idx >= len(state.quiz_questions):
        return "Quiz completed!", gr.update(), gr.update()
    
    try:
        time_taken = time.time() - state.question_start_time
        question, context = state.quiz_questions[idx]
        
        evaluation = state.rag_chain.evaluate(
            context=context,
            question=question,
            user_answer=answer.strip()
        )
        
        try:
            eval_data = json.loads(evaluation)
            is_correct = eval_data.get("correct", False)
            feedback_text = eval_data.get("feedback", evaluation)
        except:
            is_correct = any(word in evaluation.lower() for word in ["correct", "right", "accurate"])
            feedback_text = evaluation
        
        state.quiz_answers.append(answer.strip())
        state.quiz_scores.append(is_correct)
        state.quiz_evaluations.append(feedback_text)
        state.answer_times.append(time_taken)
        
        status = "✅ Correct!" if is_correct else "❌ Incorrect"
        feedback = f"**{status}**\n\n{feedback_text}\n\n⏱️ {time_taken:.1f}s"
        
        state.current_question_index += 1
        state.question_start_time = time.time()
        
        next_q, progress = get_current_question()
        
        if state.current_question_index >= len(state.quiz_questions):
            return feedback, "🎉 Quiz Complete!", progress
        
        return feedback, next_q, progress
    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update(), gr.update()

def get_results():
    if not state.quiz_scores:
        return "Complete a quiz to see results!"
    
    correct = sum(state.quiz_scores)
    total = len(state.quiz_scores)
    accuracy = (correct / total * 100) if total > 0 else 0
    avg_time = sum(state.answer_times) / len(state.answer_times) if state.answer_times else 0
    
    rating = "🌟 Excellent!" if accuracy >= 90 else "👍 Good!" if accuracy >= 70 else "📚 Keep practicing!"
    
    return f"""
### 📊 Results

**Score:** {correct}/{total} ({accuracy:.1f}%)  
**Average Time:** {avg_time:.1f}s per question  
**Rating:** {rating}
"""

# ==================== Beautiful UI ====================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.gradio-container {
    max-width: 1400px !important;
}

/* Beautiful gradient background */
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* Card styling */
.gr-box {
    background: white;
    border-radius: 16px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    border: none !important;
}

/* Button styling */
.gr-button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    border: none !important;
}

.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4) !important;
}

.gr-button-secondary {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
    color: white !important;
}

/* Input styling */
.gr-input, .gr-textbox {
    border-radius: 12px !important;
    border: 2px solid #e0e0e0 !important;
}

.gr-input:focus, .gr-textbox:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

/* Chat styling */
.message-wrap {
    padding: 16px !important;
    border-radius: 16px !important;
    margin: 8px 0 !important;
}

.message.user {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
}

.message.bot {
    background: #f7f7f7 !important;
}

/* Tab styling */
.tab-nav button {
    border-radius: 12px 12px 0 0 !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
}

/* Progress bar */
.progress-bar {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
}

/* Header */
h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 3em;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Ragify") as demo:
    
    # Header
    with gr.Row():
        gr.Markdown("""
        # 📕 Ragify
        ### AI-Powered Document Q&A & Quiz Generator
        """)
    
    # Upload Section
    with gr.Row():
        with gr.Column(scale=2):
            pdf_upload = gr.File(
                label="📤 Upload Your PDFs",
                file_count="multiple",
                file_types=[".pdf"],
                height=120
            )
        with gr.Column(scale=1):
            upload_status = gr.Textbox(label="Status", interactive=False, lines=3)
    
    upload_btn = gr.Button("Process Documents", variant="primary", size="lg", scale=1)
    
    gr.Markdown("---")
    
    # Main Content
    with gr.Tabs() as tabs:
        
        # Chat Tab
        with gr.Tab("💬 Chat"):
            chatbot = gr.Chatbot(
                height=500,
                show_label=False,
                avatar_images=(
                    "https://api.dicebear.com/7.x/avataaars/svg?seed=User",
                    "https://api.dicebear.com/7.x/bottts/svg?seed=AI"
                ),
                bubble_full_width=False
            )
            
            with gr.Row():
                chat_msg = gr.Textbox(
                    placeholder="Ask me anything about your documents...",
                    show_label=False,
                    scale=9,
                    container=False
                )
                chat_btn = gr.Button("Send", variant="primary", scale=1)
            
            gr.Button("Clear Chat", variant="secondary", size="sm").click(
                lambda: [],
                outputs=chatbot
            )
        
        # Quiz Tab
        with gr.Tab("🎯 Quiz"):
            with gr.Row():
                # Left: Setup
                with gr.Column(scale=1):
                    gr.Markdown("### Create Quiz")
                    quiz_topic = gr.Textbox(
                        label="Topic",
                        placeholder="e.g., Machine Learning, Photosynthesis...",
                        lines=1
                    )
                    quiz_num = gr.Slider(
                        label="Number of Questions",
                        minimum=1,
                        maximum=15,
                        value=5,
                        step=1
                    )
                    gen_btn = gr.Button("Generate Quiz", variant="primary", size="lg")
                    gen_status = gr.Textbox(label="Status", interactive=False, lines=2)
                
                # Right: Quiz Interface
                with gr.Column(scale=1):
                    gr.Markdown("### Take Quiz")
                    quiz_box = gr.Column(visible=False)
                    
                    with quiz_box:
                        progress = gr.Textbox(label="Progress", interactive=False)
                        question_display = gr.Textbox(
                            label="Question",
                            interactive=False,
                            lines=4
                        )
                        answer_box = gr.Textbox(
                            label="Your Answer",
                            placeholder="Type your answer here...",
                            lines=3
                        )
                        submit_btn = gr.Button("Submit Answer", variant="primary")
                        feedback = gr.Markdown()
            
            gr.Markdown("---")
            results = gr.Markdown("### 📊 Results will appear here")
        
        # Analytics Tab
        with gr.Tab("📊 Analytics"):
            analytics = gr.Markdown("Complete a quiz to see analytics!")
            gr.Button("Refresh", variant="secondary").click(
                get_results,
                outputs=analytics
            )
    
    # Footer
    gr.Markdown("""
    ---
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>💡 Upload PDFs • Ask Questions • Generate Quizzes • Track Progress</p>
        <p style='font-size: 0.9em;'>Built with ❤️ using Gradio & LangChain | Works on all devices 📱💻</p>
    </div>
    """)
    
    # Event Handlers
    upload_btn.click(process_pdfs, pdf_upload, upload_status)
    
    chat_msg.submit(chat_with_pdf, [chat_msg, chatbot], chatbot)
    chat_msg.submit(lambda: "", None, chat_msg)
    chat_btn.click(chat_with_pdf, [chat_msg, chatbot], chatbot)
    chat_btn.click(lambda: "", None, chat_msg)
    
    gen_btn.click(generate_quiz, [quiz_topic, quiz_num], [gen_status, quiz_box])
    gen_btn.click(get_current_question, outputs=[question_display, progress])
    
    submit_btn.click(submit_answer, answer_box, [feedback, question_display, progress])
    submit_btn.click(lambda: "", None, answer_box)
    submit_btn.click(get_results, outputs=results)
    
    answer_box.submit(submit_answer, answer_box, [feedback, question_display, progress])
    answer_box.submit(lambda: "", None, answer_box)
    answer_box.submit(get_results, outputs=results)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)