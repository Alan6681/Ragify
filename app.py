import streamlit as st
from services.rag_chain import RagChain
from langchain_core.messages import HumanMessage, AIMessage
import time
import json


st.set_page_config(
    page_title="Ragify",
    page_icon= "📕",
    layout="centered"
)
if not "uploaded_file_names" in st.session_state:
    st.session_state.uploaded_file_names = []

if not "current_question_index" in st.session_state:
    st.session_state.current_question_index = 0

if not "quiz_questions" in st.session_state:
    st.session_state.quiz_questions = []

if not "quiz_answers" in st.session_state:
    st.session_state.quiz_answers = []

if not "quiz_evaluations" in st.session_state:
    st.session_state.quiz_evaluations = []

if not "quiz_scores" in st.session_state:
    st.session_state.quiz_scores = []  # Store True/False for correct/incorrect

if not "answer_times" in st.session_state:
    st.session_state.answer_times = []  # Store time taken for each answer

if not "question_start_time" in st.session_state:
    st.session_state.question_start_time = None

if not "num_questions" in st.session_state:
    st.session_state.num_questions = 1

if not "topic" in st.session_state:
    st.session_state.topic = ""

if not "mode" in st.session_state:
    st.session_state.mode = "Chat"

if not "quiz_mode" in st.session_state:
    st.session_state.quiz_mode = False

if not "quiz_complete" in st.session_state:
    st.session_state.quiz_complete = False

if not "chat_history" in st.session_state:
    st.session_state["chat_history"] = []

if not "vectorstore" in st.session_state:
    st.session_state["vectorstore"] = None

if not "messages" in st.session_state:
    st.session_state["messages"] = [{"role":"assistant", "content": "Hello there ☺️, What are you curious about"}]

st.title("Ragify ")

with st.sidebar:
    st.header("Upload File(s) 📁")
    uploaded_files = st.file_uploader(label="Please Upload your pdf", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        current_file_names = [file.name for file in uploaded_files]

        files_changed = (
            set(current_file_names) != set(st.session_state.uploaded_file_names)
        )

        if files_changed:
            st.info(f"📄 {len(uploaded_files)} file(s) uploaded. Processing...")
            # Update the stored file names
            st.session_state.uploaded_file_names = current_file_names
            
            # Force recreation of RAG chain with new files
            if "rag_chain" in st.session_state:
                del st.session_state["rag_chain"]
            
            # Clear chat history when files change
            st.session_state.chat_history = []
            st.session_state.messages = [{"role":"assistant", "content": "Hello there ☺️, What are you curious about the new documents?"}]

        st.markdown("---")
        st.markdown("## Select Mode ⚙️")
        col1, col2 = st.columns(2)
        with col1:
            chat_mode = st.button("Chat Mode", type="secondary")
        if chat_mode:
            st.session_state.mode = "Chat"
            st.session_state.quiz_mode = False

        with col2:
            quiz_mode = st.button("Quiz Mode", type="secondary")
        st.markdown("<br>", unsafe_allow_html=True)
        if quiz_mode:
            st.session_state.mode = "Quiz"
            st.session_state.quiz_mode = False
            # Reset quiz state
            st.session_state.current_question_index = 0
            st.session_state.quiz_questions = []
            st.session_state.quiz_answers = []
            st.session_state.quiz_evaluations = []
            st.session_state.quiz_scores = []
            st.session_state.answer_times = []
            st.session_state.quiz_complete = False
            st.session_state.question_start_time = None

        col1, col2, col3 = st.columns([1,3,1])

        with col2:
            evaluation = st.button("Evaluation 📈")
            if evaluation:
                st.session_state.mode = "Evaluation"
                st.rerun()

        st.markdown("---")
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            clear_chat = st.button("Clear chat ❌", type="primary")
            if clear_chat:
                st.session_state.messages = [{"role":"assistant", "content": "Hello there ☺️, What are you curious about"}]
                st.session_state.chat_history = []
                st.session_state.quiz_mode = False
                st.rerun()
    

if uploaded_files:
    if not "rag_chain" in st.session_state:
        with st.spinner("🔄️ Processing documents and creating knowledge base"):
            st.session_state["rag_chain"] = RagChain(uploaded_files)
        st.success("✅ Documents processed successfully!")
        time.sleep(1)
        st.rerun()

    if st.session_state.mode == "Chat":
        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Chat input
        if user_query := st.chat_input("Ask about your Pdf"):
            st.session_state.messages.append({"role" : "user", "content": user_query})
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            with st.chat_message("user"):
                st.write(user_query)

            with st.chat_message("assistant"):
                with st.spinner("thinking..."):
                    response = st.session_state.rag_chain.ask(question=user_query, chat_history=st.session_state.chat_history)
                    st.write(response)

                    st.session_state.chat_history.append(AIMessage(content=response))
                    st.session_state.messages.append({"role": "assistant", "content": response})

    elif st.session_state.mode == "Quiz":
        if not st.session_state.quiz_mode:
            # Quiz setup screen
            st.write("This is not who wants to be a Millionaire 🤭")
            st.info("How many rounds are you down for?")
            
            with st.form("quiz_setup"):
                with st.expander("Choose a topic for your quiz (optional)"):
                     topic = st.text_input("What topic would you like to be tested on?", placeholder="e.g., Photosynthesis, World War II",)
                num_questions = st.number_input("Number of questions", min_value=1, max_value=40, value=5)
                submit = st.form_submit_button("Start Quiz 🚀")
                
                if submit and num_questions:
                    with st.spinner("Getting your questions ready..."):
                        st.session_state.num_questions = num_questions
                        st.session_state.topic = topic
                        
                        # Generate all questions at once with different contexts
                        all_questions = st.session_state.rag_chain.quiz_generator(
                            topic=topic, 
                            num_questions=num_questions
                        )
                        
                        # Store each question with its context
                        st.session_state.quiz_questions = []
                        for question, context in all_questions:
                            st.session_state.quiz_questions.append({
                                "question": question,
                                "context": context
                            })
                        
                        st.session_state.quiz_mode = True
                        st.session_state.current_question_index = 0
                        st.rerun()
        
        else:
            # Quiz active screen
            if not st.session_state.quiz_complete:
                current_idx = st.session_state.current_question_index
                total_questions = len(st.session_state.quiz_questions)
                
                # Start timer when question is displayed
                if st.session_state.question_start_time is None:
                    st.session_state.question_start_time = time.time()
                
                # Progress indicator
                st.progress((current_idx) / total_questions)
                st.write(f"### Question {current_idx + 1} of {total_questions}")
                st.write(f"**Topic:** {st.session_state.topic}")
                
                # Show timer
                # if st.session_state.question_start_time:
                #     elapsed = int(time.time() - st.session_state.question_start_time)
                #     col1, col2 = st.columns([3, 1])
                #     with col2:
                #         st.metric("⏱️ Time", f"{elapsed}s")
                
                st.write("---")
                
                # Display current question
                current_q = st.session_state.quiz_questions[current_idx]
                with st.chat_message("assistant"):
                    st.write(current_q["question"])
                
                # Display previous answers for this question if any
                if current_idx < len(st.session_state.quiz_answers):
                    with st.chat_message("user"):
                        st.write(st.session_state.quiz_answers[current_idx])
                    
                    # Show evaluation with color coding
                    eval_data = st.session_state.quiz_evaluations[current_idx]
                    is_correct = st.session_state.quiz_scores[current_idx]
                    
                    with st.chat_message("assistant"):
                        if is_correct:
                            st.success(f"✅ Correct! {eval_data}")
                        else:
                            st.error(f"❌ Incorrect. {eval_data}")
                        
                        # Show time taken
                        time_taken = st.session_state.answer_times[current_idx]
                        st.caption(f"⏱️ You took {time_taken:.1f} seconds")
                
                # Get user answer
                user_answer = st.chat_input(f"Your answer for question {current_idx + 1}...")
                
                if user_answer:
                    # Calculate time taken
                    time_taken = time.time() - st.session_state.question_start_time
                    
                    # Store answer
                    if current_idx >= len(st.session_state.quiz_answers):
                        st.session_state.quiz_answers.append(user_answer)
                        st.session_state.answer_times.append(time_taken)
                    else:
                        st.session_state.quiz_answers[current_idx] = user_answer
                        st.session_state.answer_times[current_idx] = time_taken
                    
                    # Evaluate answer
                    with st.spinner("Evaluating your answer..."):
                        evaluation = st.session_state.rag_chain.evaluate(
                            context=current_q["context"],
                            question=current_q["question"],
                            user_answer=user_answer
                        )
                        
                        # Parse evaluation to extract score
                        try:
                            eval_json = json.loads(evaluation)
                            is_correct = eval_json.get("correct", False)
                            feedback = eval_json.get("feedback", evaluation)
                        except:
                            # Fallback: check if evaluation contains positive keywords
                            is_correct = any(word in evaluation.lower() for word in ["correct", "right", "accurate", "excellent", "good"])
                            feedback = evaluation
                    
                    # Store evaluation and score
                    if current_idx >= len(st.session_state.quiz_evaluations):
                        st.session_state.quiz_evaluations.append(feedback)
                        st.session_state.quiz_scores.append(is_correct)
                    else:
                        st.session_state.quiz_evaluations[current_idx] = feedback
                        st.session_state.quiz_scores[current_idx] = is_correct
                    
                    # Reset timer for next question
                    st.session_state.question_start_time = None
                    
                    # Move to next question
                    if current_idx + 1 < total_questions:
                        st.session_state.current_question_index += 1
                    else:
                        st.session_state.quiz_complete = True
                    
                    st.rerun()
                
                # Navigation buttons
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    if current_idx > 0:
                        if st.button("⬅️ Previous"):
                            st.session_state.current_question_index -= 1
                            st.session_state.question_start_time = None
                            st.rerun()
                
                with col3:
                    if current_idx < total_questions - 1 and current_idx < len(st.session_state.quiz_answers):
                        if st.button("Next ➡️"):
                            st.session_state.current_question_index += 1
                            st.session_state.question_start_time = None
                            st.rerun()
            else:
                # Quiz complete - show results
                st.success("🎉 Quiz Complete!")
                st.write(f"### Results for: {st.session_state.topic}")
                st.write(f"Total Questions: {len(st.session_state.quiz_questions)}")
                st.write("---")
                
                # Display all Q&A with evaluations
                for idx, q_data in enumerate(st.session_state.quiz_questions):
                    with st.expander(f"Question {idx + 1}"):
                        st.write(f"**Q:** {q_data['question']}")
                        st.write(f"**Your Answer:** {st.session_state.quiz_answers[idx]}")
                        st.write(f"**Evaluation:** {st.session_state.quiz_evaluations[idx]}")
                
                # Restart or go back
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Retake Quiz"):
                        st.session_state.current_question_index = 0
                        st.session_state.quiz_answers = []
                        st.session_state.quiz_evaluations = []
                        st.session_state.quiz_complete = False
                        st.rerun()
                
                with col2:
                    if st.button("← Back to Setup"):
                        st.session_state.quiz_mode = False
                        st.session_state.current_question_index = 0
                        st.session_state.quiz_questions = []
                        st.session_state.quiz_answers = []
                        st.session_state.quiz_evaluations = []
                        st.session_state.quiz_complete = False
                        st.rerun()

    elif st.session_state.mode == "Evaluation":
        if st.session_state.quiz_complete == False:
            st.info("Please complete a quiz first to view the performance dashboard.")
            if st.button("← Back to Quiz"):
                st.session_state.mode = "Quiz"
                st.rerun()
        else:
            st.write("## 📊 Performance Dashboard")
    
    # if not st.session_state.quiz_answers:
    #     st.info("📚 Complete a quiz to unlock your performance dashboard!")
    #     if st.button("Start a Quiz 🚀"):
    #         st.session_state.mode = "Quiz"
    #         st.session_state.quiz_mode = False
    #         st.rerun()
    # else:
        # Calculate statistics
            total_questions = len(st.session_state.quiz_questions)
            answered_questions = len(st.session_state.quiz_answers)
            correct_answers = sum(st.session_state.quiz_scores)
            incorrect_answers = answered_questions - correct_answers
            accuracy = (correct_answers / answered_questions * 100) if answered_questions > 0 else 0
            avg_time = sum(st.session_state.answer_times) / len(st.session_state.answer_times) if st.session_state.answer_times else 0
            total_time = sum(st.session_state.answer_times)
            
            # Tabs for different views
            tab1, tab2, tab3 = st.tabs(["📈 Overview", "⚡ Speed Analysis", "📝 Detailed Review"])
            
            with tab1:
                # Key Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.1f}%", 
                            delta=f"{correct_answers}/{answered_questions}")
                with col2:
                    st.metric("Correct", correct_answers, 
                            delta_color="normal")
                with col3:
                    st.metric("Incorrect", incorrect_answers,
                            delta_color="inverse")
                with col4:
                    st.metric("Avg Time", f"{avg_time:.1f}s")
                
                st.write("---")
                
                # Score Breakdown Chart
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### 🎯 Score Distribution")
                    # Create a simple bar chart for correct vs incorrect
                    import pandas as pd
                    
                    score_data = pd.DataFrame({
                        'Result': ['Correct ✅', 'Incorrect ❌'],
                        'Count': [correct_answers, incorrect_answers]
                    })
                    st.bar_chart(score_data.set_index('Result'))
                
                with col2:
                    st.write("### ⏱️ Total Time")
                    st.metric("Total Quiz Time", f"{total_time:.1f}s")
                    st.metric("Questions Answered", f"{answered_questions}/{total_questions}")
                    
                    # Performance rating
                    if accuracy >= 90:
                        st.success("🌟 Excellent Performance!")
                    elif accuracy >= 70:
                        st.info("👍 Good Job!")
                    elif accuracy >= 50:
                        st.warning("📚 Keep Practicing!")
                    else:
                        st.error("💪 Need More Study!")
            
            with tab2:
                st.write("### ⚡ Response Speed Analysis")
                
                # Response time chart
                import pandas as pd
                
                time_data = pd.DataFrame({
                    'Question': [f'Q{i+1}' for i in range(len(st.session_state.answer_times))],
                    'Time (seconds)': st.session_state.answer_times,
                    'Correct': ['✅' if score else '❌' for score in st.session_state.quiz_scores]
                })
                
                st.write("#### 📊 Time per Question")
                st.line_chart(time_data.set_index('Question')['Time (seconds)'])
                
                st.write("---")
                
                # Speed statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    fastest = min(st.session_state.answer_times)
                    st.metric("⚡ Fastest", f"{fastest:.1f}s")
                with col2:
                    slowest = max(st.session_state.answer_times)
                    st.metric("🐌 Slowest", f"{slowest:.1f}s")
                with col3:
                    st.metric("📊 Average", f"{avg_time:.1f}s")
                
                st.write("---")
                
                # Detailed table
                st.write("#### 📋 Question-by-Question Breakdown")
                display_data = time_data.copy()
                display_data['Time (seconds)'] = display_data['Time (seconds)'].round(1)
                st.dataframe(display_data, use_container_width=True, hide_index=True)
                
                # Speed insights
                st.write("#### 💡 Speed Insights")
                if avg_time < 10:
                    st.info("🚀 You're a speed demon! Make sure you're reading carefully.")
                elif avg_time < 30:
                    st.success("⏱️ Great balance of speed and accuracy!")
                else:
                    st.warning("🤔 Take your time, but try to be more decisive.")
            
            with tab3:
                # Detailed Q&A review
                st.write("### 📝 Question-by-Question Review")
                
                for idx, (q, a, e, score, time_taken) in enumerate(zip(
                    st.session_state.quiz_questions,
                    st.session_state.quiz_answers,
                    st.session_state.quiz_evaluations,
                    st.session_state.quiz_scores,
                    st.session_state.answer_times
                )):
                    status = "✅ Correct" if score else "❌ Incorrect"
                    with st.expander(f"Question {idx + 1} - {status} (⏱️ {time_taken:.1f}s)", expanded=False):
                        st.markdown(f"**Question:**  \n{q['question']}")
                        st.markdown("---")
                        st.markdown(f"**Your Answer:**  \n{a}")
                        st.markdown("---")
                        st.markdown("**Evaluation:**")
                        if score:
                            st.success(e)
                        else:
                            st.error(e)
            
            st.write("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 Retake Quiz"):
                    st.session_state.mode = "Quiz"
                    st.session_state.current_question_index = 0
                    st.session_state.quiz_answers = []
                    st.session_state.quiz_evaluations = []
                    st.session_state.quiz_scores = []
                    st.session_state.answer_times = []
                    st.session_state.quiz_complete = False
                    st.session_state.question_start_time = None
                    st.rerun()
            with col2:
                if st.button("🆕 New Quiz"):
                    st.session_state.mode = "Quiz"
                    st.session_state.quiz_mode = False
                    st.session_state.current_question_index = 0
                    st.session_state.quiz_questions = []
                    st.session_state.quiz_answers = []
                    st.session_state.quiz_evaluations = []
                    st.session_state.quiz_scores = []
                    st.session_state.answer_times = []
                    st.session_state.quiz_complete = False
                    st.session_state.question_start_time = None
                    st.rerun()
            with col3:
                if st.button("← Back to Chat"):
                    st.session_state.mode = "Chat"
                    st.rerun()
            
        # st.write("## 📊 Performance Analytics")
        
        # if not st.session_state.quiz_answers or not st.session_state.quiz_evaluations:
        #     st.info("No quiz data available yet. Complete a quiz first!")
        #     if st.button("← Back"):
        #         st.session_state.mode = "Chat"
        #         st.rerun()
        # else:
        #     # Overall Stats
        #     total_questions = len(st.session_state.quiz_answers)
            
        #     col1, col2, col3 = st.columns(3)
        #     with col1:
        #         st.metric("Total Questions", total_questions)
        #     with col2:
        #         st.metric("Topic", st.session_state.topic)
        #     with col3:
        #         st.metric("Completion Rate", f"{(len(st.session_state.quiz_answers)/st.session_state.num_questions)*100:.0f}%")
            
        #     st.write("---")
            
        #     # Detailed breakdown
        #     st.write("### Question-by-Question Analysis")
        #     for idx, (question, answer, evaluation) in enumerate(zip(
        #         st.session_state.quiz_questions,
        #         st.session_state.quiz_answers,
        #         st.session_state.quiz_evaluations
        #     )):
        #         with st.expander(f"📝 Question {idx + 1}", expanded=False):
        #             st.write(f"**Question:** {question['question']}")
        #             st.write(f"**Your Answer:** {answer}")
        #             st.write(f"**Evaluation:**")
        #             st.info(evaluation)
            
        #     # Back button
        #     if st.button("← Back to Chat"):
        #         st.session_state.mode = "Chat"
        #         st.rerun()

else:
        st.info("👈Upload your PDF ")