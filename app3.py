import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import random
import seaborn as sns
from streamlit_option_menu import option_menu
from tensorflow.keras.preprocessing import image

# App Configuration
im = Image.open('assets/crafto-landing-page-features-ico-05.png')
st.set_page_config(
    page_title="HDR App (Handwritten Digit Recognition App)", 
    page_icon=im,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load custom CSS
local_css("assets/style.css")

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('OCR_MODEL/OCR_MODEL_1.h5', compile=False)

model = load_model()

# Sidebar Configuration
with st.sidebar:
    st.image('assets/crafto-landing-page-img-05.png', width=300)
    st.markdown("""
    <div class="sidebar-header">
        <h3>HDR App</h3>
        <p>Handwritten Digit Recognition & Educational Quiz</p>
    </div>
    """, unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=['Home', 'Browse File', 'Camera Input', 'Math Quiz', 'Demo', 'Documentation'],
        icons=['house', 'cloud-upload', 'camera', 'pencil-square', 'play-circle', 'book'],
        default_index=0,
        styles={
            "container": {"padding": "0!important"},
            "icon": {"color": "orange", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background": "linear-gradient(135deg, #6e8efb, #a777e3)"},
        }
    )

# Function to check if the image is blank
def is_blank_image(image, threshold=0.99):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    white_pixels = np.sum(gray_image >= 240)
    total_pixels = gray_image.size
    white_ratio = white_pixels / total_pixels
    return white_ratio > threshold

# Function to generate a random math question
def generate_random_question():
    operations = ['+', '-', '*', '/']
    operation = random.choice(operations)
    
    if operation == '+':
        num1 = random.randint(0, 9)
        num2 = random.randint(0, 9 - num1)
    elif operation == '-':
        num1 = random.randint(1, 9)
        num2 = random.randint(0, num1)
    elif operation == '*':
        num1 = random.randint(0, 3)
        num2 = random.randint(0, 3)
    elif operation == '/':
        num2 = random.randint(1, 9)
        num1 = num2 * random.randint(0, 9 // num2)
        
    question_str = f"{num1} {operation} {num2}"
    answer = eval(question_str)
    return {"question": question_str, "answer": int(answer)}

# Initialize session state
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'questions' not in st.session_state:
    st.session_state.questions = [generate_random_question() for _ in range(5)]
if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0
if 'canvas_data' not in st.session_state:
    st.session_state.canvas_data = {"objects": []}
if 'questions_asked' not in st.session_state:
    st.session_state.questions_asked = []

# Function to preprocess the image for prediction
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (90, 140))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to predict the digit
def predict_digit(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return np.argmax(prediction)

# Function for prediction probability
def prediction_proba(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return np.amax(prediction)

# Function to display a question and capture user input via canvas
def ask_question(question_data):
    with st.container():
        st.markdown(f"""
            <div class="quiz-header-container">
                <div class="quiz-progress-container">
                    <div class="progress-info">
                        <span class="progress-text">Question {st.session_state.current_question_index + 1} of 5</span>
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fill" style="width: {(st.session_state.current_question_index + 1) * 20}%"></div>
                        </div>
                    </div>
                    <div class="score-display">
                        <span class="score-label">Score:</span>
                        <span class="score-value">{st.session_state.score}</span>
                        <span class="score-divider">/</span>
                        <span class="score-total">5</span>
                    </div>
                </div>
                
                
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="question-card">
                <h4 class="question-text">What is the result of <span class="math-expression">{question_data["question"]}</span>?</h4>
            </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0)",
                stroke_width=15,
                stroke_color="#000000",
                background_color="#ffffff",
                height=300,
                width=400,
                drawing_mode="freedraw",
                key="canvas",
            )
            
            st.markdown("</div></div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="instructions-card">
                <div class="instructions-header">
                    <i class="fas fa-info-circle"></i>
                    <h4>Instructions</h4>
                </div>
                <ul class="instructions-list">
                    <li><i class="fas fa-pencil-alt"></i> Draw a single digit (0-9) in the white box</li>
                    <li><i class="fas fa-expand"></i> Make your digit large and clear</li>
                    <li><i class="fas fa-check-circle"></i> Click submit when done</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            predicted_answer = None
            if canvas_result.image_data is not None:
                image = Image.fromarray(canvas_result.image_data)
                image = image.convert("RGB")
                image = np.array(image)

                if not is_blank_image(image):
                    predicted_answer = predict_digit(image)
                    confidence = prediction_proba(image) * 100
                    
                    st.markdown(f"""
                    <div class="prediction-container">
                        <h4 class="prediction-title">AI Prediction:</h4>
                        <div class="prediction-card {'high-confidence' if confidence > 85 else 'medium-confidence' if confidence > 70 else 'low-confidence'}">
                            <span class="predicted-digit">{predicted_answer}</span>
                            <span class="confidence-level">{confidence:.1f}% confidence</span>
                        </div>
                        <div class="confidence-feedback">
                            {f"<p><i class='fas fa-thumbs-up'></i> The model is <strong>{'very' if confidence >= 85 else ''} confident</strong> in this prediction.</p>" 
                             if confidence >= 70 else 
                             "<p><i class='fas fa-exclamation-triangle'></i> The model is less confident. Try drawing more clearly.</p>"}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Please draw your answer before submitting.")

            if st.button("Submit Answer", use_container_width=True, type="primary"):
                if predicted_answer is not None:
                    if float(predicted_answer) == question_data["answer"]:
                        st.session_state.score += 1
                        st.balloons()
                        st.success("Correct! 🎉")
                    else:
                        st.error(f"Incorrect! The correct answer is {question_data['answer']}.")
                    
                    st.session_state.questions_asked.append((
                        question_data["question"], 
                        str(predicted_answer), 
                        str(question_data["answer"])
                    ))
                    
                    st.session_state.current_question_index += 1
                    st.rerun()
                else:
                    st.warning("Please draw your answer before submitting.")

# Home Page
if selected == "Home":
    st.markdown("""
    <div class="hero-gradient">
        <div class="hero-content">
            <h1>Handwritten Digit Recognition App</h1>
            <p class="hero-subtitle">AI-powered digit recognition with interactive math quiz</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-gradient">
            <div class="feature-card">
                <div class="feature-icon">📷</div>
                <h3>Image Upload</h3>
                <p>Upload images of handwritten digits for instant recognition with our AI model.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-gradient">
            <div class="feature-card">
                <div class="feature-icon">📱</div>
                <h3>Camera Input</h3>
                <p>Use your device's camera to capture digits in real-time for immediate analysis.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-gradient">
            <div class="feature-card">
                <div class="feature-icon">✏️</div>
                <h3>Interactive Quiz</h3>
                <p>Test your skills with our AI-powered math quiz that recognizes your handwritten answers.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="tech-gradient">
        <div class="tech-stack">
            <h3>Technology Stack</h3>
            <div class="tech-icons">
                <span class="tech-icon">TensorFlow</span>
                <span class="tech-icon">Streamlit</span>
                <span class="tech-icon">OpenCV</span>
                <span class="tech-icon">Python</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Quiz App
elif selected == "Math Quiz":
    st.markdown("""
    <div class="quiz-main-gradient">
        <div class="quiz-header">
            <h1>Math Quiz Challenge</h1>
            <p>Test your math skills! Draw your answers and our AI will check them.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.current_question_index < len(st.session_state.questions):
        current_question = st.session_state.questions[st.session_state.current_question_index]
        ask_question(current_question)
    else:
        st.markdown(f"""
        <div class="completion-gradient">
            <div class="quiz-complete">
                <div class="trophy-icon">
                    <i class="fas fa-trophy"></i>
                </div>
                <h2>Quiz Complete! 🎉</h2>
                <div class="final-score-display">
                    <div class="score-circle">
                        <span class="score-number">{st.session_state.score}</span>
                        <span class="score-label">out of 5</span>
                    </div>
                    <div class="score-message">
                        {'<h3>Perfect score! Amazing! 🌟</h3>' if st.session_state.score == 5 else 
                         '<h3>Great job! 👍</h3><p>You did well, but there\'s room for improvement</p>' if st.session_state.score >= 3 else 
                         '<h3>Keep practicing! 💪</h3><p>You\'ll do better next time!</p>'}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Restart Quiz", use_container_width=True, type="primary"):
            st.session_state.score = 0
            st.session_state.questions_asked = []
            st.session_state.current_question_index = 0
            st.session_state.questions = [generate_random_question() for _ in range(5)]
            st.rerun()
        
        results = pd.DataFrame(
            st.session_state.questions_asked, 
            columns=["Question", "Your Answer", "Correct Answer"]
        )
        
        st.markdown("""
        <div class="results-gradient">
            <div class="results-container">
                <h3>Quiz Results Summary</h3>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            def highlight_correct(row):
                if row['Your Answer'] == row['Correct Answer']:
                    return ['background-color: #C1D8C3'] * len(row)
                else:
                    return ['background-color: #F7374F'] * len(row)

            st.dataframe(
                results.style.apply(highlight_correct, axis=1),
                use_container_width=True
            )
        
        with col2:
            correct_count = results[results["Your Answer"] == results["Correct Answer"]].shape[0]
            incorrect_count = len(results) - correct_count
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(
                [correct_count, incorrect_count], 
                labels=['Correct', 'Incorrect'], 
                colors=['#28a745', '#dc3545'],
                autopct='%1.1f%%', 
                startangle=90,
                textprops={'fontsize': 14}
            )
            ax.axis('equal')
            
            st.pyplot(fig)
        
        st.markdown("</div></div>", unsafe_allow_html=True)

# File Upload
elif selected == "Browse File":
    st.markdown("""
    <div class="upload-gradient">
        <div class="upload-header">
            <h1>Image Upload</h1>
            <p>Upload an image of a handwritten digit for recognition</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        with st.spinner("Processing image..."):
            image = Image.open(uploaded_file)
            img = cv2.resize(np.array(image), (800, 300))
            
            col1, col2 = st.columns([1, 1], gap="large")
            
            with col1:
                st.image(img, caption='Uploaded Image', use_column_width=True)
            
            with col2:
                digit = predict_digit(image)
                proba = prediction_proba(image) * 100
                
                st.markdown(f"""
                <div class="prediction-gradient">
                    <div class="prediction-result">
                        <h3>AI Prediction Results</h3>
                        <div class="prediction-card {'high-confidence' if proba >= 85 else 'medium-confidence' if proba >= 70 else 'low-confidence'}">
                            <span class="predicted-digit">{digit}</span>
                            <span class="confidence-level">{proba:.1f}% confidence</span>
                        </div>
                        <div class="confidence-message">
                            {f"<p>The model is <strong>{'very' if proba >= 85 else ''} confident</strong> in this prediction.</p>" 
                             if proba >= 70 else 
                             "<p>The model is less confident in this prediction. Try a clearer image.</p>"}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# Camera Input
elif selected == "Camera Input":
    st.markdown("""
    <div class="camera-gradient">
        <div class="camera-header">
            <h1>Live Camera Recognition</h1>
            <p>Use your camera to capture handwritten digits in real-time</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    camera_image = st.camera_input("Take a picture of a handwritten digit", label_visibility="collapsed")
    
    if camera_image is not None:
        with st.spinner("Processing image..."):
            image = Image.open(camera_image)
            
            col1, col2 = st.columns([1, 1], gap="large")
            
            with col1:
                st.image(image, caption='Captured Image', use_column_width=True)
            
            with col2:
                digit = predict_digit(image)
                proba = prediction_proba(image) * 100
                
                st.markdown(f"""
                <div class="prediction-gradient">
                    <div class="prediction-result">
                        <h3>AI Prediction Results</h3>
                        <div class="prediction-card {'high-confidence' if proba >= 85 else 'medium-confidence' if proba >= 70 else 'low-confidence'}">
                            <span class="predicted-digit">{digit}</span>
                            <span class="confidence-level">{proba:.1f}% confidence</span>
                        </div>
                        <div class="confidence-message">
                            {f"<p>The model is <strong>{'very' if proba >= 85 else ''} confident</strong> in this prediction.</p>" 
                             if proba >= 70 else 
                             "<p>The model is less confident in this prediction. Try better lighting or clearer handwriting.</p>"}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# Demo
elif selected == "Demo":
    st.markdown("""
    <div class="demo-gradient">
        <div class="demo-header">
            <h1>App Demonstration</h1>
            <p>Watch how the Handwritten Digit Recognition App works</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.video('OCR_DEMO/OCR_APP_Demo.mkv', format="video/mkv", start_time=0)

# Documentation
elif selected == "Documentation":
    st.markdown("""
    <div class="doc-gradient">
        <div class="doc-header">
            <h1>Project Documentation</h1>
            <p>Handwritten Digit Recognition & Educational Quiz App</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("Project Overview", expanded=True):
        st.markdown("""
        <div class="doc-section-gradient">
            <div class="doc-section">
                <p>This application combines machine learning with an interactive educational interface to recognize handwritten digits 
                and provide a math quiz experience for young learners. The app features multiple input methods including image upload, 
                camera capture, and a drawing canvas for real-time digit recognition.</p>
                
                <h3>Key Features</h3>
                <ul>
                    <li><strong>Digit Recognition</strong>: AI model trained on MNIST dataset with 97% accuracy</li>
                    <li><strong>Multiple Input Methods</strong>: File upload, camera capture, and interactive drawing</li>
                    <li><strong>Educational Quiz</strong>: Math problems with handwritten answer recognition</li>
                    <li><strong>Real-time Feedback</strong>: Immediate results with confidence scoring</li>
                    <li><strong>Performance Analytics</strong>: Visual summary of quiz results</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("Technical Details"):
        st.markdown("""
        <div class="doc-section-gradient">
            <div class="doc-section">
                <h3>Model Architecture</h3>
                <p>The application uses a Convolutional Neural Network (CNN) implemented with TensorFlow/Keras. The model was trained 
                on the MNIST dataset and achieves 97% accuracy on validation data.</p>
                
                <h3>Technologies Used</h3>
                <div class="tech-stack">
                    <span class="tech-badge">Python</span>
                    <span class="tech-badge">TensorFlow</span>
                    <span class="tech-badge">Streamlit</span>
                    <span class="tech-badge">OpenCV</span>
                    <span class="tech-badge">Matplotlib</span>
                    <span class="tech-badge">Pandas</span>
                    <span class="tech-badge">NumPy</span>
                </div>
                
                <h3>Performance Metrics</h3>
                <ul>
                    <li>Training Accuracy: 99.2%</li>
                    <li>Validation Accuracy: 97.1%</li>
                    <li>Inference Time: ~50ms per prediction</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("How It Works"):
        st.markdown("""
        <div class="doc-section-gradient">
            <div class="doc-section">
                <h3>Image Processing Pipeline</h3>
                <ol>
                    <li>Input image is converted to grayscale</li>
                    <li>Image is resized to 90x140 pixels</li>
                    <li>Pixel values are normalized to 0-1 range</li>
                    <li>Image is passed through the CNN model</li>
                    <li>Model outputs probabilities for each digit (0-9)</li>
                    <li>Digit with highest probability is selected</li>
                </ol>
                
                <h3>Quiz Logic</h3>
                <ol>
                    <li>System generates random math questions</li>
                    <li>User draws answer on canvas</li>
                    <li>AI recognizes the drawn digit</li>
                    <li>System compares with correct answer</li>
                    <li>Feedback and score are updated</li>
                </ol>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("Future Enhancements"):
        st.markdown("""
        <div class="doc-section-gradient">
            <div class="doc-section">
                <ul>
                    <li>Multi-digit recognition</li>
                    <li>Support for mathematical symbols</li>
                    <li>User accounts and progress tracking</li>
                    <li>Mobile app version</li>
                    <li>Additional quiz types and difficulty levels</li>
                    <li>Enhanced model explainability features</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
