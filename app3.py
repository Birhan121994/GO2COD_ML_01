import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2  # OpenCV for image processing
import tensorflow as tf
from PIL import Image
import random
import seaborn as sns
import threading
from streamlit_option_menu import option_menu
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K

# Clear any existing Keras sessions (helps avoid conflicts with name scopes)
K.clear_session()

im = Image.open('assets/crafto-landing-page-features-ico-05.png')
st.set_page_config(page_title="HDR App (Handwritten Digit Recognition App)", page_icon = im)

# Load the trained model
model = tf.keras.models.load_model('OCR_MODEL/OCR_MODEL_1.h5')

st.sidebar.image('assets/crafto-landing-page-img-05.png', width=300)

st.write("<style>.css-18ni7ap{display:none;}</style>", unsafe_allow_html=True)
st.write("<style>.block-container css-1y4p8pa egzxvld4{padding:1rem 1rem;margin-top:0px;}</style>", unsafe_allow_html = True)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .css-1dp5vir{
                background-image:None;
            }

            .nav-link.active[data-v-ef155198] {
                background-color: #4b8eff;
            }
            .menu.nav-item.nav-link{
                background-color:cef8f7;
                color:white;
                font-weight:600;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Function to check if the image is blank
def is_blank_image(image, threshold=0.99):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Calculate the number of white pixels
    white_pixels = np.sum(gray_image >= 240)  # Count white pixels (value >= 240)
    total_pixels = gray_image.size
    white_ratio = white_pixels / total_pixels
    
    # Return True if the ratio of white pixels is above the threshold, meaning the image is blank
    return white_ratio > threshold

# Function to generate a random math question
def generate_random_question():
    operations = ['+', '-', '*', '/']
    operation = random.choice(operations)
    
    if operation == '+':
        num1 = random.randint(0, 9)
        num2 = random.randint(0, 9 - num1)  # Ensure sum does not exceed 9
    elif operation == '-':
        num1 = random.randint(1, 9)
        num2 = random.randint(0, num1)  # Ensure result is non-negative
    elif operation == '*':
        num1 = random.randint(0, 3)
        num2 = random.randint(0, 3)
    elif operation == '/':
        num2 = random.randint(1, 9)
        num1 = num2 * random.randint(0, 9 // num2)  # Ensure integer division
        
    question_str = f"{num1} {operation} {num2}"
    answer = eval(question_str)  # Calculate the answer
    return {"question": question_str, "answer": int(answer)}

# Initialize session state for score, questions, and index
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'questions' not in st.session_state:
    st.session_state.questions = [generate_random_question() for _ in range(5)]
if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0
if 'canvas_data' not in st.session_state:
    st.session_state.canvas_data = {"objects": []}
if 'questions_asked' not in st.session_state:  # Initialize questions_asked
    st.session_state.questions_asked = []


# Function to preprocess the image for prediction
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (90, 140))  # Adjust based on your model's input size
    img = img.astype('float32') / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add channel dimension
    return img

# Function to predict the digit
def predict_digit(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return np.argmax(prediction)

# Function for prediction probablity
def prediction_proba(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return np.amax(prediction)

# Function to display a question and capture user input via canvas
def ask_question(question_data):
    st.subheader(f'What is the result of {question_data["question"]} ?')
    
    # Create a canvas to draw the answer
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    predicted_answer = None
    if canvas_result.image_data is not None:
        image = Image.fromarray(canvas_result.image_data)  # Convert to PIL Image
        image = image.convert("RGB")  # Convert to RGB
        image = np.array(image)  # Convert back to NumPy array

        if not is_blank_image(image):  # Check if the image is blank
            predicted_answer = predict_digit(image)
            st.info(f"Predicted Answer: {predicted_answer}")
        else:
            st.warning("Please draw your answer before submitting.")

    if st.button("Submit"):
        if predicted_answer is not None:
            if float(predicted_answer) == question_data["answer"]:
                st.session_state.score += 1
                st.success("Correct!")
                
            else:
                st.error(f"Incorrect! The correct answer is {question_data['answer']}.")
                
            
            # Track asked questions
            st.session_state.questions_asked.append((question_data["question"], str(predicted_answer), str(question_data["answer"])))
            
            # Move to the next question
            st.session_state.current_question_index += 1
            st.rerun()  # Rerun to show the next question
        else:
            st.warning("Please draw your answer before submitting.")

    

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=['Browse File', 'Take Camera', 'Quiz app', 'Demo', 'Doc'],
        icons=['house', 'yin-yang','file-bar-graph-fill', 'steam', 'steam'],
        menu_icon='cast',
        default_index=0,
        orientation="vertical"
    )

if selected == "Quiz app":
    # Streamlit app
    st.title("Math Quiz App")
    st.write("Answer the following questions:")

    # Check if there are more questions to ask
    if st.session_state.current_question_index < len(st.session_state.questions):
        current_question = st.session_state.questions[st.session_state.current_question_index]
        ask_question(current_question)
    else:
        # Show final score
        if st.session_state.score == 5:
            st.balloons()
        st.info(f"Your final score is: {st.session_state.score}/{len(st.session_state.questions)}")
        
        # Reset quiz button
        if st.button("Restart Quiz"):
            st.session_state.score = 0
            st.session_state.questions_asked = []
            st.session_state.current_question_index = 0
            st.session_state.questions = [generate_random_question() for _ in range(5)]  # Regenerate questions
            st.rerun()
        # Create a DataFrame for charting results
        results = pd.DataFrame(st.session_state.questions_asked, columns=["Question", "Your Answer", "Correct Answer"])
        # pr = ProfileReport(df, explorative=True)
        # st_profile_report(pr)
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.write(results)
        
        with col2:
            # Plot results
            correct_count = results[results["Your Answer"] == results["Correct Answer"]].shape[0]
            incorrect_count = len(results) - correct_count
            labels = ['Correct', 'Incorrect']
            sizes = [correct_count, incorrect_count]
            
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

            st.pyplot(fig1)
        




elif selected == "Browse File":
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img = cv2.resize(np.array(image), (800, 300))
        st.image(img, caption='Uploaded Image.')
        digit= predict_digit(image)
        proba = prediction_proba(image)
        proba_val = int(proba * 100)
        st.info(f"Predicted digit: {digit}")
        if proba_val >= 85:
            st.success(f"The model predicts the digit {digit} with the prediction probablity of {proba_val}%. i.e The model is most certain about the digit it predicted is the most accurate prediction.")
        elif proba_val >= 75:
            st.info(f"The model predicts the digit {digit} with the prediction probablity of {proba_val}%. i.e The model is most precise about the digit it predicted is the most precise prediction.")
        elif proba_val >= 55:
            st.warning(f"The model predicts the digit {digit} with the prediction probablity of {proba_val}%.")
        elif proba_val < 55:
            st.error(f"The model predicts the digit {digit} with the prediction probablity of {proba_val}%.The model is simply trying guessing.")

            
            


elif selected == "Take Camera":
    camera_image = st.camera_input("Take a picture...")
    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption='Captured Image.', use_column_width=True)
        digit = predict_digit(image)
        proba = prediction_proba(image)
        proba_val = int(proba * 100)
        st.info(f"Predicted digit: {digit}")
        if proba_val >= 85:
            st.success(f"The model predicts the digit {digit} with the prediction probablity of {proba_val}%. i.e The model is most certain about the digit it predicted is the most accurate prediction.")
        elif proba_val >= 75:
            st.info(f"The model predicts the digit {digit} with the prediction probablity of {proba_val}%. i.e The model is most precise about the digit it predicted is the most precise prediction.")
        elif proba_val >= 55:
            st.warning(f"The model predicts the digit {digit} with the prediction probablity of {proba_val}%.")
        elif proba_val < 55:
            st.error(f"The model predicts the digit {digit} with the prediction probablity of {proba_val}%.The model is simply trying guessing.")


elif selected == "Demo":
    st.video('OCR_DEMO/OCR_APP_Demo.mkv', format="video/mkv", start_time=0)

elif selected == "Doc":
    st.header("Project Documentation: Handwritten Digit Recognition & Educational Quiz App")
    
    # Introduction
    st.header("Overview")
    st.write("""
        I developed a Streamlit-based web application that utilizes machine learning for handwritten digit recognition 
        and provides an interactive educational quiz experience. The app leverages a TensorFlow-based deep learning model 
        to recognize handwritten digits, enabling a range of interactive features like image upload, live camera capture, 
        and a fun quiz for young learners.
    """)

    # Key Features
    st.header("Key Features")
    st.write("""
    1. **Handwritten Digit Recognition**:
       - Users can upload a photo of a handwritten digit, take a picture using their device camera, or draw a digit 
         directly on the app’s canvas.
       - The app utilizes a TensorFlow model trained on a large handwritten digit dataset (MNIST) to predict the digit 
         in real-time with an impressive **97% accuracy** on validation data.
       
    2. **Interactive Quiz for Kindergarten Students**:
       - The app generates 5 random mathematical questions (addition, subtraction, multiplication, and division) suitable 
         for kindergarten students.
       - The questions are displayed alongside a drawable canvas where students can write their answers (e.g., handwritten digits).
       - Once the user submits their answer, the AI model predicts the digit, and the app checks if the answer is correct.
       - Feedback on the correctness of the answer is provided, with a DataFrame summary of responses and an interactive pie chart 
         visualizing the number of correct vs. incorrect answers.

    3. **User-Friendly Interface**:
       - Three main interaction modes are provided:
         - **Upload File**: Users can upload an image of handwritten digits for recognition.
         - **Take Camera**: Users can take a picture using the device's camera for real-time digit recognition.
         - **Quiz Mode**: A fun and engaging exercise for children to practice math and handwriting recognition.
    """)

    # Technical Stack
    st.header("Technical Stack")
    st.write("""
    - **Frontend**: Streamlit (for web app interface)
    - **Backend**: Python (for data processing and machine learning)
    - **Machine Learning Framework**: TensorFlow (for handwritten digit recognition model)
    - **Visualization**: Matplotlib (for data visualization in quiz results)
    - **Data**: MNIST dataset for digit recognition, custom dataset for generating math quizzes
    """)

    # Model Overview
    st.header("Model Overview")
    st.write("""
    I used TensorFlow to train a Convolutional Neural Network (CNN) model on the MNIST dataset, which consists of 20,000 
    training and 5,000 testing images of handwritten digits (0-9). The model achieved a validation accuracy of **97%**, 
    making it well-suited for real-time digit recognition tasks.

    - **Model Architecture**:
      - The CNN model consists of several convolutional layers followed by max-pooling and fully connected layers.
      - The model was trained using **Adam optimizer** and **categorical cross-entropy loss**.
    """)

    st.header("Model Deployment")
    st.write("""
    - The trained model is deployed within the Streamlit app, allowing for seamless integration of AI-powered digit recognition 
      and real-time feedback.
    """)

    # Features & Functionality
    st.header("Features & Functionality")
    st.write("""
    1. **Handwritten Digit Recognition**:
       - **Upload File**: Users upload an image of handwritten digits for prediction.
       - **Take Camera**: Users can use the built-in camera to capture real-time images of handwritten digits.
       - **Prediction**: The app displays the predicted digit with the model's confidence score.

    2. **Interactive Quiz**:
       - **Question Generation**: The app randomly generates simple math questions (e.g., "5 + 3", "8 - 4").
       - **Canvas Drawing**: Students can draw their answers (digits) directly on the canvas.
       - **Answer Evaluation**: The AI model predicts the drawn digit, compares it with the correct answer, and provides feedback.

    3. **Feedback and Visualization**:
       - After completing the quiz, the app provides feedback on whether the answers were correct or incorrect.
       - A **Matplotlib pie chart** visualizes the percentage of correct vs. incorrect answers.
       - A **DataFrame** summarizing the student's performance is displayed for review.
    """)

    # UI/UX Design
    st.header("UI/UX Design")
    st.write("""
    The application was designed to be intuitive and user-friendly, particularly for young learners. The interactive canvas 
    and simple layout ensure a seamless user experience. The design prioritizes ease of use, providing clear instructions 
    for each feature and immediate feedback on results.
    """)

    # Challenges Faced
    st.header("Challenges Faced")
    st.write("""
    - **Model Accuracy**: While the MNIST dataset is a standard benchmark for digit recognition, real-world handwriting can 
      be quite variable. I used **data augmentation** techniques such as rotation and scaling during training to improve the model's generalization.
    - **Real-time Prediction**: Integrating real-time camera input into the Streamlit app while ensuring a fast and responsive 
      prediction required careful optimization of the model inference pipeline.
    """)

    # Technologies Used
    st.header("Technologies Used")
    st.write("""
    - **Streamlit**: For building the web app and handling user interactions.
    - **TensorFlow**: For training and deploying the handwritten digit recognition model.
    - **Python**: For all backend operations, including data processing and model inference.
    - **Matplotlib**: For generating the pie chart and visualizing quiz results.
    - **NumPy and Pandas**: For handling data operations and organizing quiz results.
    """)

    # Future Enhancements
    st.header("Future Enhancements")
    st.write("""
    1. **Multi-Language Support**: Adding language support to make the app accessible to a global audience.
    2. **Improved Error Handling**: Handling edge cases such as unclear or noisy images more gracefully.
    3. **Model Explainability**: Implementing model explainability tools like Grad-CAM to provide users with insight into how the model makes predictions.
    4. **Mobile Optimization**: Further optimizing the UI for mobile devices to ensure seamless user experience on smartphones and tablets.
    """)

    # Conclusion
    st.header("Conclusion")
    st.write("""
    This project demonstrates the integration of machine learning, web development, and education technology. It showcases 
    how AI can be used in practical, real-world applications like helping students improve their math and handwriting skills. 
    The app is designed to be both fun and educational, making it ideal for young learners.
    """)
    
    # Social Media Hashtags
    st.write("""
    **#MachineLearning #AI #TensorFlow #Streamlit #Education #HandwrittenDigitRecognition #AIInEducation #Python #DataScience**
    """)
