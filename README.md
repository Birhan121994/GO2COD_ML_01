[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://digitquizai.streamlit.app)

# HDR App: Handwritten Digit Recognition and Adaptive Math Quiz

A Streamlit app that reads handwritten digits with a neural network and uses them to power an adaptive math quiz. It also includes tools to measure how well the model performs and to track how a learner is progressing.

## App Overview

This app allows users to:

- **Recognize handwritten digits** from an uploaded image or a camera photo, with a confidence score for every prediction.
- **Solve math problems** by drawing each answer on a canvas. The AI reads the digit, checks it and gives feedback.
- **Learn adaptively** with easy, medium and hard levels, hints, and targeted practice for repeated mistakes.
- **Track progress** with accuracy, response time and results by difficulty and question type.
- **Evaluate the model** with a confusion matrix, precision, recall, F1-score, example predictions, training curves and an analysis of the most confused digits.

## Navigation

The top bar is kept simple, with two dropdown menus:

| Menu item | Contents |
|---|---|
| **Home** | Overview of the app |
| **Recognize** (dropdown) | **Browse File** and **Camera Input** |
| **Math Quiz** | Adaptive quiz with hand-drawn answers |
| **Learning Analytics** | Personal progress and recommendations |
| **Model Evaluation** | Metrics, confusion matrix and error analysis |
| **Help** (dropdown) | **Demo** video and **Documentation** |

The bar is centered with equal spacing and wraps into evenly sized rows on tablets and phones.

## Features

### Digit recognition
- **Browse File**: upload a JPG or PNG of a handwritten digit and see the prediction.
- **Camera Input**: take a photo with your device camera and get the prediction.
- A confidence ring shows how sure the model is (green, amber or red) with a short explanation.

### Adaptive math quiz
- **Three modes**
  - **Fixed difficulty**: you pick easy, medium or hard and it stays there.
  - **Adaptive**: two clean correct answers in a row move you up a level, and two wrong answers in a row move you down. Answers that needed a hint do not count towards moving up.
  - **Targeted practice**: brings back questions you got wrong and favours your weakest question types, with automatic hints.
- **Difficulty levels**
  - **Easy**: addition and subtraction within 9.
  - **Medium**: adds multiplication and division.
  - **Hard**: bigger numbers and two-step problems, for example `4 × 3 − 7` or `(2 + 4) ÷ 3`.
- **Question types**: addition, subtraction, multiplication, division and multi-step. Every answer is a single digit (0-9), because the model reads one drawn digit.
- **Hints**: up to two hints per question. The first explains the method, the second narrows down the answer.
- **Rounds** of 5 or 10 questions, with a results screen that shows each answer, the correct answer and the time taken.

### Learning analytics
- Total questions attempted, correct and incorrect answers.
- Quiz accuracy and average response time.
- Progress by difficulty level and by question type.
- Trend charts for rolling accuracy and response time.
- A list of questions still to master, with how often each was missed.
- Recommendations for a suggested difficulty and weak areas to focus on.
- Download your results as CSV, or reset your progress.

> Progress is stored for the current browser session. Download the CSV to keep a copy.

### Model evaluation
- **Metrics**: accuracy, macro and weighted precision, recall and F1-score, plus a per-digit table.
- **Confusion matrix** for digits 0-9, shown as counts or as the share of each true digit.
- **Examples** of correctly and incorrectly recognized digits, filterable by digit. Mistakes can be sorted by model confidence.
- **Training curves**: training and validation accuracy and loss, with the best epoch and an overfitting check.
- **Common confusions**: the most frequent errors (for example `4` read as `9`), the most confusable digit pairs, and a comparison of confidence on right and wrong predictions.

## How to Use

1. **Recognize a digit**: open **Recognize** and choose **Browse File** or **Camera Input**. Upload or capture a digit and read the prediction.
2. **Play the quiz**: open **Math Quiz**, choose a mode, a difficulty and a round length, then press **Start round**. Draw each answer in the white box, use **Show a hint** if you need help, and press **Submit answer**.
3. **Review your progress**: open **Learning Analytics**. Use **Practice my weak spots** to jump straight into a targeted round.
4. **Evaluate the model**: open **Model Evaluation**, choose a data source, and press **Run evaluation** (see below).

## Evaluating the Model

The evaluation needs labelled images that the model has **not** been trained on. Choose one of three sources:

1. **Folder on disk** (default path `evaluation_data/`), with one sub-folder per digit:

   ```text
   evaluation_data/
   ├── 0/  img1.png  img2.png ...
   ├── 1/
   ├── ...
   └── 9/
   ```

2. **ZIP upload** with the same layout (each image inside a folder named after its digit).
3. **MNIST test set**, downloaded on first use. MNIST images are small and square, so they are stretched to the model's 90x140 input. Results are only meaningful if the model was trained on similar data.

Use **Max images per digit** to control how many images are tested.

### Training curves

Save the Keras training history after training so the curves can be drawn:

```python
history = model.fit(...)

import json
with open("OCR_MODEL/training_history.json", "w") as f:
    json.dump(history.history, f)
```

You can also use `CSVLogger` to write `OCR_MODEL/training_history.csv`, or upload a JSON or CSV file directly on the Model Evaluation page. The accepted file names are:

- `OCR_MODEL/training_history.json`
- `OCR_MODEL/history.json`
- `OCR_MODEL/training_history.csv`
- `OCR_MODEL/history.csv`

## AI Model Details

The app uses a **TensorFlow/Keras convolutional neural network** to recognize handwritten digits.

- **Input**: a 90x140 RGB image, normalised to the 0-1 range.
- **Output**: a probability for each digit from 0 to 9. The highest probability is the prediction, and its value is shown as the confidence.
- **Reported accuracy**: about **97%** on the validation set during training. Use the **Model Evaluation** page to measure accuracy, precision, recall and F1-score on your own test images.

### Image processing pipeline

1. The image is converted to a 3-channel RGB array.
2. It is resized to 90x140 pixels.
3. Pixel values are normalised to 0-1.
4. The CNN outputs probabilities for the ten digits.
5. The digit with the highest probability is selected.

## Project Structure

```text
.
├── app3.py                  # Streamlit app
├── assets/
│   ├── style.css            # Custom design system
│   └── crafto-landing-page-features-ico-05.png
├── OCR_MODEL/
│   ├── OCR_MODEL_1.h5       # Trained model
│   └── training_history.json  # Optional: enables training curves
├── OCR_DEMO/
│   └── OCR_APP_Demo.mkv     # Demo video
└── evaluation_data/         # Optional: labelled test images (0-9)
```

## Installation

```bash
git clone <your-repository-url>
cd <your-repository-folder>
pip install streamlit tensorflow opencv-python pillow numpy pandas matplotlib seaborn streamlit-drawable-canvas
streamlit run app3.py
```

Streamlit 1.39 or newer is recommended. The navigation bar styling and the dropdown highlight rely on features from that release.

## Built With

- **TensorFlow / Keras**: digit recognition model
- **Streamlit**: web interface
- **OpenCV and Pillow**: image processing
- **NumPy and Pandas**: data handling and metrics
- **Matplotlib and Seaborn**: confusion matrix and training curves
- **streamlit-drawable-canvas**: hand-drawn quiz answers

## Future Enhancements

- Multi-digit recognition
- Support for mathematical symbols
- User accounts and saved progress between sessions
- Mobile app version
- Additional quiz types
- Enhanced model explainability features