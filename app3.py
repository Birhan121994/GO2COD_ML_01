import copy
import io
import json
import os
import random
import time
import zipfile
from collections import Counter

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import seaborn as sns

# ===========================================================================
# Configuration
# ===========================================================================
MODEL_PATH = 'OCR_MODEL/OCR_MODEL_1.h5'
IMG_W, IMG_H = 90, 140                      # input size used during training

# Evaluation
EVAL_DATA_DIR = 'evaluation_data'           # evaluation_data/0 ... evaluation_data/9
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")
HISTORY_FILES = [
    'OCR_MODEL/training_history.json',
    'OCR_MODEL/history.json',
    'OCR_MODEL/training_history.csv',
    'OCR_MODEL/history.csv',
]

# Quiz
DIFFICULTIES = ["easy", "medium", "hard"]
MAX_RESPONSE_SECONDS = 120                  # cap so a long break does not skew averages

# App Configuration
im = Image.open('assets/crafto-landing-page-features-ico-05.png')
st.set_page_config(
    page_title="HDR App (Handwritten Digit Recognition App)",
    page_icon=im,
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css("assets/style.css")


# ===========================================================================
# UI helpers (presentation only, no model or quiz logic in here)
# ===========================================================================
def render(html):
    """Render an HTML snippet. Lines are stripped and joined so Markdown never
    mistakes indented HTML for a code block."""
    st.markdown("".join(line.strip() for line in html.splitlines()), unsafe_allow_html=True)


def page_head(title, subtitle):
    render(f"""
    <div class="page-head">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """)


def reading_card(digit, confidence, level, note, label="The model reads this as"):
    render(f"""
    <div class="reading {level}">
        <div class="ring-wrap">
            <div class="ring" style="--p:{min(confidence, 100):.1f}"></div>
            <div class="digit">{digit}</div>
        </div>
        <div>
            <div class="label">{label}</div>
            <div class="conf">{confidence:.1f}% confidence</div>
            <p class="note">{note}</p>
        </div>
    </div>
    """)


def empty_state(text):
    cells = "".join(f'<i class="{"on" if i in (1, 2, 3, 7, 12, 17, 22) else ""}"></i>' for i in range(25))
    render(f"""
    <div class="empty">
        <div class="mini">{cells}</div>
        <div>{text}</div>
    </div>
    """)


def confidence_note(confidence, hint):
    if confidence >= 70:
        return f"The model is {'very ' if confidence >= 85 else ''}confident in this prediction."
    return f"The model is less confident in this prediction. {hint}"


def stat_tiles(items):
    """items: list of (label, value, tone) where tone is '', 'ok', 'warn' or 'bad'."""
    html = "".join(
        f'<div class="stat {tone}"><div class="k">{label}</div><div class="v">{value}</div></div>'
        for label, value, tone in items
    )
    render(f'<div class="stats">{html}</div>')


def tone_for(pct):
    return "ok" if pct >= 80 else "warn" if pct >= 60 else "bad"


def level_for(pct):
    return "high" if pct >= 80 else "medium" if pct >= 60 else "low"


def callout(lines):
    items = "".join(f"<li>{line}</li>" for line in lines)
    render(f'<div class="callout"><ul>{items}</ul></div>')


ICON_UPLOAD = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M17 8l-5-5-5 5"/><path d="M12 3v12"/></svg>'
ICON_CAMERA = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>'
ICON_PENCIL = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>'
ICON_CHIP = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="5" width="14" height="14" rx="2"/><path d="M9 1v4M15 1v4M9 19v4M15 19v4M1 9h4M1 15h4M19 9h4M19 15h4"/></svg>'
ICON_CHECK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12l5 5L20 7"/></svg>'
ICON_CROSS = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 6l12 12M18 6L6 18"/></svg>'
ICON_CHART = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M7 15l4-5 3 3 5-7"/></svg>'
ICON_GRID = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>'

# A "7" drawn on a 7 x 9 pixel grid, used by the home page illustration
DIGIT_SEVEN = [
    "1111111",
    "1111111",
    "0000011",
    "0000110",
    "0001100",
    "0011000",
    "0011000",
    "0110000",
    "0110000",
]


def hero_reader():
    cells = []
    n = 0
    for row in DIGIT_SEVEN:
        for c in row:
            if c == "1":
                cells.append(f'<i class="px on" style="--d:{n * 28}ms"></i>')
                n += 1
            else:
                cells.append('<i class="px"></i>')
    heights = [4, 3, 6, 4, 3, 7, 4, 92, 6, 10]
    cols = "".join(
        f'<div class="col {"hit" if i == 7 else ""}"><div class="b" style="height:{h}%"></div>{i}</div>'
        for i, h in enumerate(heights)
    )
    return f"""
    <div class="reader">
        <div class="reader-bar"><i></i><i></i><i></i><span>Example reading</span></div>
        <div class="reader-body">
            <div class="px-grid">{"".join(cells)}</div>
            <div class="bars">{cols}</div>
        </div>
    </div>
    """


# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


model = load_model()

# Top bar: brand + horizontal navigation (replaces the sidebar)
render("""
<div class="topbar">
    <div class="brand">
        <div class="brand-mark"><i class="on"></i><i class="on"></i><i class="on"></i><i></i><i></i><i class="on"></i><i></i><i class="on"></i><i></i></div>
        <span class="brand-name">HDR App</span>
    </div>
    <span class="brand-sub">Handwritten digit recognition and math quiz</span>
</div>
""")

# ---------------------------------------------------------------------------
# Navigation: a few top-level items plus two dropdown menus
# ---------------------------------------------------------------------------
NAV_ITEMS = [
    {"label": "Home"},
    {"label": "Recognize", "key": "nav_recognize", "pages": ["Browse File", "Camera Input"]},
    {"label": "Math Quiz"},
    {"label": "Learning Analytics"},
    {"label": "Model Evaluation"},
    {"label": "Help", "key": "nav_help", "pages": ["Demo", "Documentation"]},
]

st.session_state.setdefault("page", "Home")


def go_to(page):
    st.session_state.page = page


def nav_popover(label, active):
    """Popover trigger that works on older and newer Streamlit versions."""
    kind = "primary" if active else "secondary"
    try:
        return st.popover(label, use_container_width=True, type=kind)
    except TypeError:                       # older Streamlit without `type`
        return st.popover(label, use_container_width=True)


def render_nav():
    active = st.session_state.page

    with st.container(key="navbar"):
        cols = st.columns(len(NAV_ITEMS), gap="small")
        for col, item in zip(cols, NAV_ITEMS):
            with col:
                if "pages" in item:
                    # Dropdown: shows the current page's name while one of its pages is open
                    is_active = active in item["pages"]
                    label = active if is_active else item["label"]
                    with nav_popover(label, is_active):
                        for page in item["pages"]:
                            st.button(
                                page,
                                key=f"{item['key']}_{page.lower().replace(' ', '_')}",
                                type="primary" if active == page else "secondary",
                                on_click=go_to,
                                args=(page,),
                                use_container_width=True,
                            )
                else:
                    st.button(
                        item["label"],
                        key="nav_" + item["label"].lower().replace(" ", "_"),
                        type="primary" if active == item["label"] else "secondary",
                        on_click=go_to,
                        args=(item["label"],),
                        use_container_width=True,
                    )


render_nav()
selected = st.session_state.page


# ===========================================================================
# Image + model helpers
# ===========================================================================
# Function to check if the image is blank
def is_blank_image(image, threshold=0.99):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    white_pixels = np.sum(gray_image >= 240)
    total_pixels = gray_image.size
    white_ratio = white_pixels / total_pixels
    return white_ratio > threshold


def to_rgb_array(image):
    """PIL image or array -> RGB array with 3 channels."""
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))

    img = np.asarray(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


def to_model_size(image):
    """Resize to the training size. Returns uint8 (140, 90, 3); handy for evaluation sets."""
    img = to_rgb_array(image)
    return cv2.resize(img, (IMG_W, IMG_H))


# Function to preprocess the image for prediction
def preprocess_image(image):
    img = to_model_size(image)
    img = img.astype(np.float32) / 255.0           # normalise to 0-1
    return np.expand_dims(img, axis=0)             # (1, 140, 90, 3)


def predict_with_confidence(image):
    """One forward pass -> (digit, confidence between 0 and 1)."""
    probs = model.predict(preprocess_image(image), verbose=0)[0]
    return int(np.argmax(probs)), float(np.max(probs))


# Kept for compatibility with the original app
def predict_digit(image):
    return predict_with_confidence(image)[0]


def prediction_proba(image):
    return predict_with_confidence(image)[1]


# ===========================================================================
# Quiz engine: question generation, difficulty, hints
# ===========================================================================
# Every answer is a single digit (0-9), because the model reads one drawn digit.
MUL_MEDIUM = [(a, b) for a in range(0, 5) for b in range(0, 5) if a * b <= 9]
MUL_HARD = [(a, b) for a in range(2, 10) for b in range(2, 10) if 4 <= a * b <= 9]
MUL_STEP = [(a, b) for a in range(2, 10) for b in range(2, 10) if 6 <= a * b <= 18]

TYPES_BY_LEVEL = {
    "easy": ["addition", "subtraction"],
    "medium": ["addition", "subtraction", "multiplication", "division"],
    "hard": ["addition", "subtraction", "multiplication", "division", "multi-step"],
}

MODE_OPTIONS = {
    "Fixed difficulty": "fixed",
    "Adaptive": "adaptive",
    "Targeted practice": "targeted",
}
MODE_LABELS = {v: k for k, v in MODE_OPTIONS.items()}


def _hint2(answer):
    parity = "even" if answer % 2 == 0 else "odd"
    return f"The answer is {parity} and between {max(0, answer - 2)} and {min(9, answer + 2)}."


def _build(text, answer, qtype, difficulty, hint1):
    return {
        "question": text,
        "answer": int(answer),
        "type": qtype,
        "difficulty": difficulty,
        "hint1": hint1,
        "hint2": _hint2(int(answer)),
    }


def _addition(d):
    if d == "hard":
        total = random.randint(5, 9)
        a = random.randint(1, total - 2)
        b = random.randint(1, total - a - 1)
        c = total - a - b
        return _build(f"{a} + {b} + {c}", total, "addition", d,
                      f"Add from left to right: first {a} + {b}, then add {c}.")
    if d == "medium":
        a = random.randint(1, 8)
        b = random.randint(1, 9 - a)
    else:
        a = random.randint(0, 5)
        b = random.randint(0, min(5, 9 - a))
    return _build(f"{a} + {b}", a + b, "addition", d, f"Start at {a} and count up {b} more.")


def _subtraction(d):
    if d == "hard":
        a = random.randint(10, 18)
        b = random.randint(a - 9, 9)
        return _build(f"{a} − {b}", a - b, "subtraction", d,
                      f"Ask yourself: what number plus {b} makes {a}?")
    if d == "medium":
        a = random.randint(5, 9)
        b = random.randint(1, a)
    else:
        a = random.randint(1, 9)
        b = random.randint(0, min(a, 5))
    return _build(f"{a} − {b}", a - b, "subtraction", d, f"Start at {a} and count back {b}.")


def _multiplication(d):
    a, b = random.choice(MUL_HARD if d == "hard" else MUL_MEDIUM)
    if a == 0 or b == 0:
        hint = "Anything multiplied by 0 is 0."
    else:
        hint = f"Multiplying is repeated adding: add {a} together {b} times."
    return _build(f"{a} × {b}", a * b, "multiplication", d, hint)


def _division(d):
    if d == "hard":
        b, q = random.randint(2, 9), random.randint(2, 9)
    else:
        b, q = random.randint(2, 5), random.randint(1, 5)
    a = b * q
    return _build(f"{a} ÷ {b}", q, "division", d, f"Ask yourself: {b} times what number makes {a}?")


def _multistep(d):
    pattern = random.choice(["add_sub", "mul_sub", "sub_mul", "add_div"])
    if pattern == "add_sub":
        a, b = random.randint(2, 9), random.randint(2, 9)
        c = random.randint(max(1, a + b - 9), a + b)
        return _build(f"{a} + {b} − {c}", a + b - c, "multi-step", d,
                      f"Work left to right: first {a} + {b} = {a + b}, then take away {c}.")
    if pattern == "mul_sub":
        a, b = random.choice(MUL_STEP)
        p = a * b
        c = random.randint(max(1, p - 9), min(9, p))
        return _build(f"{a} × {b} − {c}", p - c, "multi-step", d,
                      f"Multiply first: {a} × {b} = {p}. Then take away {c}.")
    if pattern == "sub_mul":
        b, c = random.randint(2, 4), random.randint(2, 4)
        p = b * c
        a = random.randint(p, p + 9)
        return _build(f"{a} − {b} × {c}", a - p, "multi-step", d,
                      f"Multiplication comes first: {b} × {c} = {p}. Then subtract that from {a}.")
    c, q = random.randint(2, 5), random.randint(1, 6)
    total = c * q
    a = random.randint(1, total - 1)
    b = total - a
    return _build(f"({a} + {b}) ÷ {c}", q, "multi-step", d,
                  f"Do the brackets first: {a} + {b} = {total}. Then divide by {c}.")


QUESTION_BUILDERS = {
    "addition": _addition,
    "subtraction": _subtraction,
    "multiplication": _multiplication,
    "division": _division,
    "multi-step": _multistep,
}


def make_question(difficulty, qtype):
    return QUESTION_BUILDERS[qtype](difficulty)


# ===========================================================================
# Session state
# ===========================================================================
DEFAULT_STATE = {
    "quiz_stage": "setup",          # setup | running | done
    "quiz_mode": "fixed",
    "level_idx": 0,
    "round_len": 5,
    "round_id": 0,
    "round_attempts": [],
    "round_questions": [],
    "current_question": None,
    "current_question_index": 0,
    "question_started_at": None,
    "hint_level": 0,
    "streak": 0,
    "history": [],                  # every attempt across rounds
    "mistakes": {},                 # question text -> {question, misses, last_given}
    "last_feedback": None,
    "balloons_shown": False,
    "pending_mode": None,
    "eval_result": None,
    "eval_seed": 0,
}
for _key, _value in DEFAULT_STATE.items():
    if _key not in st.session_state:
        st.session_state[_key] = copy.deepcopy(_value)

HISTORY_COLUMNS = ["round", "mode", "question", "type", "difficulty", "expected", "given",
                   "correct", "time_s", "hints", "confidence", "retry"]


def history_df():
    return pd.DataFrame(st.session_state.history, columns=HISTORY_COLUMNS)


def weak_types(min_attempts=2, threshold=0.7):
    """Question types the learner gets wrong most often (lowest accuracy first)."""
    df = history_df()
    if df.empty:
        return []
    g = df.groupby("type")["correct"].agg(["mean", "count"])
    g = g[(g["count"] >= min_attempts) & (g["mean"] < threshold)].sort_values("mean")
    return list(g.index)


def recommended_level_idx():
    recent = st.session_state.history[-10:]
    if len(recent) < 3:
        return DIFFICULTIES.index(recent[-1]["difficulty"]) if recent else 0
    acc = sum(a["correct"] for a in recent) / len(recent)
    last = DIFFICULTIES.index(recent[-1]["difficulty"])
    if acc >= 0.8:
        return min(last + 1, 2)
    if acc < 0.5:
        return max(last - 1, 0)
    return last


def next_question():
    s = st.session_state
    level = DIFFICULTIES[s.level_idx]
    asked = set(s.round_questions)
    mode = s.quiz_mode

    # Targeted practice: bring back questions that were answered wrongly before
    if mode == "targeted":
        pool = [m for key, m in s.mistakes.items() if key not in asked]
        if pool and random.random() < 0.6:
            entry = random.choice(pool)
            q = dict(entry["question"])
            q["retry"] = True
            q["auto_hint"] = entry["misses"] >= 2
            return q

    available = TYPES_BY_LEVEL[level]
    weak_here = [t for t in weak_types() if t in available]
    focus_chance = 0.7 if mode == "targeted" else 0.4 if mode == "adaptive" else 0.0

    focused = bool(weak_here) and random.random() < focus_chance
    qtype = random.choice(weak_here) if focused else random.choice(available)

    q = make_question(level, qtype)
    for _ in range(15):
        if q["question"] not in asked:
            break
        q = make_question(level, qtype)

    q["focus"] = focused
    q["auto_hint"] = focused and mode == "targeted"
    return q


def start_round(mode, level_idx, length):
    s = st.session_state
    s.quiz_mode = mode
    s.level_idx = level_idx
    s.round_len = length
    s.round_id += 1
    s.round_attempts = []
    s.round_questions = []
    s.current_question = None
    s.current_question_index = 0
    s.streak = 0
    s.hint_level = 0
    s.last_feedback = None
    s.balloons_shown = False
    s.quiz_stage = "running"


def start_from_setup():
    s = st.session_state
    start_round(MODE_OPTIONS[s.setup_mode], DIFFICULTIES.index(s.setup_level.lower()), int(s.setup_len))


def replay_round():
    s = st.session_state
    start_round(s.quiz_mode, s.level_idx, s.round_len)


def targeted_round():
    s = st.session_state
    start_round("targeted", s.level_idx, s.round_len)


def back_to_setup():
    st.session_state.quiz_stage = "setup"


def show_next_hint():
    st.session_state.hint_level = min(st.session_state.hint_level + 1, 2)


def request_targeted_setup():
    st.session_state.pending_mode = "Targeted practice"
    st.session_state.quiz_stage = "setup"
    st.session_state.page = "Math Quiz"


def reset_progress():
    s = st.session_state
    s.history = []
    s.mistakes = {}
    s.streak = 0
    s.last_feedback = None


def submit_answer(q, predicted, confidence):
    s = st.session_state
    started = s.question_started_at or time.time()
    elapsed = min(time.time() - started, MAX_RESPONSE_SECONDS)
    correct = int(predicted) == q["answer"]

    attempt = {
        "round": s.round_id,
        "mode": s.quiz_mode,
        "question": q["question"],
        "type": q["type"],
        "difficulty": q["difficulty"],
        "expected": q["answer"],
        "given": int(predicted),
        "correct": correct,
        "time_s": round(elapsed, 1),
        "hints": s.hint_level,
        "confidence": round(confidence, 1),
        "retry": bool(q.get("retry")),
    }
    s.history.append(attempt)
    s.round_attempts.append(attempt)

    # Track mistakes so they can be practised again
    key = q["question"]
    if correct:
        if key in s.mistakes and s.hint_level < 2:
            del s.mistakes[key]
    else:
        base = {k: q[k] for k in ("question", "answer", "type", "difficulty", "hint1", "hint2")}
        entry = s.mistakes.setdefault(key, {"question": base, "misses": 0, "last_given": None})
        entry["misses"] += 1
        entry["last_given"] = int(predicted)

    # Adaptive difficulty: two clean answers in a row go up, two misses in a row go down
    level_note = ""
    if s.quiz_mode in ("adaptive", "targeted"):
        if correct and s.hint_level == 0:
            s.streak = s.streak + 1 if s.streak > 0 else 1
        elif not correct:
            s.streak = s.streak - 1 if s.streak < 0 else -1
        else:
            s.streak = 0
        if s.streak >= 2:
            s.streak = 0
            if s.level_idx < 2:
                s.level_idx += 1
                level_note = f" Moving up to {DIFFICULTIES[s.level_idx]}."
        elif s.streak <= -2:
            s.streak = 0
            if s.level_idx > 0:
                s.level_idx -= 1
                level_note = f" Let's try {DIFFICULTIES[s.level_idx]} questions for a bit."

    if correct:
        text = f"Correct! {q['question']} = {q['answer']}.{level_note}"
    else:
        text = f"Not quite: {q['question']} = {q['answer']}. The model read your answer as {int(predicted)}.{level_note}"

    s.last_feedback = {
        "ok": correct,
        "text": text,
        "round": s.round_id,
        "index": s.current_question_index,
    }

    s.current_question_index += 1
    s.current_question = None
    if s.current_question_index >= s.round_len:
        s.quiz_stage = "done"


# Function to display a question and capture user input via canvas
def ask_question():
    s = st.session_state

    if s.current_question is None:
        q = next_question()
        s.current_question = q
        s.round_questions.append(q["question"])
        s.question_started_at = time.time()
        s.hint_level = 1 if q.get("auto_hint") else 0

    q = s.current_question
    attempts = s.round_attempts
    current = s.current_question_index
    total = s.round_len

    # Step tracker: one dot per question, coloured by outcome
    dots = []
    for i in range(total):
        if i < len(attempts):
            state = "ok" if attempts[i]["correct"] else "bad"
        elif i == current:
            state = "current"
        else:
            state = ""
        dots.append(f'<span class="dot {state}">{i + 1}</span>')

    score = sum(a["correct"] for a in attempts)
    render(f"""
    <div class="quiz-bar">
        <div class="dots">{"".join(dots)}</div>
        <div class="score">Score <strong>{score}</strong> / {total}</div>
    </div>
    """)

    # Feedback from the previous question
    fb = s.last_feedback
    if fb and fb["round"] == s.round_id and fb["index"] == current - 1:
        render(f'<div class="feedback {"ok" if fb["ok"] else "bad"}">{fb["text"]}</div>')

    tags = [f'<span class="pill {q["difficulty"]}">{q["difficulty"]}</span>',
            f'<span class="tag">{q["type"]}</span>']
    if s.quiz_mode != "fixed":
        tags.append(f'<span class="tag">{MODE_LABELS[s.quiz_mode].lower()}</span>')
    if q.get("retry"):
        tags.append('<span class="tag retry">Retry: you missed this before</span>')
    elif q.get("focus"):
        tags.append('<span class="tag retry">Focus area</span>')

    render(f"""
    <div class="worksheet">
        <div class="tags">{"".join(tags)}</div>
        <p class="ask">Question {current + 1}: what is the result of</p>
        <span class="expr">{q["question"]} = ?</span>
    </div>
    """)

    col1, col2 = st.columns([1.15, 1], gap="large")

    with col1:
        with st.container(border=True):
            st.caption("Draw one digit (0-9) in the white box")
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0)",
                stroke_width=15,
                stroke_color="#000000",
                background_color="#ffffff",
                height=300,
                width=400,
                drawing_mode="freedraw",
                # One canvas per question (and per round) so each answer starts blank
                key=f"canvas_{s.round_id}_{current}",
                return_image_data=True,
            )

    with col2:
        predicted_answer = None
        confidence = 0.0
        if canvas_result.image_data is not None:
            drawn = Image.fromarray(canvas_result.image_data).convert("RGB")
            drawn = np.array(drawn)

            if not is_blank_image(drawn):
                predicted_answer, conf = predict_with_confidence(drawn)
                confidence = conf * 100

                level = 'high' if confidence > 85 else 'medium' if confidence > 70 else 'low'
                reading_card(
                    predicted_answer, confidence, level,
                    confidence_note(confidence, "Try drawing more clearly.")
                )
            else:
                empty_state("Your drawing will be read here.")

        # Hints
        if s.hint_level >= 1:
            title = "Targeted tip" if q.get("auto_hint") and s.hint_level == 1 else "Hint"
            render(f'<div class="hint-box"><strong>{title}</strong>{q["hint1"]}</div>')
        if s.hint_level >= 2:
            render(f'<div class="hint-box"><strong>Another hint</strong>{q["hint2"]}</div>')
        if s.hint_level < 2:
            st.button(
                "Show a hint" if s.hint_level == 0 else "Show another hint",
                on_click=show_next_hint,
                use_container_width=True,
            )

        render("""
        <div class="tips">
            <span>Make the digit large</span>
            <span>Keep it clear</span>
            <span>Submit when ready</span>
        </div>
        """)

        if st.button("Submit answer", use_container_width=True, type="primary"):
            if predicted_answer is not None:
                submit_answer(q, predicted_answer, confidence)
                st.rerun()
            else:
                st.warning("Please draw your answer before submitting.")


def render_setup():
    s = st.session_state
    page_head("Math quiz", "Pick a mode and a difficulty, then draw each answer by hand.")

    # Allow other pages to preselect a mode (widget keys are cleared when a page is left)
    if s.pending_mode:
        s.setup_mode = s.pending_mode
        s.pending_mode = None
    s.setdefault("setup_mode", "Fixed difficulty")
    s.setdefault("setup_level", DIFFICULTIES[recommended_level_idx()].title())
    s.setdefault("setup_len", 5)

    if s.history:
        recent = s.history[-10:]
        acc = sum(a["correct"] for a in recent) / len(recent) * 100
        rec = DIFFICULTIES[recommended_level_idx()]
        callout([f"Based on your last {len(recent)} answers ({acc:.0f}% correct), "
                 f"we suggest starting on <strong>{rec}</strong>."])

    with st.container(border=True):
        st.radio(
            "Mode",
            list(MODE_OPTIONS.keys()),
            key="setup_mode",
            horizontal=True,
            captions=[
                "You choose the level and it stays put.",
                "Gets harder when you do well, easier when you struggle.",
                "Revisits your mistakes and weak question types, with extra hints.",
            ],
        )
        st.radio(
            "Difficulty" if MODE_OPTIONS[s.setup_mode] == "fixed" else "Starting difficulty",
            ["Easy", "Medium", "Hard"],
            key="setup_level",
            horizontal=True,
        )
        st.radio("Questions per round", [5, 10], key="setup_len", horizontal=True)

        render("""
        <div class="tips">
            <span><b>Easy</b>: add and subtract within 9</span>
            <span><b>Medium</b>: adds multiplication and division</span>
            <span><b>Hard</b>: bigger numbers and two-step problems</span>
        </div>
        """)

        if MODE_OPTIONS[s.setup_mode] == "targeted" and not s.mistakes and not weak_types():
            st.caption("No mistakes recorded yet, so targeted practice will behave like adaptive mode "
                       "until you have a few wrong answers to work on.")

        st.button("Start round", type="primary", use_container_width=True, on_click=start_from_setup)


def render_round_report():
    s = st.session_state
    attempts = s.round_attempts
    n = max(len(attempts), 1)
    score = sum(a["correct"] for a in attempts)
    pct = score / n * 100
    avg_time = sum(a["time_s"] for a in attempts) / n
    hints_used = sum(1 for a in attempts if a["hints"] > 0)

    page_head("Round complete", "Here is how you did.")

    if score == len(attempts):
        score_title, score_text = "Perfect score", "You answered every question correctly."
    elif pct >= 60:
        score_title, score_text = "Great job", "You did well, but there's room for improvement."
    else:
        score_title, score_text = "Keep practicing", "You'll do better next time."

    rows_html = ""
    for a in attempts:
        rows_html += f"""
        <div class="row">
            <span class="mark {'ok' if a['correct'] else 'bad'}">{ICON_CHECK if a['correct'] else ICON_CROSS}</span>
            <span class="q">{a["question"]}</span>
            <span>{a["given"]}</span>
            <span>{a["expected"]}</span>
            <span>{a["time_s"]:.1f}s</span>
        </div>
        """

    render(f"""
    <div class="report">
        <div class="report-score">
            <div class="big">{score}<small>/{len(attempts)}</small></div>
            <h2>{score_title}</h2>
            <p>{score_text}</p>
            <div class="split"><span style="width:{pct:.0f}%"></span></div>
        </div>
        <div class="rows">
            <div class="head"><span></span><span>Question</span><span>Your answer</span><span>Correct</span><span>Time</span></div>
            {rows_html}
        </div>
    </div>
    """)

    stat_tiles([
        ("Accuracy", f"{pct:.0f}%", tone_for(pct)),
        ("Average response time", f"{avg_time:.1f}<small> s</small>", ""),
        ("Hints used", f"{hints_used}<small> / {len(attempts)}</small>", ""),
        ("Next level", DIFFICULTIES[s.level_idx].title() if s.quiz_mode != "fixed" else "Fixed", ""),
    ])

    missed = [a for a in attempts if not a["correct"]]
    if missed:
        by_type = Counter(a["type"] for a in missed).most_common()
        worst, count = by_type[0]
        callout([
            f"You missed {len(missed)} question{'s' if len(missed) != 1 else ''}. "
            f"Most were <strong>{worst}</strong> ({count}).",
            "Targeted practice brings those questions back with extra hints until you've got them.",
        ])
    elif s.level_idx < 2:
        callout(["No mistakes this round. Try adaptive mode or a harder level for more of a challenge."])

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.button("Play again", use_container_width=True, type="primary", on_click=replay_round)
    with c2:
        st.button("Targeted practice", use_container_width=True, on_click=targeted_round,
                  disabled=not (s.mistakes or weak_types()))
    with c3:
        st.button("Change settings", use_container_width=True, on_click=back_to_setup)

    if score == len(attempts) and not s.balloons_shown:
        s.balloons_shown = True
        st.balloons()


# ===========================================================================
# Learning analytics page
# ===========================================================================
def meter_rows(df, key, order):
    rows = ""
    for name in order:
        sub = df[df[key] == name]
        if sub.empty:
            rows += (f'<div class="meter"><span class="m-name">{name}</span>'
                     f'<div class="track"></div><span class="m-val">not tried yet</span></div>')
            continue
        acc = sub["correct"].mean() * 100
        rows += (
            f'<div class="meter {level_for(acc)}"><span class="m-name">{name}</span>'
            f'<div class="track"><span class="fill" style="width:{acc:.0f}%"></span></div>'
            f'<span class="m-val">{acc:.0f}% · {len(sub)} tries · {sub["time_s"].mean():.1f}s</span></div>'
        )
    return f'<div class="meters">{rows}</div>'


def render_analytics():
    s = st.session_state
    page_head("Learning analytics", "See how your answers, speed and difficulty level are progressing.")

    df = history_df()
    if df.empty:
        empty_state("Play a round of the math quiz and your progress will show up here.")
        return

    total = len(df)
    correct = int(df["correct"].sum())
    acc = correct / total * 100
    stat_tiles([
        ("Questions attempted", f"{total}", ""),
        ("Correct", f"{correct}", "ok"),
        ("Incorrect", f"{total - correct}", "bad" if total - correct else ""),
        ("Accuracy", f"{acc:.0f}%", tone_for(acc)),
        ("Avg response time", f"{df['time_s'].mean():.1f}<small> s</small>", ""),
        ("Answered with hints", f"{int((df['hints'] > 0).sum())}", ""),
    ])

    c1, c2 = st.columns(2, gap="large")
    with c1:
        render('<h2 class="section-title">Progress by difficulty</h2>')
        render(meter_rows(df, "difficulty", DIFFICULTIES))
    with c2:
        render('<h2 class="section-title">Progress by question type</h2>')
        render(meter_rows(df, "type", list(QUESTION_BUILDERS.keys())))

    if total >= 3:
        render('<h2 class="section-title" style="margin-top:2rem">Trend</h2>')
        t1, t2 = st.columns(2, gap="large")
        trend = pd.DataFrame({
            "Attempt": range(1, total + 1),
            "Rolling accuracy (%)": df["correct"].astype(int).rolling(5, min_periods=1).mean().values * 100,
            "Response time (s)": df["time_s"].values,
        }).set_index("Attempt")
        with t1:
            st.caption("Accuracy over the last 5 answers")
            st.line_chart(trend[["Rolling accuracy (%)"]])
        with t2:
            st.caption("Response time per question (seconds)")
            st.line_chart(trend[["Response time (s)"]])

    # Recommendations
    tips = []
    weak = weak_types()
    if weak:
        tips.append(f"Your weakest area is <strong>{weak[0]}</strong>. "
                    f"Targeted practice will give you more of it, with hints.")
    rec = DIFFICULTIES[recommended_level_idx()]
    tips.append(f"Suggested difficulty for your next round: <strong>{rec}</strong>.")
    if s.mistakes:
        tips.append(f"You have <strong>{len(s.mistakes)}</strong> question"
                    f"{'s' if len(s.mistakes) != 1 else ''} still to master.")
    render('<h2 class="section-title" style="margin-top:2rem">Recommendations</h2>')
    callout(tips)

    # Mistakes to revisit
    if s.mistakes:
        render('<h2 class="section-title" style="margin-top:2rem">Questions to revisit</h2>')
        rows = [{
            "Question": m["question"]["question"],
            "Type": m["question"]["type"],
            "Level": m["question"]["difficulty"],
            "Times missed": m["misses"],
            "Your last answer": m["last_given"],
            "Correct answer": m["question"]["answer"],
        } for m in s.mistakes.values()]
        st.dataframe(pd.DataFrame(rows).sort_values("Times missed", ascending=False), hide_index=True)

    b1, b2, b3 = st.columns(3, gap="medium")
    with b1:
        st.button("Practice my weak spots", type="primary", use_container_width=True,
                  on_click=request_targeted_setup)
    with b2:
        st.download_button("Download my results (CSV)", df.to_csv(index=False).encode("utf-8"),
                           file_name="quiz_results.csv", mime="text/csv", use_container_width=True)
    with b3:
        st.button("Reset progress", use_container_width=True, on_click=reset_progress)

    st.caption("Progress is kept for this browser session. Download the CSV to keep a copy.")


# ===========================================================================
# Model evaluation
# ===========================================================================
MUTED = "#8b8f98"
INK = "#3b5bfd"
AMBER = "#e8590c"


def themed_axes(ncols=1, width=8, height=4):
    """Transparent matplotlib figure with mid-grey text, readable in light and dark themes."""
    fig, axes = plt.subplots(1, ncols, figsize=(width, height))
    fig.patch.set_alpha(0)
    axes = np.atleast_1d(axes)
    for ax in axes:
        ax.set_facecolor("none")
        ax.tick_params(colors=MUTED)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(MUTED)
        for spine in ax.spines.values():
            spine.set_color(MUTED)
            spine.set_alpha(0.4)
    return fig, axes


def load_folder_dataset(path, max_per_class):
    X, y = [], []
    for digit in range(10):
        folder = os.path.join(path, str(digit))
        if not os.path.isdir(folder):
            continue
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))
        random.Random(42).shuffle(files)
        for name in files[:max_per_class]:
            try:
                with Image.open(os.path.join(folder, name)) as img:
                    arr = to_model_size(img)
            except Exception:
                continue
            X.append(arr)
            y.append(digit)
    return np.array(X, dtype=np.uint8), np.array(y, dtype=int)


def load_zip_dataset(data, max_per_class):
    X, y = [], []
    counts = Counter()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [n for n in zf.namelist()
                 if n.lower().endswith(IMG_EXT) and not n.startswith("__MACOSX")]
        random.Random(42).shuffle(names)
        for name in names:
            parts = name.replace("\\", "/").split("/")
            if len(parts) < 2 or parts[-2] not in list("0123456789"):
                continue
            digit = int(parts[-2])
            if counts[digit] >= max_per_class:
                continue
            try:
                with Image.open(io.BytesIO(zf.read(name))) as img:
                    arr = to_model_size(img)
            except Exception:
                continue
            X.append(arr)
            y.append(digit)
            counts[digit] += 1
    return np.array(X, dtype=np.uint8), np.array(y, dtype=int)


def load_mnist_dataset(max_per_class, invert):
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    X, y = [], []
    counts = Counter()
    for img, label in zip(x_test, y_test):
        label = int(label)
        if counts[label] >= max_per_class:
            continue
        gray = 255 - img if invert else img
        X.append(to_model_size(gray))
        y.append(label)
        counts[label] += 1
    return np.array(X, dtype=np.uint8), np.array(y, dtype=int)


def predict_in_batches(X, batch=64):
    bar = st.progress(0.0, text="Running the model...")
    chunks = []
    for i in range(0, len(X), batch):
        chunks.append(model.predict(X[i:i + batch].astype(np.float32) / 255.0, verbose=0))
        bar.progress(min((i + batch) / len(X), 1.0), text="Running the model...")
    bar.empty()
    return np.vstack(chunks)


def confusion_matrix_np(y_true, y_pred, n=10):
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_metrics(cm):
    tp = np.diag(cm).astype(float)
    pred_total = cm.sum(axis=0).astype(float)
    true_total = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, pred_total, out=np.zeros_like(tp), where=pred_total > 0)
    recall = np.divide(tp, true_total, out=np.zeros_like(tp), where=true_total > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return pd.DataFrame({
        "Digit": list(range(10)),
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Support": true_total.astype(int),
    })


def normalise_history(df):
    aliases = {
        "accuracy": ["accuracy", "acc", "sparse_categorical_accuracy", "categorical_accuracy"],
        "val_accuracy": ["val_accuracy", "val_acc", "val_sparse_categorical_accuracy",
                         "val_categorical_accuracy"],
        "loss": ["loss"],
        "val_loss": ["val_loss"],
    }
    out = pd.DataFrame()
    for target, names in aliases.items():
        for name in names:
            if name in df.columns:
                out[target] = pd.to_numeric(df[name], errors="coerce")
                break
    return out if not out.empty else None


def read_history(name, data):
    if name.lower().endswith(".json"):
        raw = json.loads(data.decode("utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("history"), dict):
            raw = raw["history"]
        df = pd.DataFrame(raw)
    else:
        df = pd.read_csv(io.BytesIO(data))
    return normalise_history(df)


def find_history_file():
    for path in HISTORY_FILES:
        if os.path.exists(path):
            return path
    return None


def image_grid(res, indices, show_truth, cols=6):
    for start in range(0, len(indices), cols):
        columns = st.columns(cols)
        for col, i in zip(columns, indices[start:start + cols]):
            true, pred, conf = int(res["y"][i]), int(res["pred"][i]), res["conf"][i] * 100
            caption = f"True {true} → read {pred} ({conf:.0f}%)" if show_truth else f"{true} ({conf:.0f}%)"
            with col:
                st.image(res["X"][i], caption=caption, width="stretch")


def eval_overview(res, cm, metrics):
    valid = metrics["Support"] > 0
    acc = np.trace(cm) / cm.sum() * 100
    macro = metrics.loc[valid, ["Precision", "Recall", "F1-score"]].mean()
    weighted = {
        c: np.average(metrics.loc[valid, c], weights=metrics.loc[valid, "Support"])
        for c in ["Precision", "Recall", "F1-score"]
    }

    stat_tiles([
        ("Accuracy", f"{acc:.1f}%", tone_for(acc)),
        ("Macro precision", f"{macro['Precision'] * 100:.1f}%", ""),
        ("Macro recall", f"{macro['Recall'] * 100:.1f}%", ""),
        ("Macro F1-score", f"{macro['F1-score'] * 100:.1f}%", ""),
        ("Images tested", f"{len(res['y'])}", ""),
    ])

    table = metrics.copy()
    table["Digit"] = table["Digit"].astype(str)
    extra = pd.DataFrame([
        {"Digit": "Macro avg", **macro.to_dict(), "Support": int(metrics["Support"].sum())},
        {"Digit": "Weighted avg", **weighted, "Support": int(metrics["Support"].sum())},
    ])
    table = pd.concat([table, extra], ignore_index=True)

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.dataframe(
            table.style.format({"Precision": "{:.1%}", "Recall": "{:.1%}", "F1-score": "{:.1%}"}),
            hide_index=True,
        )
        st.caption("Precision: of the images read as this digit, how many were right. "
                   "Recall: of the real images of this digit, how many were found. "
                   "F1-score balances the two.")
    with right:
        st.caption("F1-score by digit")
        st.bar_chart(metrics.set_index(metrics["Digit"].astype(str))["F1-score"])

    if valid.any():
        worst = metrics[valid].nsmallest(1, "F1-score").iloc[0]
        best = metrics[valid].nlargest(1, "F1-score").iloc[0]
        callout([
            f"Strongest digit: <strong>{int(best['Digit'])}</strong> (F1 {best['F1-score']:.1%}).",
            f"Weakest digit: <strong>{int(worst['Digit'])}</strong> (F1 {worst['F1-score']:.1%}).",
        ])
    missing = [int(d) for d in metrics.loc[~valid, "Digit"]]
    if missing:
        st.warning(f"No test images were found for digit(s) {', '.join(map(str, missing))}.")


def eval_confusion(cm):
    mode = st.radio("Show", ["Counts", "Share of each true digit"], horizontal=True, key="cm_mode")
    if mode == "Counts":
        data, fmt = cm, "d"
    else:
        sums = cm.sum(axis=1, keepdims=True)
        data = np.divide(cm, sums, out=np.zeros_like(cm, dtype=float), where=sums > 0)
        fmt = ".0%"

    cmap = LinearSegmentedColormap.from_list("ink", ["#f1f4ff", INK])
    fig, (ax,) = themed_axes(1, 7, 6)
    sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap, cbar=False, linewidths=0.6,
                linecolor="#8b8f9833", square=True, ax=ax,
                xticklabels=range(10), yticklabels=range(10), annot_kws={"size": 10})
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.tick_params(colors=MUTED)
    st.pyplot(fig, width="content")
    plt.close(fig)
    st.caption("Rows are the real digit, columns are what the model predicted. "
               "Bright cells off the diagonal are mistakes.")


def eval_examples(res):
    c1, c2, c3 = st.columns([1, 1, 1], gap="medium")
    with c1:
        digit = st.selectbox("Digit", ["All"] + list(range(10)), key="ex_digit")
    with c2:
        count = st.slider("Examples to show", 6, 24, 12, 6, key="ex_count")
    with c3:
        order = st.radio("Mistakes order", ["Most confident first", "Random"], key="ex_order")
        st.button("Shuffle", on_click=lambda: st.session_state.update(eval_seed=st.session_state.eval_seed + 1))

    y, pred, conf = res["y"], res["pred"], res["conf"]
    mask = np.ones(len(y), dtype=bool) if digit == "All" else (y == digit)
    right = np.where(mask & (y == pred))[0]
    wrong = np.where(mask & (y != pred))[0]
    rng = np.random.default_rng(st.session_state.eval_seed)

    render('<h2 class="section-title" style="margin-top:1rem">Recognised correctly</h2>')
    if len(right):
        picks = rng.choice(right, size=min(count, len(right)), replace=False)
        image_grid(res, list(picks), show_truth=False)
    else:
        st.info("No correct predictions for this selection.")

    render('<h2 class="section-title" style="margin-top:1.5rem">Recognised incorrectly</h2>')
    if len(wrong):
        if order == "Most confident first":
            picks = wrong[np.argsort(-conf[wrong])][:count]
        else:
            picks = rng.choice(wrong, size=min(count, len(wrong)), replace=False)
        image_grid(res, [int(i) for i in picks], show_truth=True)
        st.caption("Captions show the true digit, the model's reading and its confidence. "
                   "A confident mistake usually means the image is ambiguous or unlike the training data.")
    else:
        st.success("No mistakes for this selection.")


def eval_curves():
    path = find_history_file()
    upload = st.file_uploader("Upload a training history (JSON or CSV)", type=["json", "csv"],
                              key="hist_upload")

    hist, source = None, None
    try:
        if upload is not None:
            hist, source = read_history(upload.name, upload.getvalue()), upload.name
        elif path:
            with open(path, "rb") as f:
                hist, source = read_history(path, f.read()), path
    except Exception as e:
        st.error(f"Could not read the history file: {e}")

    with st.expander("How do I save my training history?"):
        st.code(
            "history = model.fit(...)\n"
            "import json\n"
            "with open('OCR_MODEL/training_history.json', 'w') as f:\n"
            "    json.dump(history.history, f)\n\n"
            "# or log it while training:\n"
            "# from tensorflow.keras.callbacks import CSVLogger\n"
            "# model.fit(..., callbacks=[CSVLogger('OCR_MODEL/training_history.csv')])",
            language="python",
        )

    if hist is None or hist.empty:
        empty_state("No training history found. Upload a file or save one next to the model.")
        return

    st.caption(f"Loaded from {source}")
    epochs = np.arange(1, len(hist) + 1)

    fig, axes = themed_axes(2, 11, 3.8)
    panels = [("Accuracy", "accuracy", "val_accuracy", np.nanargmax),
              ("Loss", "loss", "val_loss", np.nanargmin)]
    for ax, (title, train_col, val_col, best_fn) in zip(axes, panels):
        if train_col in hist:
            ax.plot(epochs, hist[train_col], color=INK, lw=2.2, label="Training")
        if val_col in hist:
            ax.plot(epochs, hist[val_col], color=AMBER, lw=2.2, label="Validation")
            best = int(best_fn(hist[val_col].values))
            ax.axvline(best + 1, color=MUTED, ls="--", lw=1, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.15, color=MUTED)
        ax.legend(frameon=False, labelcolor=MUTED)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption("The dashed line marks the best validation epoch.")

    tiles, notes = [], []
    if "val_accuracy" in hist:
        best = int(np.nanargmax(hist["val_accuracy"].values))
        tiles.append(("Best validation accuracy", f"{hist['val_accuracy'].iloc[best] * 100:.1f}%", "ok"))
        tiles.append(("Best epoch", f"{best + 1}<small> / {len(hist)}</small>", ""))
    if "accuracy" in hist and "val_accuracy" in hist:
        gap = (hist["accuracy"].iloc[-1] - hist["val_accuracy"].iloc[-1]) * 100
        tiles.append(("Final train - val gap", f"{gap:.1f}<small> pts</small>",
                      "bad" if gap > 5 else "warn" if gap > 2 else "ok"))
        if gap > 5:
            notes.append("Training accuracy is well above validation accuracy, which suggests overfitting. "
                         "Dropout, data augmentation or early stopping can help.")
        elif gap < 2:
            notes.append("Training and validation accuracy are close, so the model generalises well.")
    if "val_loss" in hist:
        best_loss = int(np.nanargmin(hist["val_loss"].values))
        if best_loss < len(hist) - 3:
            notes.append(f"Validation loss was lowest at epoch {best_loss + 1} and rose afterwards, "
                         "so training could have stopped earlier.")
    if tiles:
        stat_tiles(tiles)
    if notes:
        callout(notes)


def eval_confusions(res, cm):
    y, pred, conf = res["y"], res["pred"], res["conf"]
    off = cm.copy()
    np.fill_diagonal(off, 0)
    errors = int(off.sum())

    if errors == 0:
        st.success("The model made no mistakes on this evaluation set.")
        return

    rows = [(t, p, int(off[t, p])) for t in range(10) for p in range(10) if off[t, p] > 0]
    rows.sort(key=lambda r: r[2], reverse=True)
    true_totals = cm.sum(axis=1)
    table = pd.DataFrame([{
        "True digit": t,
        "Read as": p,
        "Times": n,
        "Share of that digit": n / true_totals[t] if true_totals[t] else 0.0,
    } for t, p, n in rows[:10]])

    pairs = sorted(
        [(a, b, int(off[a, b] + off[b, a])) for a in range(10) for b in range(a + 1, 10)
         if off[a, b] + off[b, a] > 0],
        key=lambda r: r[2], reverse=True,
    )

    t, p, n = rows[0]
    wrong_guess = off.sum(axis=0)
    most_guessed = int(np.argmax(wrong_guess))
    insights = [
        f"Most frequent error: <strong>{t} read as {p}</strong>, {n} time{'s' if n != 1 else ''} "
        f"({n / true_totals[t] * 100:.1f}% of all {t}s).",
    ]
    if pairs:
        a, b, total = pairs[0]
        insights.append(f"Most confusable pair: <strong>{a} and {b}</strong> "
                        f"({total} mix-ups in both directions).")
    insights.append(f"The digit the model wrongly guesses most often is <strong>{most_guessed}</strong> "
                    f"({int(wrong_guess[most_guessed])} times).")
    right_mask = y == pred
    insights.append(
        f"Average confidence is {conf[right_mask].mean() * 100:.1f}% on correct readings "
        f"and {conf[~right_mask].mean() * 100:.1f}% on mistakes."
    )
    callout(insights)

    left, right = st.columns([1, 1.1], gap="large")
    with left:
        st.caption("Top 10 errors (true digit → predicted digit)")
        st.dataframe(table.style.format({"Share of that digit": "{:.1%}"}), hide_index=True)
    with right:
        top = rows[:10][::-1]
        fig, (ax,) = themed_axes(1, 6, 3.8)
        ax.barh([f"{t} → {p}" for t, p, _ in top], [n for _, _, n in top], color=INK)
        ax.set_xlabel("Number of mistakes")
        ax.grid(axis="x", alpha=0.15, color=MUTED)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    if pairs:
        st.caption("Most confusable pairs (both directions combined)")
        st.dataframe(pd.DataFrame(pairs[:8], columns=["Digit A", "Digit B", "Mix-ups"]), hide_index=True)


def render_evaluation():
    s = st.session_state
    page_head("Model evaluation", "Test the model on labelled digits and see exactly where it goes wrong.")

    with st.container(border=True):
        source = st.radio(
            "Evaluation data",
            ["Folder on disk", "Upload a ZIP", "MNIST test set"],
            horizontal=True,
            key="eval_source",
        )
        c1, c2 = st.columns([2, 1], gap="large")
        folder, zip_file, invert = EVAL_DATA_DIR, None, True
        with c1:
            if source == "Folder on disk":
                folder = st.text_input("Folder path", value=EVAL_DATA_DIR, key="eval_folder")
                st.caption("One sub-folder per digit: evaluation_data/0, evaluation_data/1 ... evaluation_data/9. "
                           "Use images the model did not train on.")
            elif source == "Upload a ZIP":
                zip_file = st.file_uploader("ZIP of labelled images", type=["zip"], key="eval_zip")
                st.caption("Each image must sit in a folder named after its digit (0-9).")
            else:
                invert = st.checkbox("Invert colours (MNIST is white on black)", value=True, key="eval_invert")
                st.caption("Downloads MNIST on first use. Square 28x28 images are stretched to the model's "
                           "90x140 input, so results only make sense if you trained on the same kind of data.")
        with c2:
            per_class = st.slider("Max images per digit", 20, 500, 100, 10, key="eval_per_class")

        run = st.button("Run evaluation", type="primary", use_container_width=True)

    if run:
        X = y = None
        try:
            with st.spinner("Loading images..."):
                if source == "Folder on disk":
                    if not os.path.isdir(folder):
                        raise FileNotFoundError(f"Folder not found: {folder}")
                    X, y = load_folder_dataset(folder, per_class)
                elif source == "Upload a ZIP":
                    if zip_file is None:
                        raise ValueError("Upload a ZIP file first.")
                    X, y = load_zip_dataset(zip_file.getvalue(), per_class)
                else:
                    X, y = load_mnist_dataset(per_class, invert)
        except Exception as e:
            st.error(str(e))

        if X is not None:
            if len(X) == 0:
                st.error("No labelled images were found. Check the folder names (0-9) and file types.")
            else:
                probs = predict_in_batches(X)
                s.eval_result = {
                    "X": X, "y": y,
                    "pred": probs.argmax(axis=1),
                    "conf": probs.max(axis=1),
                    "source": source,
                }
                s.eval_seed = 0

    res = s.eval_result
    if res is None:
        empty_state("Run an evaluation to see the metrics here.")
        # Training curves do not need an evaluation run
        st.markdown("---")
        render('<h2 class="section-title">Training curves</h2>')
        eval_curves()
        return

    cm = confusion_matrix_np(res["y"], res["pred"])
    metrics = per_class_metrics(cm)

    tabs = st.tabs(["Metrics", "Confusion matrix", "Examples", "Training curves", "Common confusions"])
    with tabs[0]:
        eval_overview(res, cm, metrics)
    with tabs[1]:
        eval_confusion(cm)
    with tabs[2]:
        eval_examples(res)
    with tabs[3]:
        eval_curves()
    with tabs[4]:
        eval_confusions(res, cm)


# ===========================================================================
# Pages
# ===========================================================================

# Home Page
if selected == "Home":
    render(f"""
    <div class="hero-grid">
        <div class="hero-copy">
            <h1>Write a digit. Watch it get read.</h1>
            <p>Upload a photo, snap one with your camera, or draw your answers to a short math quiz. A neural network reads every digit and tells you how sure it is.</p>
            <span class="nudge">Pick a mode from the menu above</span>
        </div>
        {hero_reader()}
    </div>
    """)

    render("""
    <h2 class="section-title">How it works</h2>
    <div class="steps">
        <div class="step">
            <span class="n">1</span>
            <h3>Show a digit</h3>
            <p>Draw it, upload a picture, or take a photo.</p>
        </div>
        <div class="step">
            <span class="n">2</span>
            <h3>The model reads it</h3>
            <p>The image is resized and scored against all ten digits.</p>
        </div>
        <div class="step">
            <span class="n">3</span>
            <h3>See the result</h3>
            <p>You get the best match and a confidence score right away.</p>
        </div>
    </div>
    """)

    render(f"""
    <h2 class="section-title">Ways to use it</h2>
    <div class="bento">
        <div class="tile hot s4">
            <div class="icon">{ICON_PENCIL}</div>
            <h3>Adaptive math quiz</h3>
            <p>Pick easy, medium or hard, or let the quiz adapt to you. Draw each answer by hand, use hints, and revisit the questions you got wrong.</p>
        </div>
        <div class="tile s2">
            <div class="icon">{ICON_UPLOAD}</div>
            <h3>Upload an image</h3>
            <p>Read a digit from a JPG or PNG.</p>
        </div>
        <div class="tile s2">
            <div class="icon">{ICON_CAMERA}</div>
            <h3>Use the camera</h3>
            <p>Capture a digit with your device.</p>
        </div>
        <div class="tile s3">
            <div class="icon">{ICON_CHART}</div>
            <h3>Learning analytics</h3>
            <p>Track accuracy, response time and progress by difficulty and question type.</p>
        </div>
        <div class="tile s3">
            <div class="icon">{ICON_GRID}</div>
            <h3>Model evaluation</h3>
            <p>Confusion matrix, precision, recall, F1 and the digits the model mixes up.</p>
        </div>
        <div class="tile s6">
            <div class="icon">{ICON_CHIP}</div>
            <h3>Built with</h3>
            <div class="chips">
                <span class="chip">TensorFlow</span>
                <span class="chip">Streamlit</span>
                <span class="chip">OpenCV</span>
                <span class="chip">Python</span>
            </div>
        </div>
    </div>
    """)

# Quiz App
elif selected == "Math Quiz":
    stage = st.session_state.quiz_stage
    if stage == "running" and st.session_state.current_question_index >= st.session_state.round_len:
        stage = "done"

    if stage == "setup":
        render_setup()
    elif stage == "running":
        page_head("Math quiz", "Draw each answer and the model will check it.")
        ask_question()
    else:
        render_round_report()

# Learning analytics
elif selected == "Learning Analytics":
    render_analytics()

# Model evaluation
elif selected == "Model Evaluation":
    render_evaluation()

# File Upload
elif selected == "Browse File":
    page_head("Upload an image", "Choose a photo of a handwritten digit and the model will read it.")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

    if uploaded_file is not None:
        with st.spinner("Processing image..."):
            image = Image.open(uploaded_file)
            img = cv2.resize(np.array(image), (800, 300))

            with col1:
                st.image(img, caption='Uploaded image', width="stretch")

            with col2:
                digit, conf = predict_with_confidence(image)
                proba = conf * 100

                level = 'high' if proba >= 85 else 'medium' if proba >= 70 else 'low'
                reading_card(
                    digit, proba, level,
                    confidence_note(proba, "Try a clearer image.")
                )
    else:
        with col2:
            empty_state("Your result will appear here.")

# Camera Input
elif selected == "Camera Input":
    page_head("Camera", "Take a photo of a handwritten digit and the model will read it.")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        camera_image = st.camera_input("Take a picture of a handwritten digit", label_visibility="collapsed")

    if camera_image is not None:
        with st.spinner("Processing image..."):
            image = Image.open(camera_image)

            with col2:
                digit, conf = predict_with_confidence(image)
                proba = conf * 100

                level = 'high' if proba >= 85 else 'medium' if proba >= 70 else 'low'
                reading_card(
                    digit, proba, level,
                    confidence_note(proba, "Try better lighting or clearer handwriting.")
                )
    else:
        with col2:
            empty_state("Your result will appear here.")

# Demo
elif selected == "Demo":
    page_head("Demo", "Watch how the app works.")

    st.video('OCR_DEMO/OCR_APP_Demo.mkv', format="video/mkv", start_time=0)

# Documentation
elif selected == "Documentation":
    page_head("Documentation", "How the digit recognition and quiz app works.")

    # Each section: title on the left, content on the right
    c1, c2 = st.columns([1, 2.4], gap="large")
    with c1:
        render('<h3 class="doc-title">Project overview</h3><p>What the app does</p>')
    with c2:
        st.markdown("""
This application combines machine learning with an interactive educational interface to recognize handwritten digits
and provide an adaptive math quiz for young learners. The app features multiple input methods including image upload,
camera capture, and a drawing canvas for real-time digit recognition.

**Key features**

- **Digit recognition**: AI model trained on handwritten digit images
- **Multiple input methods**: File upload, camera capture, and interactive drawing
- **Adaptive quiz**: Easy, medium and hard questions, hints, and targeted practice
- **Learning analytics**: Accuracy, response time and progress by type and difficulty
- **Model evaluation**: Confusion matrix, precision, recall, F1-score and error analysis
- **Real-time feedback**: Immediate results with confidence scoring
        """)

    st.markdown("---")

    c1, c2 = st.columns([1, 2.4], gap="large")
    with c1:
        render('<h3 class="doc-title">Technical details</h3><p>Model and stack</p>')
    with c2:
        st.markdown("""
**Model architecture**

The application uses a Convolutional Neural Network (CNN) implemented with TensorFlow/Keras. It takes a
90x140 RGB image and outputs a probability for each digit from 0 to 9.

**Technologies used**
        """)
        render("""
        <div class="chips">
            <span class="chip">Python</span>
            <span class="chip">TensorFlow</span>
            <span class="chip">Streamlit</span>
            <span class="chip">OpenCV</span>
            <span class="chip">Matplotlib</span>
            <span class="chip">Seaborn</span>
            <span class="chip">Pandas</span>
            <span class="chip">NumPy</span>
        </div>
        """)
        st.markdown("""
**Performance metrics**

Use the **Model Evaluation** page to measure accuracy, precision, recall and F1-score on your own
labelled test images, and to see training and validation curves.
        """)

    st.markdown("---")

    c1, c2 = st.columns([1, 2.4], gap="large")
    with c1:
        render('<h3 class="doc-title">How it works</h3><p>Pipeline and quiz logic</p>')
    with c2:
        st.markdown("""
**Image processing pipeline**

1. Input image is converted to a 3-channel RGB array
2. Image is resized to 90x140 pixels
3. Pixel values are normalized to the 0-1 range
4. Image is passed through the CNN model
5. Model outputs probabilities for each digit (0-9)
6. Digit with the highest probability is selected

**Quiz logic**

1. You choose a mode: fixed difficulty, adaptive, or targeted practice
2. The system generates a question whose answer is a single digit
3. You draw the answer on the canvas, optionally asking for hints
4. The AI recognizes the drawn digit and compares it with the correct answer
5. Accuracy, response time, hints and mistakes are recorded

**Adaptive difficulty**

- Two clean correct answers in a row move you up a level
- Two wrong answers in a row move you down a level
- Answers that needed a hint do not count towards moving up
- Targeted practice brings back missed questions and weak question types, with hints
        """)

    st.markdown("---")

    c1, c2 = st.columns([1, 2.4], gap="large")
    with c1:
        render('<h3 class="doc-title">Evaluating the model</h3><p>Data and files needed</p>')
    with c2:
        st.markdown("""
**Test images**

Place labelled images the model has not seen in `evaluation_data/0` ... `evaluation_data/9`, upload a ZIP with the
same layout, or use the MNIST test set.

**Training curves**

Save the Keras history after training as `OCR_MODEL/training_history.json` (or a CSV from `CSVLogger`), or upload it on
the Model Evaluation page.
        """)

    st.markdown("---")

    c1, c2 = st.columns([1, 2.4], gap="large")
    with c1:
        render('<h3 class="doc-title">Future enhancements</h3><p>What could come next</p>')
    with c2:
        st.markdown("""
- Multi-digit recognition
- Support for mathematical symbols
- User accounts and saved progress between sessions
- Mobile app version
- Additional quiz types
- Enhanced model explainability features
        """)