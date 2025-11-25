import streamlit as st
from transformers import pipeline
import torch
import re

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

# ---------- FIXED: custom background + chat styling ----------
st.markdown(
    """
    <style>
    /* Whole app background */
    .stApp {
        background: radial-gradient(circle at top left, #2c3e50 0, #0b1018 40%, #050710 100%);
        color: #e5e7eb;
    }

    /* Main title and caption - FORCE WHITE TEXT */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }

    /* Force all text to be white/light */
    .stMarkdown, .stText, .stCaption {
        color: #ffffff !important;
    }
    
    /* Sidebar text - FORCE WHITE */
    .stSidebar {
        background-color: #1e293b !important;
    }
    
    .stSidebar * {
        color: #ffffff !important;
    }
    
    /* Sidebar headers */
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4 {
        color: #ffffff !important;
    }
    
    /* Sidebar labels and text */
    .stSidebar label, .stSidebar .stMarkdown, .stSidebar p {
        color: #ffffff !important;
    }
    
    /* Slider labels */
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
    }
    
    /* Sidebar header text - Session Analytics, etc */
    .stSidebar .element-container {
        color: #ffffff !important;
    }
    
    /* Force all sidebar children to be white */
    [data-testid="stSidebar"] div, 
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] p {
        color: #ffffff !important;
    }
    
    /* Top app header/toolbar */
    header[data-testid="stHeader"] {
        background-color: #1e293b !important;
    }
    
    /* Main toolbar at top */
    .main .block-container header {
        color: #ffffff !important;
    }
    
    /* Main content area text */
    .main .block-container, .main .block-container * {
        color: #ffffff !important;
    }
    
    /* Caption specifically */
    .main .element-container .stCaption {
        color: #9ca3af !important;
    }

    /* Chat message container tweaks */
    [data-testid="stChatMessage"] {
        border-radius: 18px;
        padding: 0.35rem 0.75rem;
        margin-bottom: 0.35rem;
        background: rgba(15, 23, 42, 0.85);
    }

    /* Try to differentiate user vs bot bubble colors */
    [data-testid="stChatMessage"] div[data-testid="stChatMessageAvatarUser"] + div {
        background: linear-gradient(135deg, #0f172a, #020617);
        border-radius: 18px;
        padding: 0.5rem 0.75rem;
    }

    [data-testid="stChatMessage"] div[data-testid="stChatMessageAvatarAssistant"] + div {
        background: linear-gradient(135deg, #020617, #0b1120);
        border-radius: 18px;
        padding: 0.5rem 0.75rem;
    }
    
    /* Chat message avatar styling - fix white background */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        background: rgba(30, 41, 59, 0.6) !important;
        border-radius: 8px !important;
    }
    
    /* Spinner/loading text color */
    .stSpinner > div {
        border-color: rgba(100, 200, 255, 0.3) rgba(100, 200, 255, 0.3) rgba(100, 200, 255, 0.9) rgba(100, 200, 255, 0.3) !important;
    }
    
    /* "Thinking..." text color - Make it more visible */
    .stSpinner > div > div {
        color: #60a5fa !important;
    }
    
    /* Spinner container text */
    [data-testid="stStatusWidget"] {
        color: #60a5fa !important;
    }
    
    /* All spinner related text */
    .stSpinner, .stSpinner * {
        color: #60a5fa !important;
    }

    /* Chat input background - White rounded like ChatGPT */
    textarea {
        background: #ffffff !important;
        color: #000000 !important;
        border-radius: 24px !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        padding: 12px 20px !important;
    }
    
    /* Chat input placeholder text */
    textarea::placeholder {
        color: #6b7280 !important;
    }
    
    /* Chat input container - remove ALL white borders - FIXED */
    [data-testid="stChatInput"], 
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div,
    .stChatInputContainer,
    .stChatInput > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Bottom chat input wrapper - remove white background */
    .main > div > div > div > div[data-testid="stBottom"],
    [data-testid="stBottom"] > div {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    /* Target the white border container */
    div[data-testid="InputInstructions"],
    .stChatFloatingInputContainer {
        background: transparent !important;
        border: none !important;
    }
    
    /* Remove any extra white space/padding around input */
    [data-testid="stChatInput"] > div:first-child {
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Fix the white overflow on edges */
    [data-testid="stChatInput"] > div > div:first-child {
        background: transparent !important;
        overflow: hidden !important;
        border-radius: 26px !important;
    }
    
    /* Sidebar info/warning boxes - HIDE white background completely */
    .stSidebar .stAlert {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        color: #ffffff !important;
    }
    
    /* Info/Warning/Success boxes in sidebar */
    .stSidebar [data-baseweb="notification"] {
        background: rgba(30, 41, 59, 0.6) !important;
        color: #ffffff !important;
    }
    
    /* All alert children text white */
    .stSidebar .stAlert * {
        color: #ffffff !important;
    }
    
    /* Sidebar button styling - make it dark */
    .stSidebar button {
        background-color: rgba(30, 41, 59, 0.6) !important;
        color: #ffffff !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
    }
    
    .stSidebar button:hover {
        background-color: rgba(30, 41, 59, 0.9) !important;
        border-color: rgba(148, 163, 184, 0.5) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# -----------------------------------------------------------

st.title("🤖 AskGenie AI")
st.caption("Powered by local AI - No API key needed!")

# -------------------------------
# Chatbot "personality"
# -------------------------------
SYSTEM_PROMPT = (
    "You are a helpful, friendly AI assistant. "
    "You run locally on the user's computer. "
    "Always answer the user's latest question. "
    "Answer clearly and concisely, using simple language when possible. "
    "If the user asks for code, provide short, working examples. "
    "If you are not sure, say so honestly and suggest what the user could try. "
    "Do not repeat yourself. Do not say 'I am a bot' unless the user asks. "
    "If the user uses another language, reply in the same language."
)

HELP_TEXT = """
### 🤖 How I Can Help

You can ask me things like:
- "Explain what Streamlit is in simple words."
- "Give me a short Python example that adds two numbers."
- "What is NLP and how is it used?"
- "Summarize: what is machine learning?"

### 🧭 Commands you can use

- `/help` – Show this help message.
- `/restart` – Clear the chat and start fresh.
- `/about` – Show details about this local AI bot.
- `/menu` – Show common options and example questions.

If you need a human, you can type:
- "talk to a human"
- "connect me to an agent"
"""

MENU_TEXT = """
### 📋 Main Menu (Demo)

You can try questions like:
- "Explain Streamlit."
- "What is NLP?"
- "Show a simple Python function."
- "What is deep learning in one paragraph?"
"""

ABOUT_TEXT = """
### ℹ️ About this bot

- Runs **entirely on your computer** using a local instruction-tuned AI model (Phi-3-mini-4k-instruct).
- No API keys, no data sent to the cloud in this demo.
- Best at: explanations, simple reasoning, short code snippets.
- Still lighter than ChatGPT, so sometimes answers may be shorter or less detailed.
"""

# -------------------------------
# FORMATTING FIX FUNCTION
# -------------------------------
def clean_and_format_response(text: str) -> str:
    """
    Clean and properly format the bot's response.
    Fixes: spacing, punctuation, length, formatting issues.
    """
    if not text or len(text.strip()) < 3:
        return "I'm not sure how to respond. Could you rephrase that?"
    
    # Remove extra whitespace and normalize
    text = ' '.join(text.split())
    
    # Fix missing spaces after punctuation
    text = re.sub(r'([.!?,;:])([A-Za-z])', r'\1 \2', text)
    
    # Fix missing spaces between concatenated words
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # Remove any leaked tags like [SYSTEM], [USER], [ASSISTANT]
    text = re.sub(r'\[(SYSTEM|USER|ASSISTANT)\]', '', text, flags=re.IGNORECASE)
    
    # Remove duplicate spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split into sentences properly
    sentences = re.split(r'([.!?]+)', text)
    
    # Reconstruct with proper spacing
    cleaned = []
    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i].strip()
        punct = sentences[i+1] if i+1 < len(sentences) else '.'
        if sentence:
            cleaned.append(sentence + punct)
    
    # Join sentences
    result = ' '.join(cleaned)
    
    # Limit to 2 sentences max
    final_sentences = re.split(r'[.!?]+\s*', result)
    final_sentences = [s.strip() for s in final_sentences if s.strip()]
    
    if len(final_sentences) > 2:
        final_sentences = final_sentences[:2]
    
    result = '. '.join(final_sentences)
    
    # Ensure it ends with punctuation
    if result and result[-1] not in '.!?':
        result += '.'
    
    # Limit overall length to 200 characters
    if len(result) > 200:
        result = result[:200]
        last_period = result.rfind('.')
        if last_period > 100:
            result = result[:last_period+1]
        else:
            result = result[:200].rstrip() + '...'
    
    return result

# -------------------------------
# Intent detection (simple NLU)
# -------------------------------
def detect_intent(user_text: str) -> str:
    text = user_text.lower().strip()

    # Slash-style commands
    if text.startswith("/help"):
        return "help"
    if text.startswith("/restart"):
        return "restart"
    if text.startswith("/about"):
        return "about"
    if text.startswith("/menu"):
        return "menu"

    # Human handoff phrases (demo)
    if "human" in text or "agent" in text or "support" in text:
        return "handoff"

    # Escape / reset phrases
    if "start over" in text or "restart chat" in text:
        return "restart"

    # Default = normal chat with the model
    return "chat"

# -------------------------------
# Load model (only once)
# -------------------------------
@st.cache_resource
def load_model():
    try:
        # Use GPU if available, else CPU
        device = 0 if torch.cuda.is_available() else -1

        # 🔁 INSTRUCTION-TUNED MODEL HERE
        model_name = "microsoft/Phi-3-mini-4k-instruct"

        # Simple pipeline - if you get CUDA OOM, see note below
        chatbot = pipeline(
            "text-generation",
            model=model_name,
            device=device,
        )
        return chatbot
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# -------------------------------
# Helpers
# -------------------------------
def build_chat_prompt(messages, max_messages=6):
    """
    Build a prompt in an instruction-style format:

    [SYSTEM] ...
    [USER] ...
    [ASSISTANT] ...
    """

    lines = []

    # System / instruction part
    lines.append(f"[SYSTEM] {SYSTEM_PROMPT}")

    # Only keep the most recent messages
    recent = messages[-max_messages:]

    for m in recent:
        if m["role"] == "user":
            lines.append(f"[USER] {m['content']}")
        elif m["role"] == "assistant":
            # Include last assistant replies so model has context
            lines.append(f"[ASSISTANT] {m['content']}")

    # Ask the model to reply now
    lines.append("[ASSISTANT]")
    return "\n".join(lines)

def handle_intent(intent: str):
    """
    Handle non-model intents directly: help, menu, about, handoff, etc.
    Returns (handled: bool, reply: str or None)
    """
    if intent == "help":
        return True, HELP_TEXT

    if intent == "menu":
        return True, MENU_TEXT

    if intent == "about":
        return True, ABOUT_TEXT

    if intent == "handoff":
        reply = (
            "It sounds like you'd like to talk to a human.\n\n"
            "👤 In a real deployment, I would now transfer this conversation, with context, "
            "to a human support agent or open a support ticket.\n\n"
            "In this local demo, I can't actually connect you, but you can copy this chat "
            "and share it with your support team."
        )
        return True, reply

    if intent == "restart":
        reply = "🔁 Chat restarted. You can start fresh with a new question."
        return True, reply

    return False, None

# -------------------------------
# Session state
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model" not in st.session_state:
    st.session_state.model = None

if "total_user_messages" not in st.session_state:
    st.session_state.total_user_messages = 0
if "total_bot_messages" not in st.session_state:
    st.session_state.total_bot_messages = 0
if "last_intent" not in st.session_state:
    st.session_state.last_intent = "none"

# -------------------------------
# Sidebar: controls + info
# -------------------------------
with st.sidebar:
    st.header("WELCOME TO THE AI CHATBOT CONTROLS")

    response_style = st.radio(
        "Response style",
        ["Fast", "Balanced", "Detailed"],
        index=1,  # default Balanced
    )

    temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.05)
    top_p = st.slider("Top-p (nucleus sampling)", 0.1, 1.0, 0.9, 0.01)

    max_tokens = st.slider("Max reply tokens", 20, 200, 50, 10)

    if response_style == "Fast":
        max_new_tokens = 80
    elif response_style == "Detailed":
        max_new_tokens = 160
    else:
        max_new_tokens = 100

    st.caption(f"Max new tokens this reply: {max_new_tokens}")

    st.markdown("---")
    if st.button("🔁 Restart conversation"):
        st.session_state.messages = []
        st.session_state.total_user_messages = 0
        st.session_state.total_bot_messages = 0
        st.session_state.last_intent = "restart"
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Session Analytics")
    st.text(f"User messages: {st.session_state.total_user_messages}")
    st.text(f"Bot messages:  {st.session_state.total_bot_messages}")
    st.text(f"Last intent:   {st.session_state.last_intent}")

    st.markdown("---")
    st.markdown("### ABOUT")
    st.info(
        "You are chatting with a **local instruction-tuned AI bot** "
        "based on Phi-3-mini-4k-instruct. It runs fully on your machine in this demo."
    )

    st.markdown("### FIRST TIME SETUP")
    st.warning(
        "On the first message, it must download & load the AI model. "
        "After that, responses are much faster."
    )

    st.markdown("### RESOURCES")
    st.markdown("This AI chatbot is powered by the following open-source technologies:")

    st.markdown("### 🧩 CORE FRAMEWORKS :")
    st.markdown("- 🌐 **Streamlit** – App UI framework  https://streamlit.io/")
    st.markdown("- 🧠 **Hugging Face Transformers** – NLP model toolkit   https://huggingface.co/docs/transformers")
    st.markdown("- 🔥 **PyTorch** – Deep learning engine  https://pytorch.org/")

    st.markdown("### 🤖 MODEL USED :")
    st.markdown("- 🗣 **microsoft/Phi-3-mini-4k-instruct** – Instruction-tuned language model")

    st.markdown("### ⚙️ Text Generation")
    st.markdown("- ✨ **Generation Parameters Guide**  https://huggingface.co/docs/transformers/main/en/main_classes/text_generation")

    st.markdown("### 🔤 Tokenization")
    st.markdown("- 🔍 **Tokenizer Documentation**  https://huggingface.co/docs/transformers/main/en/tokenizer_summary")
    st.markdown("---")
    st.markdown("Developed by Shashivardan J. 2025.")

# -------------------------------
# Show chat history
# -------------------------------
for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# -------------------------------
# Chat input + generation
# -------------------------------
if prompt := st.chat_input("What would you like to know? (type /help for options)"):
    st.session_state.total_user_messages += 1

    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    intent = detect_intent(prompt)
    st.session_state.last_intent = intent

    handled, intent_reply = handle_intent(intent)

    if intent == "restart":
        st.session_state.messages = []
        st.session_state.total_user_messages = 0
        st.session_state.total_bot_messages = 0
        st.rerun()

    # Bot reply with avatar
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                if handled and intent_reply:
                    st.markdown(intent_reply)
                    st.session_state.messages.append({"role": "assistant", "content": intent_reply})
                    st.session_state.total_bot_messages += 1
                else:
                    if st.session_state.model is None:
                        st.info("Loading AI model for the first time... This might take a bit on first run.")
                        st.session_state.model = load_model()

                    model = st.session_state.model

                    if model:
                        chat_prompt = build_chat_prompt(st.session_state.messages)

                        raw = model(
                            chat_prompt,
                            max_new_tokens=max_new_tokens,
                            num_return_sequences=1,
                            do_sample=True,
                            top_p=top_p,
                            temperature=temperature,
                            repetition_penalty=1.1,
                            no_repeat_ngram_size=3,
                            pad_token_id=model.tokenizer.eos_token_id,
                            return_full_text=False,
                        )

                        reply = raw[0]["generated_text"].strip()
                        
                        reply = clean_and_format_response(reply)

                        if not reply or len(reply.split()) < 2:
                            reply = (
                                "I'm not sure I understood that. "
                                "You can type `/help` to see what I can do, "
                                "or try asking in a different way."
                            )

                        st.markdown(reply)
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                        st.session_state.total_bot_messages += 1
                    else:
                        st.error("Model failed to load. Please refresh the page.")
            except Exception as e:
                st.error(f"Error: {str(e)}")