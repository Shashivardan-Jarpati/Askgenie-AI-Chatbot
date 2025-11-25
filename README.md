# Askgenie-AI-Chatbot
This AI Chatbot is a locally hosted, instruction-tuned language model built using Streamlit for the user interface and the Hugging Face Transformers library with the Phi-3-mini-4k-instruct model for text generation. It operates fully on the user's machine without needing API keys or sending data to the cloud, ensuring privacy and speed.

# Key Features
Streamlit UI with customized styles and chat message formatting.
Uses Hugging Face Transformers pipeline for text generation.
Detects user intents like help, restart, about, and human handoff to handle commands gracefully.
Supports options to control response style, temperature, max tokens, and top-p nucleus sampling dynamically via sidebar.
Conversation history maintained in session state; clean formatting of bot responses.
Uses simple text processing (regex) to clean and format bot responses for better readability.
Includes system prompt to guide the bot's behavior and response style.
Loads AI model on first interaction with graceful error handling for model loading failures.

# How it works
User inputs a message via the Streamlit chat input box.
The bot detects if the message is an intent command.(e.g., help, restart)
If intent, replies with predefined text or actions.
Else, builds a prompt context (system + recent conversation messages) and generates a reply using the model pipeline.
Cleans up and formats the bot response before displaying it.
Supports GPU acceleration if available, else uses CPU.

# Technologies used
Streamlit: App UI and input/output handling.
Hugging Face Transformers: Text generation with pretrained language models.
PyTorch: Underlying deep learning engine for model inference.
Regex: Text cleaning and formatting.


# Usage
Run the app with streamlit run chatbot.py
On first run, the AI model downloads locally which may take a little time.
Use chat interface to converse with the AI assistant.
Use sidebar controls to adjust response style and generation parameters.

# Commands:
help - shows usage instructions
restart - clears conversation history
about - details about the bot and technologies
menu - shows common sample questions and commands
talk to a human or similar handoff phrases trigger human handoff simulation

# Overview
This is a local AI chatbot application built with Streamlit as the interface and powered by a Hugging Face instruction-tuned language model (microsoft/Phi-3-mini-4k-instruct) running on PyTorch. The bot runs entirely on the user's machine without sending data to the cloud, emphasizing privacy.

# Why It Works Automatically:
The AI model is a trained neural network that learned from millions of conversations. When you give it:
A system prompt (personality)
Conversation history (context)
Parameters (temperature, length)

# It automatically predicts the most appropriate response based on patterns it learned during training.
# Think of it like autocomplete on steroids:
Your phone suggests next words based on what you typed
This AI suggests entire sentences based on the conversation context
It's been trained on so much text that it "understands" how to respond

# Key Technologies Making It Work:
Transformers Library - Loads and runs the AI model
PyTorch - The math engine that powers the neural network
Phi-3-mini-4k-instruct - The actual AI brain (trained by Microsoft)
Streamlit - The web interface you see







