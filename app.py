
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os

# Load the fine-tuned model and tokenizer from the local directory
@st.cache_resource
def load_model_and_tokenizer():
    model_path = "./riddle_factory_model"  # This will be in the GitHub repo
    if not os.path.exists(model_path):
        st.error(f"Model directory '{model_path}' not found in the app directory.")
        return None, None, None
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token
    return model, tokenizer, device

model, tokenizer, device = load_model_and_tokenizer()

# Function to generate a riddle
def generate_riddle(model, tokenizer, device, max_length=100):
    if model is None or tokenizer is None:
        return "Error: Model or tokenizer not loaded."
    prompt = "Riddle: "
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        top_k=40,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit app interface
st.title("Math Riddle Factory")
st.write("Click the button below to generate a new math riddle!")

if st.button("Generate Riddle"):
    with st.spinner("Generating your riddle..."):
        try:
            riddle = generate_riddle(model, tokenizer, device)
            st.success("Here's your riddle:")
            st.write(riddle)
        except Exception as e:
            st.error(f"Error generating riddle: {str(e)}")
