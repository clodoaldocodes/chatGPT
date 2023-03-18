import os
import shutil
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Define path to downloaded data folder
download_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloaded_data")

# Create the folder if it doesn't exist
if not os.path.exists(download_folder):
    os.makedirs(download_folder)

# Define the URLs and file names of the data to download
tokenizer_url = "https://huggingface.co/EleutherAI/gpt-neo-2.7B/resolve/main/tokenizer.json"
model_url = "https://huggingface.co/EleutherAI/gpt-neo-2.7B/resolve/main/pytorch_model.bin"
config_url = "https://huggingface.co/EleutherAI/gpt-neo-2.7B/resolve/main/config.json"

tokenizer_filename = "tokenizer.json"
model_filename = "pytorch_model.bin"
config_filename = "config.json"

# Define the paths to save the downloaded files
tokenizer_path = os.path.join(download_folder, tokenizer_filename)
model_path = os.path.join(download_folder, model_filename)
config_path = os.path.join(download_folder, config_filename)

# Download the tokenizer and save it to the file path
with torch.hub.download_url_to_file(tokenizer_url, tokenizer_path, progress=True):
    print(f"Downloaded {tokenizer_filename} to {tokenizer_path}")

# Download the model and save it to the file path
with torch.hub.download_url_to_file(model_url, model_path, progress=True):
    print(f"Downloaded {model_filename} to {model_path}")

# Download the config and save it to the file path
with torch.hub.download_url_to_file(config_url, config_path, progress=True):
    print(f"Downloaded {config_filename} to {config_path}")

# Set up the tokenizer and model for GPT-3.5-turbo
tokenizer = GPT2Tokenizer.from_pretrained(download_folder, cache_dir=download_folder)
model = GPT2LMHeadModel.from_pretrained(download_folder, cache_dir=download_folder)

# Set up the prompt for ChatGPT
prompt = "Hello, I'm ChatGPT. How can I assist you today?"

# Create a function to generate a response from ChatGPT
def generate_response(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=2048, do_sample=True)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.strip()

# Loop to continue generating responses based on user input
while True:
    user_input = input("> ")
    prompt += "\nUser: " + user_input
    response = generate_response(prompt)
    prompt += "\nChatGPT: " + response
    print(response)
