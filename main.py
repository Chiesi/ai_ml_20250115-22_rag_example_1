import os
import openai
from openai import OpenAI
import time
import textwrap

# Instance the client with the OpenAPI key
f = open("openai.key", "r")
os.environ['OPENAI_API_KEY']=f.readline().strip()
f.close()
client = OpenAI()

# Setup our knowledge storage
db_records = [
    "Retrieval Augmented Generation (RAG) represents a sophisticated hybrid \
    approach in the field of artificial intelligence, particularly within the realm \
    of natural language processing (NLP).",
]

gptmodel="gpt-4o"
start_time = time.time()

# Invoke the LLM with the user prompt
def call_llm_with_full_text(itext):
    # Join all lines to form a single string
    text_input = '\n'.join(itext)
    prompt = f"Please elaborate on the following content:\n{text_input}"
    try:
        response = client.chat.completions.create(
            model=gptmodel,
            messages=[
                {"role": "system", "content": "You are an expert Natural Language Processing exercise expert."},
                {"role": "assistant", "content": "1.You can explain read the input and answer in detail"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # Add the temperature parameter here and other parameters you need
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return str(e)

# This helps users avoid mangled responses
def print_formatted_response(response):
    # Define the width for wrapping the text
    wrapper = textwrap.TextWrapper(width=80)  # Set to 80 columns wide, but adjust as needed
    wrapped_text = wrapper.fill(text=response)
    # Print the formatted response with a header and footer
    print("Response:")
    print("---------------")
    print(wrapped_text)
    print("---------------\n")

#TODO: Read from user input
query = "define a rag store"

# And here the magic happens!
llm_response = call_llm_with_full_text(query)
print_formatted_response(llm_response)