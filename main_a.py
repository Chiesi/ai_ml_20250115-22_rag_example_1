import os
import openai
from openai import OpenAI
import time
import textwrap

# Instance the client with the OpenAPI key
f = open("openai.key", "r")
client = OpenAI(
    api_key=f.readline().strip()
)
f.close()

# Set up basic configurations and timing references
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
            temperature=0.1  # Let's discuss what temperature does!
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

query = "define a rag store"

# And here the magic happens!
llm_response = call_llm_with_full_text(query)
print_formatted_response(llm_response)