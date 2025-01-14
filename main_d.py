# Needed for OpenAI generator - pip install openai
import openai
from openai import OpenAI
import time
import textwrap
# Needed to calculate cosine distance - pip install scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Needed to calculate enhanced distance - pip install spacy nltk numpy && python -m spacy download en_core_web_sm
import spacy
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
from collections import Counter
import numpy as np

# Instance the client with the OpenAPI key
f = open("openai.key", "r")
client = OpenAI(
    api_key=f.readline().strip()
)
f.close()

# Set up basic configurations and timing references
gptmodel="gpt-4o"
start_time = time.time()

# Load spaCy with en_core_web_sm tokenizer and suite
nlp = spacy.load("en_core_web_sm")

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

# Set up our knowledge storage
db_records = [
    "Retrieval Augmented Generation (RAG) represents a sophisticated hybrid \
    approach in the field of artificial intelligence, particularly within the realm \
    of natural language processing (NLP).",
]

def find_best_match(query, vectorizer, tfidf_matrix):
    query_tfidf = vectorizer.transform([query])
    similarities = cosine_similarity(query_tfidf, tfidf_matrix)
    # This will return the index of the highest similarity score
    best_index = similarities.argmax()
    best_score = similarities[0, best_index]
    return best_score, best_index

query = "define a rag store"

# Set up the TF-IDF vector store
vectorizer = TfidfVectorizer(
    stop_words='english', # This will have the calculation focus
    # on meaningful words, skipping frequently found ones
    use_idf=True, # Using Inverse Domain Frequency weighting, we
    # accentuate differences
    norm='l2', # Uses L2 for normalization
    ngram_range=(1, 2), # Only unigrams and bigrams
    sublinear_tf=True, # Will use sublinear term frequency scaling
    analyzer='word' # This will analyze the distance by word level
    # 'char' or 'char_wb' exist too, and are character-based, but are
    # seldom useful
)
tfidf_matrix = vectorizer.fit_transform(db_records)
best_similarity_score, best_index = find_best_match(query, vectorizer, tfidf_matrix)
best_matching_record = db_records[best_index]

# Let's augment the input with the best matching k
augmented_input=query+ ": "+ best_matching_record
print_formatted_response(augmented_input)

# And here the magic happens!
llm_response = call_llm_with_full_text(augmented_input)
print_formatted_response(llm_response)
