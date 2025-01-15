# Needed for OpenAI generator - pip install openai
import openai
from openai import OpenAI
import time
import textwrap
# Needed to calculate cosine distance - pip install scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Needed to calculate enhanced distance - pip install spacy nltk numpy && python -m spacy download en_core_web_sm
# import spacy
# import nltk
# nltk.download('wordnet')
# from nltk.corpus import wordnet
from collections import Counter
import numpy as np

# Instance the client with the OpenAPI key
f = open("openai.key", "r")
client = OpenAI(
    api_key=f.readline().strip()
)
f.close()

# Set up basic configurations and timing references
# gptmodel="gpt-4o"
gptmodel = "gpt-3.5-turbo"
start_time = time.time()

# Load spaCy with en_core_web_sm tokenizer and suite
# nlp = spacy.load("en_core_web_sm")

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

# Calculate the distance as teh angle between vectors.
# We all remember the definition of vector?
def calculate_cosine_similarity(text1, text2):
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
    tfidf = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return similarity[0][0]

# Set up our knowledge storage
db_records = [
    "Product A, 2, 10",
    "Product B, 3, 15",
    "Product C, 4, 40",
    "Product D, 5, 25",
    "Product E, 6, 30", 
    "Product F, 7, 35", 
    "Product G, 8, 40", 
    "Product H, 9, 45", 
]

# Tries to find the best match against a database of knowledge
def find_best_match_keyword_search(query, db_records):
    best_score = 0
    best_record = None
    # This will split the query into several words, to find a keyword amongst them
    query_keywords = set(query.lower().split())
    # Iterate through each record in db_records
    for record in db_records:
        # Split the record into keywords
        record_keywords = set(record.lower().split())
        # Calculate the number of common keywords
        common_keywords = query_keywords.intersection(record_keywords)
        current_score = len(common_keywords)
        # Update the best score and record if the current score is higher
        if current_score > best_score:
            best_score = current_score
            best_record = record
    return best_score, best_record

query = "Find the most valuable product from the sales count and sales revenue in this data." \
    + "The data is a list of CSV rows, with the first column bearing the product name, " \
    + "the second column bearing the number of sold units, and the third column " \
    + "bearing the total revenue from all sold units"
# query = "Find the most valuable product from the data"

# What if we don't invoke the LLM at all?
# best_keyword_score, best_matching_record = find_best_match_keyword_search(query, db_records)
# print(f"Best Keyword Score: {best_keyword_score}")

# Let's augment the input with the best matching k
augmented_input=query+ ": "+ " \n".join(db_records)
#augmented_input=query
print_formatted_response(augmented_input)

# And here the magic happens!
llm_response = call_llm_with_full_text(augmented_input)
print_formatted_response(llm_response)
