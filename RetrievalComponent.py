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

# Set up our knowledge storage
db_records = [
    "Retrieval Augmented Generation (RAG) represents a sophisticated hybrid \
    approach in the field of artificial intelligence, particularly within the realm \
    of natural language processing (NLP).",
]

class RetrievalComponent:
    def __init__(self, method='vector'):
        """A class acting as the Retrieval component of our RAG system

        Args:
            method (str, optional): The retrieval method to implement.
            Defaults to 'vector'.
        """
        self.method = method
        if self.method == 'vector' or self.method == 'indexed':
            self.vectorizer = TfidfVectorizer()
            self.tfidf_matrix = None

    def fit(self, records):
        """Builds a TF-IDF matrix of the knowledge base and stores it
        in memory

        Args:
            records (str): The knowledge-base undergoing vectorization
        """
        if self.method == 'vector' or self.method == 'indexed':
            self.tfidf_matrix = self.vectorizer.fit_transform(records)

    def retrieve(self, query):
        if self.method == 'keyword':
            return self.keyword_search(query)
        elif self.method == 'vector':
            return self.vector_search(query)
        elif self.method == 'indexed':
            return self.indexed_search(query)
        
    def keyword_search(self, query):
        """Returns the best-matching document accoring
        to the greatest amount of shared keywords

        Args:
            query (str): The input query

        Returns:
            str: The best-matching document
        """
        best_score = 0
        best_record = None
        query_keywords = set(query.lower().split())
        for index, doc in enumerate(self.documents):
            doc_keywords = set(doc.lower().split())
            common_keywords = query_keywords.intersection(doc_keywords)
            score = len(common_keywords)
            if score > best_score:
                best_score = score
                best_record = self.documents[index]
        return best_record
    
    def vector_search(self, query):
        """Searches for the lowest-distance document in the
        knowledge-base vector representation against our query

        Args:
            query (str): The input query

        Returns:
            str: The best-matching document
        """
        query_tfidf = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_tfidf, self.tfidf_matrix)
        best_index = similarities.argmax()
        return db_records[best_index]
    
    def indexed_search(self, query):
        """Uses a pre-compiled IF-TDF matrix for fast retrieval of
        the best-matching document

        Args:
            query (str): The input query

        Returns:
            str: The best-matching document
        """
        # Assuming the tfidf_matrix is precomputed and stored
        query_tfidf = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_tfidf, self.tfidf_matrix)
        best_index = similarities.argmax()
        return db_records[best_index]

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

# We can now use our new class!
retrieval = RetrievalComponent(method='vector')  # Choose from 'keyword', 'vector', 'indexed'
retrieval.fit(db_records)
query = "define a rag store"
best_matching_record = retrieval.retrieve(query)
print_formatted_response(best_matching_record)
