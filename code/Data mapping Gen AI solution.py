import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('/content/sample_data/data/mapping_metadata.csv')


# Define a function that will combine the entire fields scope
def knowledge_base(data):
  data['combined_text'] = (
    data['Source Column'].fillna('') + ' | ' +
    data['Source Field Description'].fillna('') + ' | ' +
    data['Transformation Rule'].fillna('') + ' | ' +
    data['Target Column'].fillna('') + ' | ' +
    data['Target Field Description'].fillna('')
)
  return data

new_data = knowledge_base(df) # to be used in main
#new_data[['combined_text','Source Column']] # preview the important segment---can be removed"

def get_embedding(combined_text, model="all-MiniLM-L6-v2"):
   
  model = SentenceTransformer(model)

  embedding = model.encode(combined_text.tolist())
  return embedding


#preview the embeddings generated#
new_data = knowledge_base(df)
embedding = get_embedding(new_data['combined_text'])
print(embedding)


# Example user query (dynamic) -- you can take input from chat
def user_query(data,embedding,model="all-MiniLM-L6-v2"):

  query = "How do I transform the order_amount field?"
  model = SentenceTransformer(model)
# Generate the embedding for the user query
  query_embedding = model.encode(query)

# Calculate cosine similarity between the query and the knowledge base embeddings
  cos_similarities = cosine_similarity(query_embedding,embedding)

# Find the index of the most similar knowledge base entry
  most_similar_idx = cos_similarities.argmax()

# Retrieve the most similar knowledge base entry (combined text)
  most_similar_entry = data.iloc[most_similar_idx]['combined_text']
  print(f"Most similar knowledge base entry: {most_similar_entry}")

user_query(new_data,embedding)




      
 

