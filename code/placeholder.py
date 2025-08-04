import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
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




      
 

