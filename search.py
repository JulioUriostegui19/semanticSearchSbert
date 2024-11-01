"""
#----------------------------------------------------------------#
# Semantic Search with Sbert                                     #
# date: 31.10.2024                                               #
# author: Julio Uriostegui                                       #
# email: uriosteguisanchez@campus.tu-berlin.de                   #
#----------------------------------------------------------------#
"""
# Auxiliary functions
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np
import time
# This is only a dummy function and it should be change to get
#information from the Database
def fetch_info(dataframe_idx, df):
    # import data
    df = pd.read_csv('english_books_faiss.csv',memory_map=True)
    info = df.iloc[dataframe_idx]
    meta_dict = {}
    meta_dict['book_title'] = info['book_title']
    return meta_dict

# Search auxiliary function
def search(query, top_k, index, model):
    t=time.time()
    # encode the query to vector
    query_vector = model.encode([query])
    # similarity search
    top_k = index.search(query_vector, top_k)
    print('>>>> Results in Total Time: {}'.format(time.time()-t))
    # Postprocessing of the output to get a list of the IDs
    top_k_ids = top_k[1].tolist()[0]
    top_k_ids = list(np.unique(top_k_ids))
    # Get metadata from Database
    results =  [fetch_info(idx) for idx in top_k_ids]
    return results
