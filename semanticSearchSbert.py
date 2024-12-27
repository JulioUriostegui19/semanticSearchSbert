"""
#----------------------------------------------------------------#
# Semantic Search with Sbert                                     #
# date: 31.10.2024                                               #
# author: Julio Uriostegui                                       #
# email: uriosteguisanchez@campus.tu-berlin.de                   #
#----------------------------------------------------------------#
"""
from search import search
import argparse
from pprint import pprint
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np
import sys
import json

# Argument parsing setup
parser = argparse.ArgumentParser(description="Search for books based on a query and model type.")
parser.add_argument("--query", type=str, required=True, help="The search query")
parser.add_argument("--model_type", type=str, required=True, help="Choose between 'original' and 'fine_tuned' model")
parser.add_argument("--top_k", type=int, default=5, help="The number of top results to return")

args = parser.parse_args()

# Load index and model with error handling for invalid model names
try:
    if args.model_type == "original":
        index = faiss.read_index('book_plot.index')
        model = SentenceTransformer("msmarco-distilbert-base-dot-prod-v3")
    elif args.model_type == "fine_tuned":
        index = faiss.read_index('book_plot_finetuned.index')
        model = SentenceTransformer('fine_tuned/sbert_semantic_search-model')
    else:
        raise ValueError("Invalid model type specified. Choose between 'original' and 'fine_tuned'.")
except ValueError as e:
    print(e)
    sys.exit(1)

# Run the search with the provided query, top_k, and model
results = search(args.query, top_k=args.top_k, index=index, model=model)

pd.DataFrame(results).to_csv('results.csv')

# Transform into a dictionary of dictionaries with enumerated keys for JSON
formatted_data = {f"{i}": entry for i, entry in enumerate(data)}

# Save to a JSON file
output_file  = "results.json"
with open(output_file, "w") as file:
    json.dump(formatted_data, output_file, indent=4)  # Use indent=4 for formatting
    
#print("\nResults:")
#for result in results:
#    pprint(result)
