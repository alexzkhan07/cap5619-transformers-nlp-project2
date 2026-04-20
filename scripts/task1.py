import torch
import torch.nn.functional as F
import os
from transformers import BertTokenizer, BertModel

# 1. Configuration
DATA_FILE = "./data/raw/word-test.v1.txt"
OUTPUT_FILE = "./results/task1_results.txt"
TARGET_GROUPS = [": city-in-state", ": family", ": gram2-opposite"]
K_VALUES = [1, 2, 5, 10, 20]
MODEL_NAME = "bert-base-uncased"

def load_and_parse_data():
    """Parses the text file and groups the analogies."""
    groups_data = {group: [] for group in TARGET_GROUPS}
    current_group = None
    
    with open(DATA_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line is a group header
            if line.startswith(":"):
                current_group = line
                continue
            
            # If we are in one of the target groups, parse the analogy
            if current_group in TARGET_GROUPS:
                words = line.lower().split()
                if len(words) == 4:
                    groups_data[current_group].append(words)
                    
    return groups_data

def get_word_embedding(word, tokenizer, model):
    """
    Tokenizes a word and returns its embedding. 
    If it splits into multiple tokens, averages their embeddings.
    """
    # Tokenize without special tokens like [CLS] and [SEP]
    tokens = tokenizer(word, return_tensors='pt', add_special_tokens=False)
    input_ids = tokens['input_ids']
    
    with torch.no_grad():
        # Access the initial word embeddings directly
        # model.get_input_embeddings() returns the embedding layer
        embeddings = model.get_input_embeddings()(input_ids)
    
    # embeddings shape: (1, num_tokens, hidden_size)
    # Take the mean across the num_tokens dimension (dim=1)
    return embeddings.mean(dim=1).squeeze(0)

def evaluate_group(group_name, analogies, tokenizer, model, out_file):
    """Runs the prediction task for a specific group."""
    print(f"\nEvaluating Group: {group_name}")
    print(f"Total analogies: {len(analogies)}")
    
    # 1. Build candidate pool (2nd and 4th words of all lines in this group)
    candidates = set()
    unique_words = set()
    for a, b, c, d in analogies:
        candidates.add(b)
        candidates.add(d)
        unique_words.update([a, b, c, d])
        
    candidates = list(candidates)
    
    # 2. Pre-compute embeddings for all unique words to save time
    print("Pre-computing embeddings...")
    embeddings_cache = {}
    for word in unique_words:
        embeddings_cache[word] = get_word_embedding(word, tokenizer, model)
        
    candidate_embeddings = torch.stack([embeddings_cache[w] for w in candidates])
    
    # 3. Evaluate analogies
    correct_cosine = {k: 0 for k in K_VALUES}
    correct_l2 = {k: 0 for k in K_VALUES}
    
    print("Running predictions...")
    for a, b, c, d in analogies:
        emb_a = embeddings_cache[a]
        emb_b = embeddings_cache[b]
        emb_c = embeddings_cache[c]
        
        # Calculate target vector: a - b
        target_vec = emb_a - emb_b
        
        # Calculate candidate vectors: c - d' (for all d' in candidate pool)
        # Broadcasting allows us to do this efficiently for all candidates at once
        cand_vecs = emb_c.unsqueeze(0) - candidate_embeddings
        
        # Metric 1: Cosine Similarity (Higher is better)
        # F.cosine_similarity expects inputs of shape (batch, dim)
        cos_sims = F.cosine_similarity(target_vec.unsqueeze(0), cand_vecs)
        
        # Metric 2: L2 Distance (Lower is better)
        l2_dists = torch.norm(target_vec.unsqueeze(0) - cand_vecs, p=2, dim=1)
        
        # Find ranks
        # Sort cosine descending
        sorted_indices_cos = torch.argsort(cos_sims, descending=True)
        # Sort L2 ascending
        sorted_indices_l2 = torch.argsort(l2_dists, descending=False)
        
        # Check ranks
        for k in K_VALUES:
            top_k_cos = [candidates[idx] for idx in sorted_indices_cos[:k]]
            top_k_l2 = [candidates[idx] for idx in sorted_indices_l2[:k]]
            
            if d in top_k_cos:
                correct_cosine[k] += 1
            if d in top_k_l2:
                correct_l2[k] += 1
                
    # 4. Save results table to file
    total = len(analogies)
    out_file.write(f"\nGroup: {group_name}\n")
    out_file.write(f"Total analogies: {total}\n")
    out_file.write("| k  | Accuracy (Cosine) | Accuracy (L2) |\n")
    out_file.write("|----|-------------------|---------------|\n")
    for k in K_VALUES:
        acc_cos = (correct_cosine[k] / total) * 100
        acc_l2 = (correct_l2[k] / total) * 100
        out_file.write(f"| {k:<2} | {acc_cos:>16.2f}% | {acc_l2:>12.2f}% |\n")
    out_file.write("\n")
    print(f"Finished evaluating {group_name}. Results written to file.")
        
def main():
    groups_data = load_and_parse_data()
    
    print("Loading BERT Tokenizer and Model...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME)
    model.eval() # Set to evaluation mode
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    print(f"\nResults will be saved to: {OUTPUT_FILE}")
    
    with open(OUTPUT_FILE, 'w') as out_file:
        for group_name in TARGET_GROUPS:
            analogies = groups_data[group_name]
            if not analogies:
                print(f"Warning: No data found for group '{group_name}'")
                continue
            evaluate_group(group_name, analogies, tokenizer, model, out_file)

if __name__ == "__main__":
    main()