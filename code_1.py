from docx import Document

def read_word_file(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

document_path = 'data/cano.txt'
document_text = read_word_file(document_path)


from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the document
embeddings = model.encode([document_text])


# Initialize FAISS index
d = embeddings.shape[1]  # dimension of embeddings
index = faiss.IndexFlatL2(d)

# Add embeddings to the index
index.add(np.array(embeddings, dtype=np.float32))

# Save the index to a file (optional)
faiss.write_index(index, 'document_embeddings.index')

# To query, encode the query text
query = "Your query text goes here."
query_embedding = model.encode([query])

# Search the FAISS index
D, I = index.search(np.array(query_embedding, dtype=np.float32), k=1)  # k is the number of nearest neighbors to retrieve

# Get the result
print(f"Nearest neighbor ID: {I[0][0]}, Distance: {D[0][0]}")