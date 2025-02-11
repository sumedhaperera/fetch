from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
"""
his is a simple sentence transformer that uses a pretrained model (all-MiniLM-L6-v2). 
This transformer model is chosen because it has been trained specifically for sentence embeddings and returns
fixed length embeddings.
The forward function simply calls this model to encode sentences.
Since the selected model already provides senetence level embedding, we do not need to pool the output, nor 
do we need to use padding as they are of fixed length.
We do, however, normalize the output so that if do need to compare embedding we already have it normalized.
Since we are not doing any training and use the existing weights we use orch.no_grad during inference.
"""

class SimpleSentenceTransformer(nn.Module):
    def __init__(self, model_name= "all-MiniLM-L6-v2"):
        super(SimpleSentenceTransformer, self).__init__()
        self.encoder = SentenceTransformer(model_name)
        
    def forward(self, sentences):
        #Handle both an imput array and a string.
        if isinstance(sentences, str):
            sentences = [sentences]
        embeddings = self.encoder.encode(sentences, convert_to_tensor=True)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings
    
# Get device
if torch.cuda.is_available():
    device = torch.device("cuda") 
elif torch.backends.mps.is_available():
    device = torch.device("mps")  
else:
    device = torch.device("cpu") 

print(f"Device: {device}")

model = SimpleSentenceTransformer().to(device)

test_sentences = [
    "Implement a sentence transformer model",
    "Encode input sentences into fixed-length embeddings",
    "Test your implementation",
    "Showcase the obtained embeddings"
]

#No need to calculate gradients during inference.
with torch.no_grad(): 
    embeddings = model(test_sentences).to(device)

print("Shape:", embeddings.shape) 
print("Embeddings:\n", embeddings)