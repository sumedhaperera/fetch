from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim

""""
A multitask learning model that adds a single hidden layer mlp head each for classification of two tasks.
Each mlp uses a hidden layer output size of 128. The input dependant on the embedding size (our model is 384).
We use a RELU activation function.
The transformer weight are frozen but can be unfrozen and finetuned if need be. We freeze them to 
avoid overfitting/catastrophic forgetting.
"""
class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self, model_name='all-MiniLM-L6-v2', num_classes_A=3, num_classes_B=3, fine_tune=False):
        super(MultiTaskSentenceTransformer, self).__init__()
        self.encoder = SentenceTransformer(model_name)
        #Freeze the model by default
        if not fine_tune:
            for param in self.encoder.parameters():
                param.requires_grad = False 
        # Define mlp blocks for classification tasks. 
        self.classifier_A = nn.Sequential(
            nn.Linear(self.encoder.get_sentence_embedding_dimension(), 128),
            nn.ReLU(), 
            nn.Linear(128, num_classes_A)
        )

        self.classifier_B = nn.Sequential(
            nn.Linear(self.encoder.get_sentence_embedding_dimension(), 128), 
            nn.ReLU(), 
            nn.Linear(128, num_classes_B) 
        )

   
    def forward(self, sentences):
        #use embeddings from the transformer to feed the classifiers as features.
        embeddings = self.encoder.encode(sentences, convert_to_tensor=True)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        output_A = self.classifier_A(embeddings)
        output_B = self.classifier_B(embeddings)
        return output_A, output_B

def train_model(model, sentences, labels_A, labels_B, epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
   
    for epoch in range(epochs):
        optimizer.zero_grad()
        output_A, output_B = model(sentences)
        loss_A = loss_fn(output_A, labels_A)
        loss_B = loss_fn(output_B, labels_B)
        loss = loss_A + loss_B  
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}, loss_A: {loss_A.item()}, loss_B: {loss_B.item()}")

if torch.cuda.is_available():
    device = torch.device("cuda") 
elif torch.backends.mps.is_available():
    device = torch.device("mps")  
else:
    device = torch.device("cpu") 

print(f"Using device: {device}")

model = MultiTaskSentenceTransformer().to(device)

test_sentences = [
    "Implement a sentence transformer model",
    "Encode input sentences into fixed-length embeddings",
    "Test your implementation",
    "Showcase the obtained embeddings"
]


labels_A = torch.tensor([0, 1, 2, 1]).to(device)
labels_B = torch.tensor([1, 2, 0, 2]).to(device)  

train_model(model, test_sentences, labels_A, labels_B)
