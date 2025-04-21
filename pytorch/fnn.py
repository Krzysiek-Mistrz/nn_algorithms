import re
import nltk 
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.vocab import build_vocab_from_iterator

#preprocessing & vocab
def clean_and_tokenize(text):
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    tokens = word_tokenize(text)
    return [t.lower() for t in tokens if t]

with open("data/song.txt", "r", encoding="utf-8") as f:
    raw = f.read()
tokens = clean_and_tokenize(raw)
vocab = build_vocab_from_iterator([tokens], specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

#trigram
CONTEXT_SIZE = 2
trigrams = [
    (tokens[i-CONTEXT_SIZE:i], tokens[i])
    for i in range(CONTEXT_SIZE, len(tokens))
]
data = [
    ([vocab(tup) for tup in ctx], vocab([target]))
    for ctx, target in trigrams
]

#model
class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.linear1 = nn.Linear(context_size * embed_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        embeds = embeds.view(embeds.size(0), -1)
        out = F.relu(self.linear1(embeds))
        logits = self.linear2(out)
        return logits

#training
EMBED_DIM = 10
BATCH_SIZE = 16
EPOCHS     = 50
lr         = 0.01

model = NGramLanguageModeler(len(vocab), EMBED_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    total_loss = 0
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i:i+BATCH_SIZE]
        contexts = torch.tensor([b[0] for b in batch])
        targets  = torch.tensor([b[1] for b in batch]).squeeze()

        optimizer.zero_grad()
        logits = model(contexts)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss/len(data):.4f}")

#generating text
def generate(model, vocab, start_words, length=50):
    model.eval()
    w1, w2 = start_words
    output = [w1, w2]
    for _ in range(length):
        idxs = torch.tensor([[vocab(w1), vocab(w2)]])
        with torch.no_grad():
            logits = model(idxs)
            next_idx = torch.argmax(logits, dim=1).item()
        next_word = vocab.get_itos()[next_idx]
        output.append(next_word)
        w1, w2 = w2, next_word
    return " ".join(output)

print(generate(model, vocab, ["never", "gonna"], length=30))
