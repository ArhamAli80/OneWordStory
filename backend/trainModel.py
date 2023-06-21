import torch
import torch.nn as nn
import torch.optim as optim
from back_end.api.model import LanguageModel
import string
import nltk
nltk.download('punkt')

with open('Sentences2.txt', 'r', encoding='utf-8') as file:
    file_content = file.read()

def getPunctuation(text):
    punctuation_marks = string.punctuation
    marks = []
    for char in text:
        if char in punctuation_marks:
            marks.append(char)
    return set(marks)

def removePunctuation(text, marks):
    for mark in marks:
        text = text.replace(mark, '')
    return text

def generateTrainingSamples(text):
    trainingSamples = []
    tokenizer = nltk.tokenize.word_tokenize

    for sentence in text:
        tokens = tokenizer(sentence)
        for i in range(len(tokens) - 1):
            input_word = tokens[i]
            output_word = tokens[i + 1]
            trainingSamples.append((input_word, output_word))

    return trainingSamples

unwantedMarks = ['=', '*', '&', "", '+', '"', ',', '@', '(', ';', '?', ':', '#', '/', '%', ')', '!', '$']
textData = removePunctuation(file_content, unwantedMarks)


#split into sentences
textData = textData.split('.')
trainingSamples = generateTrainingSamples(textData)
vocab = set()
for inputWord, outputWord in trainingSamples:
    vocab.add(inputWord)
    vocab.add(outputWord)

word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for idx, word in enumerate(vocab)}

vocab_size = len(vocab)
print(vocab_size)
embedding_dim = 100
hidden_dim = 128

# Convert training samples to input and target tensors
inputs = torch.LongTensor([[word2idx[inputWord]] for inputWord, _ in trainingSamples])
targets = torch.LongTensor([[word2idx[outputWord]] for _, outputWord in trainingSamples])

# Create and train the language model
model = LanguageModel(vocab_size, embedding_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

torch.save(model.state_dict(), 'back_end/model.pth')
torch.save(word2idx, 'back_end/word2idx.pth')
torch.save(idx2word, 'back_end/idx2word.pth')

