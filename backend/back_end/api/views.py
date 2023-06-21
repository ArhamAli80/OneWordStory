from django.shortcuts import render
from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .models import Story
import torch
from .model import LanguageModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@api_view(['GET'])
def generate_word(request):
    # Load the saved model state dictionary
    model_state_dict = torch.load('model.pth')
    vocab_size = 4036
    embedding_dim = 100
    hidden_dim = 128

    # Create an instance of the language model
    model = LanguageModel(vocab_size, embedding_dim, hidden_dim)
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()

    word2idx = torch.load('word2idx.pth')
    idx2word = torch.load('idx2word.pth')

    if request.method == 'GET':
        input_word = request.GET.get('word', '')

        if not input_word:
            return Response({'error': 'Input word is empty'})

        try:
            input_tensor = torch.LongTensor([[word2idx[input_word]]]).to(device)
        except KeyError:
            return Response({'error': 'Input word not found in vocabulary'})

        generated_words = []
        with torch.no_grad():
            input_tensor = torch.LongTensor([[word2idx[input_word]]]).to(device)
            max_length = 3

            for _ in range(max_length):
                output = model(input_tensor)
                _, predicted_idx = torch.max(output[:, -1, :], 1)
                predicted_word = idx2word[predicted_idx.item()]
                input_tensor = torch.cat((input_tensor, predicted_idx.unsqueeze(0)), dim=1)
                generated_words.append(predicted_word)


        # Return the generated word as a JSON response
        response_data = {
            'generated_word': generated_words
        }

        return Response(response_data)


# Create your views here.
