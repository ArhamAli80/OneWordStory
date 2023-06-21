# import torch
# from backend.api.model import LanguageModel
#
# vocab_size = 1343
# embedding_dim = 100
# hidden_dim = 128
# model = LanguageModel(vocab_size,embedding_dim,hidden_dim)
#
# # Load the saved model state dictionary
# model.load_state_dict(torch.load('trained_model.pth'))
# word2idx = torch.load('word2idx.pth')
# idx2word = torch.load('idx2word.pth')
#
# # Set the model in evaluation mode
# model.eval()
#
# startWord = "once"
# input_tensor = torch.LongTensor([[word2idx[startWord]]])
# max_length = 2
#
# with torch.no_grad():
#     model.eval()
#     for _ in range(max_length):
#         output = model(input_tensor)
#         _, predicted_idx = torch.max(output[:, -1, :], 1)
#         predicted_word = idx2word[predicted_idx.item()]
#         print(predicted_word)
#         input_tensor = torch.cat((input_tensor, predicted_idx.unsqueeze(0)), dim=1)