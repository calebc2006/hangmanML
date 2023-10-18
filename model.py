import torch
import torch.nn as nn
import numpy as np
import os

class MainModel(nn.Module):
    def __init__(self, config, weights_path='none'):
        if not isinstance(config, ModelConfig):
            raise TypeError('Invalid config format. Config must be an instance of ModelConfig')
        
        super().__init__()
        self.config = config
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        
        # input to RNN model is tensor (batch, letter sequence, 1-hot letter encoding)
        self.rnn = nn.LSTM(input_size=config.input_dim,
                        hidden_size=config.hidden_dim,
                        num_layers=config.num_layers,
                        dropout=config.dropout,
                        bidirectional=True, batch_first=True)
        
        # preprocess previous guesses before joining with RNN output
        self.prev_guess_layer = nn.Linear(26, config.prev_guess_dim)
        
        # final linear layers
        linear_input_dim = config.prev_guess_dim + config.hidden_dim*2 # x2 because of bidirectional RNN
        self.linear1 = nn.Linear(linear_input_dim, config.mid_dim)
        self.linear2 = nn.Linear(config.mid_dim, config.output_dim)
        self.relu = nn.ReLU()
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)

        if weights_path != 'none' and os.path.isfile(weights_path):
            self.load_state_dict(torch.load(weights_path))

    def forward(self, x, x_lengths, prev_guesses):
        """
        Forward pass through RNN
        x: input tensor of shape (batch size, max sequence length, encoding length (28))
        x_lengths: actual lengths of each sequence < max sequence length (since padded with zeros)
        prev_guesses: tensor of length batch_size x 26. 1 at index i indicates that ith character is NOT present
        return: tensor of shape (batch size, max sequence length, output dim)
        """     
        
        # Pack padded inputs and pass through RNN
        x = nn.utils.rnn.pack_padded_sequence(x.cpu(), x_lengths.cpu(), batch_first=True, enforce_sorted=False).to(torch.float32).to(self.device)
        output, hidden = self.rnn(x) # ignore outputs, just use 2nd hidden state, since RNN is for encoding only
    
        # Get RNN output
        hidden = hidden[-1].view(self.config.num_layers, 2, -1, self.config.hidden_dim)
        hidden = hidden[-1]
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.contiguous().view(hidden.shape[0], -1)
        
        # Pass prev_guesses through linear layer
        prev_guesses = self.prev_guess_layer(prev_guesses.to(torch.float32))

        # Concatenate hidden and prev_guesses, pass through final linear layers
        concatenated = torch.cat((hidden, prev_guesses), dim=1)
        return self.linear2(self.relu(self.linear1(concatenated)))
        
    def calculate_loss(self, model_out, labels, input_lengths, prev_guesses):
        outputs = nn.functional.log_softmax(model_out, dim=1)
        #calculate model output loss for how often it misses characters
        miss_penalty = torch.sum(outputs*prev_guesses, dim=(0,1))/outputs.shape[0]
        
        input_lengths = input_lengths.float()
        #weights per example is inversely proportional to length of word
        #this is because shorter words are harder to predict due to higher chances of missing a character
        weights_orig = (1/input_lengths)/torch.sum(1/input_lengths).unsqueeze(-1)
        weights = torch.zeros((weights_orig.shape[0], 1))    
        #resize so that torch can process it correctly
        weights[:, 0] = weights_orig

        #actual loss
        loss_func = nn.BCEWithLogitsLoss(weight=weights, reduction='sum')
        actual_penalty = loss_func(model_out, labels.to(torch.float32))
        return actual_penalty, miss_penalty

    def save(self, path):
        torch.save(self.state_dict(), path)


    def infer(self, encoded_word, word_length: int, prev_guesses):
        """
        encoded_word: one-hot encoded input word
        word_length: length of word 
        prev_guesses: array of length 26. 1 at index i indicates that ith character is NOT present
        return: tensor of shape (max sequence length, output dim)
        """

        encoded_word_t = torch.tensor([encoded_word])
        word_length_t = torch.tensor([word_length])
        prev_guesses_t = torch.tensor([prev_guesses])

        output = self.forward(encoded_word_t, word_length_t, prev_guesses_t).tolist()[0]
        return getLetterFromOutputs(output, prev_guesses)


class ModelConfig():
    def __init__(self):	
        self.input_dim = 28 # (0-25) for each letter, and one null value (26)
        self.hidden_dim = 256
        self.num_layers = 2
        self.dropout = 0.3
        
        self.prev_guess_dim = 128
        self.mid_dim = 128 
        self.output_dim = 26 # one for each letter
        self.lr = 0.0005

# Makes the model's guess, based on output while ensuring it does not repeat guesses
def getLetterFromOutputs(output, prev_guesses): 
    indices = np.flip(np.argsort(output))
    for i in indices:
        if not prev_guesses[i]:
            return i
        
def unencode(target, length):
    word = ''
    for i in range(length):
        isFilled = False
        for j in range(26):
            if target[i][j] == 1:
                word += chr(j + 97)
                isFilled = True
                break
        
        if not isFilled:
            word += '_'
    return word
