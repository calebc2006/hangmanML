import numpy as np
import math
import time
import random
import torch

MAX_WORD_LEN = 20
MIN_WORD_LEN = 3
random.seed(42)

isCuda = torch.cuda.is_available()
device = torch.device('cpu')
if isCuda:
    device = torch.device('cuda:0')

def get_all_words(filename):
    with open(f'./data/{filename}', 'r') as f:
        raw_words = [word.strip() for word in f.read().splitlines()]
        words = []
        for word in raw_words:
            if len(word) >= MIN_WORD_LEN and len(word) <= MAX_WORD_LEN:
                words.append(Word(word))
        
        return np.array(words)

class Word():
    def __init__(self, word_str: str):
        assert len(word_str) <= MAX_WORD_LEN, f'Word {word_str} is too long!'
        assert len(word_str) > 0, 'Blank word found!'
        
        self.string = word_str
        self.length = len(word_str)
        self.num_letters = len(np.unique(list(word_str)))
        self.blank = '_'
        
        if self.blank in self.string:
            self.num_letters -= 1
        
    def encode(self, letters_to_use=[1]*26): # by default, use all letters (a-z)
        '''
        Returns an tensor of one-hot encoded words, shape (MAX_WORD_LEN, 28)
        Letters indexed (0-25), index 26 is used for blanks, index 27 is used for padding
        '''
        encoded_word = []
        
        for letter in self.string:
            cur_letter = [0] * 28
            letter_idx = ord(letter) - 97
            if letter != self.blank and letters_to_use[letter_idx]:
                cur_letter[letter_idx] = 1
            else:
                cur_letter[26] = 1
                
            encoded_word.append(cur_letter)
         
        while len(encoded_word) < MAX_WORD_LEN:
            cur_letter = [0] * 28
            cur_letter[27] = 1
            encoded_word.append(cur_letter)
        return encoded_word
    
    def encode_label(self):
        encoded = [0]*26
        for letter in self.string:
            if letter == self.blank:
                continue
            encoded[ord(letter) - 97] = 1
        return encoded
    
    def get_letters(self, within=True):
        letters = []
        others = list(range(26))
        for letter in self.string:
            if letter == self.blank:
                continue
            letter_idx = ord(letter) - 97
            letters.append(letter_idx)
            if letter_idx in others:
                others.remove(letter_idx)
        
        if within:
            return letters
        return others
    
def tensorize(list, value_type=torch.float32):
    return torch.tensor(list).to(device).to(value_type)

class DataLoader():
    def __init__(self, filename, verbose=False):
        self.verbose = verbose
        self.words = get_all_words(filename)
        
        self.shuffle()
        
    def get_num_words(self):
        return len(self.words)
    
    
    def get_train_batch(self, batch_size: int, start_idx: int, fraction_correct: float, fraction_completed: float):
        '''
        Returns [encoded_words, prev_guesses, word_lengths, labels], each of length batchSize    
        '''
        encoded_words, word_lengths, prev_guesses, labels = [], [], [], []
        for i in range(start_idx, start_idx+batch_size):
            word: Word = self.words[i]
            
            if self.verbose:
                print(f'Word: {word.string}')
            
            word_lengths.append(word.length)
            labels.append(word.encode_label())            
            
            num_correct_letters = math.floor(fraction_completed * len(word.get_letters(True)))
            correct_letters_idx = random.sample(word.get_letters(True), num_correct_letters)
            used_letters = [0] * 26
            for idx in correct_letters_idx:
                used_letters[idx] = 1
            encoded_words.append(word.encode(used_letters))            
            
            num_wrong_guesses = min(math.floor(num_correct_letters / fraction_correct), 26-num_correct_letters)
            num_wrong_guesses = min(len(word.get_letters(False)), num_wrong_guesses)
            
            wrong_letters_idx = random.sample(word.get_letters(False), num_wrong_guesses)
            for idx in wrong_letters_idx:
                used_letters[idx] = 1
            prev_guesses.append(used_letters)

        return [tensorize(encoded_words, torch.bool), 
                tensorize(prev_guesses, torch.bool), 
                tensorize(word_lengths, torch.int8), 
                tensorize(labels, torch.bool)]
    
    def shuffle(self):
        random.shuffle(self.words)



# TESTING:

def main():
    print("Initializing DataLoader...")
    dl = DataLoader(filename='100k.txt', verbose=False)

    print("Getting Batch...")
    start_time = time.process_time_ns()
    batch_size = 500
    batch = dl.get_train_batch(batch_size=batch_size, start_idx=0, num_guesses=10, fraction_completed=0.4)

    time_taken = time.process_time_ns() - start_time

    print(f'Batch: {batch[0].shape}')

    print(f'Time Taken (batch_size={batch_size}): {time_taken/1000000}ms')
        

# if __name__ == '__main__':
#     main()
        
