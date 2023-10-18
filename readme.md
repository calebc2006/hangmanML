## hangmanML
This model's architecture is heavily based on a hybrid RNN model by methi1999 (https://github.com/methi1999/hangman). 

`train.py` and `test.py` are the main executable files. Model and version can be configured in ModelConfig class (more functionality coming soon)

### Model Architecture 

Input: [encodedWords, wordLengths, guessedLetters]
- pack_padded_sequence
- rnn encoding model (LSTM)
- encoding of prev_guesses
- concat
- linear layer
- letter guess output

#### [insert flowchart]
<br />

### Note about training:
Don't be distressed when you see the train/val loss increasing with epoch count. This is a natural consequence of the increasing difficulty of the scenarios presented to the model.
<br />

### TODO:
- [ ] Version control for model
- [ ] Better epoch system for training
- [ ] KeyboardInterrupt handling