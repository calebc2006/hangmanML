## Model
This model is heavily based on a hybrid RNN model by methi1999 (https://github.com/methi1999/hangman)


Input: [encodedWords, wordLengths, guessedLetters]
- pack_padded_sequence
- rnn encoding model (LSTM)
- encoding of prev_guesses
- concat
- linear layer
- letter guess output

### Note about training:
Don't be distressed when you see the train/val loss increasing with epoch count. This is a natural consequence of the increasing difficulty of the scenarios presented to the model.

### TODO:
- idk
- More checks and asserts
- 