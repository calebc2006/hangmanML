# hangmanML
This model's architecture is heavily inspired by a hybrid RNN model by [methi1999](https://github.com/methi1999/hangman).

### TODO:
- [x] Version control for model
- [ ] Better epoch system for training
- [ ] KeyboardInterrupt handling

- [ ] Add support for hints

<br>

## Training
`train.py` is the main file used for training.

The training process will train to a specified `weights_path` within `/prev`, being able to leverage the pre-trained models provided. If `weights_path` is not found, it will create the file and train from scratch.

**Model version** can be configured by specifying version (eg. `"v2.4"`) in `train.py` or `test.py`. Version config files are placed in the `/config` directory, and are named `<version_name>.yaml`

<br>

The model will be provided with a increasing difficulty of training examples as the epoch number increases. This relative difficulty is based on the fraction $current\:epoch \over number\:of\:epochs$.

**Don't be distressed** when you see the train/val loss increasing with epoch count. This is a natural consequence of increasing training difficulty with each subsequent epoch.

<br>

## Testing
`test.py` is the main file used for testing.

It currently has 2 modes which need to be manually switched between in the code:

- The first mode (enabled by default) is a comprehensive test over the words in the dataset, returning the mean and median number of tries taken to guess the correct word in a simulated game. It also generates a histogram of the distribution of the number of tries needed to solve each word, saved in `/test-results`

- The second mode allows the user to specify a single word of their choice for the model to solve.

<br>

## Model Architecture 

Input: [encodedWords, wordLengths, guessedLetters]
- pack_padded_sequence
- rnn encoding model (LSTM)
- encoding of prev_guesses
- concat
- linear layer
- letter guess output

#### [insert flowchart]
<br>