from model import MainModel, ModelConfig, getLetterFromOutputs, unencode
from dataloader import DataLoader
import time

class Trainer():
    def __init__(self, model: MainModel, dataloader: DataLoader):
        self.model = model
        self.dataloader = dataloader
        
        self.fraction_correct = 0.5
        self.fraction_completed = 0.8
        
    def train(self, batch_size, num_epochs, start_epoch=1, save_path=f'/prev/untitled'):
        batches_per_epoch = int(self.dataloader.get_num_words() / batch_size)
        print(f'\nTraining to: {save_path}')
        
        for epoch_idx in range(start_epoch, num_epochs+1):
            self.dataloader.shuffle()
            self.fraction_correct = 0.5 + 0.3 * (epoch_idx / num_epochs)    # Slightly increase accuracy over time
            self.fraction_completed = 0.8 - 0.6 * (epoch_idx / num_epochs)  # Decrease fraction completed over time
            
            for g in self.model.optimizer.param_groups:
                g['lr'] = 0.0005 - 0.00049 * (epoch_idx / num_epochs)   # Decrease learning rate over time
            
            epoch_loss = 0
            batch_number = 1
            start_time = time.time()

            print('\n------------------------------------')
            print(f'Starting Epoch #{epoch_idx} ...')
            print(f'Total Batches: {batches_per_epoch}')
            
            for batch_number in range(1, batches_per_epoch): 
                start_idx = (batch_number - 1)* batch_size 
                [encoded_words, prev_guesses, word_lengths, labels] = self.dataloader.get_train_batch(batch_size, 
                                                                                                start_idx, 
                                                                                                self.fraction_correct, 
                                                                                                self.fraction_completed)
                
                self.model.optimizer.zero_grad()
                
                outputs = self.model(encoded_words, word_lengths, prev_guesses)
                loss, miss_penalty = self.model.calculate_loss(outputs, labels, word_lengths, prev_guesses)
                loss.backward()
                self.model.optimizer.step()
                
                epoch_loss += loss.item()
                print(f'Batch #{batch_number}: Loss = {round(loss.item(), 5)}', end='\r')
                
            print(f'Epoch Loss: {round(epoch_loss / batches_per_epoch, 5)}              ')
            print(f'Time Taken: {round((time.time() - start_time), 2)}s')
            self.model.save(path=save_path)
            
            self.validate(batch_size=100)
    
    
    def validate(self, batch_size):
        [encoded_words, prev_guesses, word_lengths, labels] = self.dataloader.get_train_batch(batch_size, 
                                                                                              start_idx=0, 
                                                                                              fraction_correct=self.fraction_correct, 
                                                                                              fraction_completed=self.fraction_completed)
        
        output = self.model(encoded_words, word_lengths, prev_guesses)
        val_loss, miss_penalty = self.model.calculate_loss(output, labels, word_lengths, prev_guesses)
        print(f'Validation Loss: {round(val_loss.item(), 5)}, Miss: {round(miss_penalty.item(), 2)}')
        for i in range(batch_size):
            cur_output = output[i].tolist()
            cur_prev = prev_guesses[i].tolist()
            target = encoded_words[i].tolist()
            length = word_lengths[i].tolist()
            letterIdx = getLetterFromOutputs(cur_output, cur_prev)
            
            if (i % 20 == 0): # Only show some val examples for brevity
                print(f'{unencode(target, length)}: {chr(letterIdx + 97)}')            
            
            
def main():
    dataloader = DataLoader('10k.txt')
    model_config = ModelConfig()
    save_path = 'prev/v2.2-10k-200.pth'
    
    # Load model (if path does not exist, starts from scratch)
    model = MainModel(model_config, weights_path=save_path)
    
    # Create trainer
    trainer = Trainer(model, dataloader)
    trainer.train(batch_size=20, num_epochs=200, start_epoch=1, save_path=save_path)

            
if __name__ == "__main__":
    main()
    