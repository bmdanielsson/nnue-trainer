import argparse
import model as M
import nnue_dataset
import halfkp
import torch
import ranger
import os.path
from torch.utils.data import DataLoader, Dataset

OUTPUT_DIR = 'output'
LATEST_LAST_PATH = ''
LATEST_BEST_PATH = ''
LATEST_EPOCH_PATH = ''

def save_model(nnue, output_path, epoch, idx, val_loss, new_best, epoch_end):
  global LATEST_LAST_PATH
  global LATEST_BEST_PATH
  global LATEST_EPOCH_PATH

  # Save the model as the latest version 
  if os.path.exists(LATEST_LAST_PATH):
    os.remove(LATEST_LAST_PATH) 
  last_path = f'{output_path}/last_epoch={epoch}_iter={idx+1}_loss={val_loss:.5f}.pt'
  LATEST_LAST_PATH = last_path
  torch.save(nnue.state_dict(), last_path)
  
  # Save the model as the new best version
  if new_best:
    if os.path.exists(LATEST_BEST_PATH):
      os.remove(LATEST_BEST_PATH) 
    best_path = f'{output_path}/best_epoch={epoch}_iter={idx+1}_loss={val_loss:.5f}.pt'
    LATEST_BEST_PATH = best_path
    torch.save(nnue.state_dict(), best_path)
    
  # Save the model as the final version for this epoch
  if epoch_end: 
    epoch_path = f'{output_path}/epoch={epoch}_loss={val_loss:.5f}.pt'
    LATEST_EPOCH_PATH = epoch_path
    torch.save(nnue.state_dict(), epoch_path)
  

def prepare_output_directory():
  path = os.getcwd() + '/' + OUTPUT_DIR 
  if not os.path.exists(path):
    os.mkdir(path)
  val = 1
  while os.path.exists(path+'/'+str(val)):
    val += 1
  path = path + '/' + str(val)
  os.mkdir(path)
  return path


def calculate_validation_loss(nnue, val_data_loader, lambda_):
  nnue.eval()
  with torch.no_grad():
    val_loss = []
    for k, sample in enumerate(val_data_loader):
      us, them, white, black, outcome, score = sample
      pred = nnue(us, them, white, black)
      loss = M.loss_function(lambda_, pred, sample)
      val_loss.append(loss)
  
    val_loss = torch.mean(torch.tensor(val_loss))
    nnue.train()
    
  return val_loss
  
  
def train_step(nnue, sample, optimizer, lambda_, epoch, idx, num_batches):
  us, them, white, black, outcome, score = sample
  
  pred = nnue(us, them, white, black)
  loss = M.loss_function(lambda_, pred, sample)
  print(f'Epoch {epoch}, {int(((idx+1)/num_batches)*100.0)}% ({idx+1}/{num_batches}) => {loss.item():.5f}', end='\r')
  loss.backward()
  optimizer.step()
  nnue.zero_grad()
  
  
def create_data_loaders(train_filename, val_filename, features_name, num_workers, batch_size, filtered, random_fen_skipping, main_device, training_size, validation_size):
  epoch_size = training_size
  val_size = validation_size
  train_infinite = nnue_dataset.SparseBatchDataset(features_name, train_filename, batch_size, num_workers=num_workers,
                                                   filtered=filtered, random_fen_skipping=random_fen_skipping, device=main_device)
  val_infinite = nnue_dataset.SparseBatchDataset(features_name, val_filename, batch_size, filtered=filtered,
                                                   random_fen_skipping=random_fen_skipping, device=main_device)
  train = DataLoader(nnue_dataset.FixedNumBatchesDataset(train_infinite, (epoch_size + batch_size - 1) // batch_size), batch_size=None, batch_sampler=None)
  val = DataLoader(nnue_dataset.FixedNumBatchesDataset(val_infinite, (val_size + batch_size - 1) // batch_size), batch_size=None, batch_sampler=None)
  return train, val


def main(args):
  # Create directory to store data in
  output_path = prepare_output_directory()
    
  # Select which device to use
  if torch.cuda.is_available():
    main_device = 'cuda:0'
  else:
    main_device = 'cpu'
    
  # Set feature set
  features = halfkp
  features_name = features.Features.name

  # Configure batch size
  batch_size = args.batch_size
  if batch_size <= 0:
    batch_size = 8192 if torch.cuda.is_available() else 128
  
  # Create data loaders
  train_data_loader, val_data_loader = create_data_loaders(args.train, args.val, features_name, args.num_workers, batch_size, args.smart_fen_skipping, args.random_fen_skipping, main_device, args.training_size, args.validation_size)
  
  # Create model
  nnue = M.NNUE(feature_set=features.Features()).to(main_device)
  if args.resume_from_model:
    nnue.load_state_dict(torch.load(args.resume_from_model))

  # Configure optimizer
  optimizer = ranger.Ranger(nnue.parameters(), lr=1e-3)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True, min_lr=1e-6)

  # Print configuration info
  print(f'Training set: {args.train} ({args.training_size})')
  print(f'Validation set: {args.val} ({args.validation_size})')
  print(f'Batch size: {batch_size}')
  print(f'Smart fen skipping: {args.smart_fen_skipping}')
  print(f'Random fen skipping: {args.random_fen_skipping}')
  print(f'Validation check interval: {args.val_check_interval}')
  print(f'Resuming from: {args.resume_from_model}')
  print('')

  # Main training loop
  best_val_loss = 1000000.0
  num_batches = len(train_data_loader)
  epoch = 1
  while True:
    for k, sample in enumerate(train_data_loader):
      train_step(nnue, sample, optimizer, args.lambda_, epoch, k, num_batches)
      
      if k > 0 and (k+1)%args.val_check_interval == 0:
        val_loss = calculate_validation_loss(nnue, val_data_loader, args.lambda_)
        new_best = False
        if (val_loss < best_val_loss):
          new_best = True
          best_val_loss = val_loss
        save_model(nnue, output_path, epoch, k, val_loss, new_best, False)
    
    val_loss = calculate_validation_loss(nnue, val_data_loader, args.lambda_)
    new_best = False
    if (val_loss < best_val_loss):
      new_best = True
      best_val_loss = val_loss
    save_model(nnue, output_path, epoch, num_batches-1, val_loss, new_best, True)
    print('')    

    scheduler.step(val_loss)
    epoch += 1


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Train a NNUE style network.')
  parser.add_argument('train', help='Training data (.bin or .binpack)')
  parser.add_argument('val', help='Validation data (.bin or .binpack)')
  parser.add_argument('--lambda', default=1.0, type=float, dest='lambda_', help='lambda=1.0 = train on evaluations, lambda=0.0 = train on game results, interpolates between (default=1.0).')
  parser.add_argument('--num-workers', default=1, type=int, help='Number of worker threads to use for data loading. Currently only works well for binpack.')
  parser.add_argument('--batch-size', default=-1, type=int, help='Number of positions per batch / per iteration. Default on GPU = 8192 on CPU = 128.')
  parser.add_argument('--smart-fen-skipping', action='store_true', help='If enabled positions that are bad training targets will be skipped during loading. Default: False')
  parser.add_argument('--random-fen-skipping', default=0, type=int, help='skip fens randomly on average random_fen_skipping before using one.')
  parser.add_argument('--resume-from-model', help='Initializes training using the weights from the given .pt model')
  parser.add_argument('--training-size', default=100000000, type=int, help='Number of training samples')
  parser.add_argument('--validation-size', default=1000000, type=int, help='Number of validation samples')
  parser.add_argument('--val-check-interval', default=2000, type=int, help='How often to check validation loss')
  args = parser.parse_args()
  
  main(args)
