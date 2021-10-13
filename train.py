import argparse
import model as M
import nnue_dataset
import halfkp
import torch
import ranger
import os.path
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

SAMPLE_SIZE = 40
OUTPUT_DIR = 'output'
LOG_DIR = 'logs'
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
    if new_best and not epoch_end:
        if os.path.exists(LATEST_BEST_PATH):
            os.remove(LATEST_BEST_PATH) 
        best_path = f'{output_path}/best_epoch={epoch}_iter={idx+1}_loss={val_loss:.5f}.pt'
        LATEST_BEST_PATH = best_path
        torch.save(nnue.state_dict(), best_path)

    # Save the model as the final version for this epoch
    if epoch_end: 
        epoch_path = f'{output_path}/epoch={epoch}_loss={val_loss:.5f}.pt'
        LATEST_EPOCH_PATH = epoch_path
        LATEST_BEST_PATH = ''
        torch.save(nnue.state_dict(), epoch_path)
  

def prepare_output_directory():
    path = OUTPUT_DIR
    if not os.path.exists(path):
        os.mkdir(path)
    val = 1
    while os.path.exists(path+'/'+str(val)):
        val += 1
    path = path + '/' + str(val)
    os.mkdir(path)
    return path
  
  
def prepare_log_directory():
    path = LOG_DIR 
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

    return loss
  
  
def create_data_loaders(train_filename, val_filename, epoch_size, val_size,
                        batch_size, use_factorizer, main_device):
    train_dataset = nnue_dataset.SparseBatchDataset(train_filename, epoch_size,
            batch_size, use_factorizer,
            (epoch_size + batch_size - 1) // batch_size, device=main_device)
    val_dataset = nnue_dataset.SparseBatchDataset(val_filename, val_size,
            batch_size, use_factorizer,
            (val_size + batch_size - 1) // batch_size, device=main_device)

    train = DataLoader(train_dataset, batch_size=None, batch_sampler=None)
    val = DataLoader(val_dataset, batch_size=None, batch_sampler=None)

    return train, val


def main(args):
    # Select which device to use
    if torch.cuda.is_available():
        main_device = 'cuda:0'
    else:
        main_device = 'cpu'
    
    # Create directories to store data and logs in
    output_path = prepare_output_directory()
    log_path = prepare_log_directory()

    # Print configuration info
    print(f'Device: {main_device}')
    print(f'Training set: {args.train}')
    print(f'Validation set: {args.val}')
    print(f'Batch size: {args.batch_size}')
    print(f'Using factorizer: {args.use_factorizer}')
    print(f'Lambda: {args.lambda_}')
    print(f'Validation check interval: {args.val_check_interval}')
    print(f'Resuming from: {args.resume_from_model}')
    print(f'Logs written to: {log_path}')
    print(f'Data written to: {output_path}')
    print('')

    # Create log writer
    writer = SummaryWriter(log_path)

    # Create data loaders
    train_data_loader, val_data_loader = create_data_loaders(args.train, args.val, args.train_size, args.val_size, args.batch_size, args.use_factorizer, main_device)

    # Create model
    nnue = M.NNUE(args.use_factorizer, feature_set=halfkp.Features()).to(main_device)
    if args.resume_from_model:
        nnue.load_state_dict(torch.load(args.resume_from_model))

    # Configure optimizer
    optimizer = ranger.Ranger(nnue.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True, min_lr=1e-6)

    # Main training loop
    num_batches = len(train_data_loader)
    epoch = 0
    running_train_loss = 0.0
    while True:
        best_val_loss = 1000000.0

        for k, sample in enumerate(train_data_loader):
            train_loss = train_step(nnue, sample, optimizer, args.lambda_, epoch, k, num_batches)
            running_train_loss += train_loss.item()
          
            if k%args.val_check_interval == (args.val_check_interval-1):
                val_loss = calculate_validation_loss(nnue, val_data_loader, args.lambda_)
                new_best = False
                if (val_loss < best_val_loss):
                    new_best = True
                    best_val_loss = val_loss
                save_model(nnue, output_path, epoch, k, val_loss, new_best, False)
                writer.add_scalar('training loss', running_train_loss/args.val_check_interval, epoch*num_batches + k)
                writer.add_scalar('validation loss', val_loss, epoch*num_batches + k)
                running_train_loss = 0.0
    
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
    parser.add_argument('--train-size', type=int, required=True, help='Number of training samples')
    parser.add_argument('--val-size', type=int, required=True, help='Number of Validation samples')
    parser.add_argument('--lambda', default=1.0, type=float, dest='lambda_', help='lambda=1.0 = train on evaluations, lambda=0.0 = train on game results, interpolates between (default=1.0)')
    parser.add_argument('--batch-size', default=8192, type=int, help='Number of positions per batch / per iteration (default=8192)')
    parser.add_argument('--use-factorizer', action='store_true', help='Use factorizer when training')
    parser.add_argument('--val-check-interval', default=2000, type=int, help='How often to check validation loss (default=2000)')
    parser.add_argument('--resume-from-model', help='Initializes training using the weights from the given .pt model')
    args = parser.parse_args()

    main(args)
