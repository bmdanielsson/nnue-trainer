import argparse
import model as M
import nnue_dataset
import torch
import time
import os
import os.path
from datetime import timedelta
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

BIN_SAMPLE_SIZE = 40
OUTPUT_DIR = 'output'


def filter_saved_models(saved_models, top_n):
    # Sore saved models based on loss
    saved_models.sort()

    # Rename the best saved models (strip the .tmp extension)
    for idx in range(top_n):
        old_path = saved_models[idx][1]
        new_path = os.path.splitext(old_path)[0]
        os.rename(old_path, new_path)

    # Remove remaining saved models for this epoch
    for idx in range(top_n, len(saved_models)):
        os.remove(saved_models[idx][1])


def write_model(nnue, path):
    torch.save(nnue.state_dict(), path)


def save_model(nnue, output_path, epoch, idx, val_loss):
    # Construct the full path
    path = f'{output_path}/epoch_{epoch}_iter_{idx+1}_loss_{val_loss:.5f}.pt.tmp'

    # Save the model
    write_model(nnue, path)

    return path


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
  
  
def calculate_validation_loss(nnue, val_data_loader, wdl):
    nnue.eval()
    with torch.no_grad():
        val_loss = []
        for k, sample in enumerate(val_data_loader):
            us, them, white, black, outcome, score = sample
            pred = nnue(us, them, white, black)
            loss = M.loss_function(wdl, pred, sample)
            val_loss.append(loss)
  
        val_loss = torch.mean(torch.tensor(val_loss))
        nnue.train()
    
    return val_loss
  
  
def train_step(nnue, sample, optimizer, wdl, epoch, idx, num_batches):
    us, them, white, black, outcome, score = sample

    pred = nnue(us, them, white, black)
    loss = M.loss_function(wdl, pred, sample)
    loss.backward()
    optimizer.step()
    nnue.zero_grad()
    if (idx+1) < num_batches:
        endstr='\r'
    else:
        endstr=''
    print(f'Epoch {epoch}, {int(((idx+1)/num_batches)*100.0)}% ({idx+1}/{num_batches}) => {loss.item():.5f}', end=endstr)

    return loss
  
  
def create_data_loaders(train_filename, val_filename, epoch_size, val_size,
                        batch_size, main_device):
    train_dataset = nnue_dataset.SparseBatchDataset(train_filename, epoch_size,
            batch_size,
            (epoch_size + batch_size - 1) // batch_size, device=main_device)
    val_dataset = nnue_dataset.SparseBatchDataset(val_filename, val_size,
            batch_size,
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
    
    # Create directories to store data in
    output_path = prepare_output_directory()

    # Print configuration info
    print(f'Device: {main_device}')
    print(f'Training set: {args.train}')
    print(f'Validation set: {args.val}')
    print(f'Batch size: {args.batch_size}')
    print(f'WDL: {args.wdl}')
    print(f'Validation check interval: {args.val_check_interval}')
    if args.log:
        print(f'Logs written to: {output_path}')
    print(f'Data written to: {output_path}')
    print('')

    # Create log writer
    if args.log:
        writer = SummaryWriter(output_path)

    # Create data loaders
    train_size = int(os.path.getsize(args.train)/BIN_SAMPLE_SIZE)
    val_size = int(os.path.getsize(args.val)/BIN_SAMPLE_SIZE)
    train_data_loader, val_data_loader = create_data_loaders(args.train, args.val, train_size, val_size, args.batch_size, main_device)

    # Create model
    nnue = M.NNUE().to(main_device)

    # Configure optimizer
    optimizer = torch.optim.RAdam(nnue.parameters(), lr=1e-3, betas=(.95, 0.999), eps=1e-5, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=1, verbose=True, min_lr=1e-6)

    # Main training loop
    start = time.monotonic()
    num_batches = len(train_data_loader)
    epoch = 0
    running_train_loss = 0.0
    while True:
        best_val_loss = 1000000.0
        saved_models = []

        for k, sample in enumerate(train_data_loader):
            train_loss = train_step(nnue, sample, optimizer, args.wdl, epoch, k, num_batches)
            running_train_loss += train_loss.item()
          
            if k%args.val_check_interval == (args.val_check_interval-1):
                val_loss = calculate_validation_loss(nnue, val_data_loader, args.wdl)
                if (val_loss < best_val_loss):
                    best_val_loss = val_loss
                path = save_model(nnue, output_path, epoch, k, val_loss)
                saved_models.append((val_loss, path))
                if args.log:
                    writer.add_scalar('training loss', running_train_loss/args.val_check_interval, epoch*num_batches + k)
                    writer.add_scalar('validation loss', val_loss, epoch*num_batches + k)
                running_train_loss = 0.0
    
        val_loss = calculate_validation_loss(nnue, val_data_loader, args.wdl)
        new_best = False
        if (val_loss < best_val_loss):
            new_best = True
            best_val_loss = val_loss
        path = save_model(nnue, output_path, epoch, num_batches-1, val_loss)
        saved_models.append((val_loss, path))
        stop = time.monotonic()
        print(f' ({timedelta(seconds=stop-start)})')

        # Only keep the best snapshots (based on validation loss)
        filter_saved_models(saved_models, args.top_n)

        # Update learning rate scheduler
        scheduler.step(best_val_loss)
        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a NNUE network.')
    parser.add_argument('train', help='Training data (.bin)')
    parser.add_argument('val', help='Validation data (.bin)')
    parser.add_argument('--wdl', default=1.0, type=float, help='wdl=0.0 = train on evaluations, wdl=1.0 = train on game results, interpolates between (default=1.0)')
    parser.add_argument('--batch-size', default=16384, type=int, help='Number of positions per batch / per iteration (default=16384)')
    parser.add_argument('--val-check-interval', default=2000, type=int, help='How often to check validation loss (default=2000)')
    parser.add_argument('--log', action='store_true', help='Enable logging during training')
    parser.add_argument('--top-n', default=2, type=int, help='Number of models to save for each epoch (default=2)')
    args = parser.parse_args()

    main(args)
