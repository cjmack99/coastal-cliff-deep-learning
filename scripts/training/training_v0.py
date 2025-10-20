#!/usr/bin/env python3
"""
Cliff Failure Prediction using U-Net ConvLSTM with Parallel Processing
Usage: python cliff_failure_prediction.py --location Delmar --epochs 50 --batch_size 4
"""

import argparse
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless systems
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import psutil
import gc
from datetime import datetime
import multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class CliffFailureDataset(Dataset):
    """Dataset for cliff failure prediction using temporal sequences"""
    
    def __init__(self, erosion_cube, cluster_cube, sequence_length=5, prediction_horizon=3):
        """
        Args:
            erosion_cube: (time, height, width) erosion rate data
            cluster_cube: (time, height, width) cluster ID data  
            sequence_length: number of time steps to use as input
            prediction_horizon: how many steps ahead to predict failure
        """
        self.erosion_cube = erosion_cube
        self.cluster_cube = cluster_cube
        self.seq_len = sequence_length
        self.pred_horizon = prediction_horizon
        
        # Create failure labels by detecting large erosion events
        self.failure_labels = self._create_failure_labels()
        
        # Generate valid indices for sequences
        self.valid_indices = list(range(
            sequence_length, 
            len(erosion_cube) - prediction_horizon
        ))
    
    def _create_failure_labels(self):
        """Create binary failure labels based on erosion magnitude"""
        # Define failure as erosion > 95th percentile in any location
        threshold = np.nanpercentile(self.erosion_cube, 95)
        failures = (self.erosion_cube > threshold).astype(float)
        return failures
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """Get sequence of inputs and future failure label"""
        time_idx = self.valid_indices[idx]
        
        # Input sequence: erosion + cluster data
        erosion_seq = self.erosion_cube[time_idx-self.seq_len:time_idx]
        cluster_seq = self.cluster_cube[time_idx-self.seq_len:time_idx]
        
        # Stack as 2-channel input (erosion, clusters)
        input_seq = np.stack([erosion_seq, cluster_seq], axis=1)  # (seq_len, 2, H, W)
        
        # Target: failure probability map at future time
        target = self.failure_labels[time_idx + self.pred_horizon]
        
        return torch.FloatTensor(input_seq), torch.FloatTensor(target)

class ConvLSTMCell(nn.Module):
    """ConvLSTM cell for processing spatial-temporal sequences"""
    
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
    
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # Concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        # Split into gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class UNetConvLSTM(nn.Module):
    """U-Net with ConvLSTM for cliff failure prediction"""
    
    def __init__(self, input_channels=2, hidden_dims=[32, 64], kernel_size=3):
        super().__init__()
        self.hidden_dims = hidden_dims
        
        # Encoder ConvLSTM layers
        self.encoder_convlstms = nn.ModuleList()
        self.encoder_convlstms.append(
            ConvLSTMCell(input_channels, hidden_dims[0], kernel_size)
        )
        for i in range(1, len(hidden_dims)):
            self.encoder_convlstms.append(
                ConvLSTMCell(hidden_dims[i-1], hidden_dims[i], kernel_size)
            )
        
        # Decoder layers
        self.decoder_convs = nn.ModuleList()
        for i in range(len(hidden_dims)-1, 0, -1):
            self.decoder_convs.append(
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i-1], 4, 2, 1)
            )
        
        # Final output layer
        self.final_conv = nn.Conv2d(hidden_dims[0], 1, 1)
        
        # Pooling for encoder
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, channels, height, width)
        Returns:
            failure_prob: (batch, 1, height, width)
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # Initialize hidden states
        encoder_states = []
        for i, hidden_dim in enumerate(self.hidden_dims):
            h_size = (height // (2**i), width // (2**i))
            h = torch.zeros(batch_size, hidden_dim, *h_size).to(x.device)
            c = torch.zeros(batch_size, hidden_dim, *h_size).to(x.device)
            encoder_states.append((h, c))
        
        # Process sequence through encoder
        skip_connections = []
        
        for t in range(seq_len):
            layer_input = x[:, t]  # (batch, channels, H, W)
            layer_outputs = []
            
            for i, convlstm in enumerate(self.encoder_convlstms):
                if i > 0:
                    layer_input = self.pool(layer_outputs[i-1])
                
                h, c = convlstm(layer_input, encoder_states[i])
                encoder_states[i] = (h, c)
                layer_outputs.append(h)
            
            # Store skip connections from last timestep
            if t == seq_len - 1:
                skip_connections = layer_outputs[:-1]  # Exclude bottleneck
        
        # Decoder with skip connections
        x = layer_outputs[-1]  # Bottleneck features
        
        for i, decoder_conv in enumerate(self.decoder_convs):
            x = F.relu(decoder_conv(x))
            if i < len(skip_connections):
                # Add skip connection
                skip_idx = len(skip_connections) - 1 - i
                x = x + skip_connections[skip_idx]
        
        # Final prediction
        failure_prob = torch.sigmoid(self.final_conv(x))
        
        return failure_prob

def create_data_loaders_temporal(data_path, batch_size=4, train_ratio=0.6, val_ratio=0.2, 
                                downsample_factor=4, max_time_steps=None, num_workers=None, 
                                distributed=False, world_size=1, rank=0):
    """Create train/val/test data loaders with TEMPORAL splitting, memory optimization, and parallel processing"""
    
    # Calculate number of workers if not specified
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() // 4)
    
    print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print(f"Using {num_workers} worker processes for data loading")
    
    # Load data cubes
    erosion_data = np.load(f"{data_path}/cube_ero_10cm_filled.npz")['data']
    cluster_data = np.load(f"{data_path}/cube_clusters_ero_10cm_filled.npz")['data']
    
    print(f"Original data shapes: erosion {erosion_data.shape}, cluster {cluster_data.shape}")
    print(f"Original memory usage: {(erosion_data.nbytes + cluster_data.nbytes) / (1024**3):.2f} GB")
    
    # Limit time steps if specified
    if max_time_steps:
        erosion_data = erosion_data[:max_time_steps]
        cluster_data = cluster_data[:max_time_steps]
        print(f"Limited to {max_time_steps} time steps")
    
    # Downsample spatially to reduce memory
    if downsample_factor > 1:
        print(f"Downsampling by factor of {downsample_factor}...")
        erosion_data = erosion_data[:, ::downsample_factor, ::downsample_factor]
        cluster_data = cluster_data[:, ::downsample_factor, ::downsample_factor]
        print(f"New shapes: erosion {erosion_data.shape}, cluster {cluster_data.shape}")
    
    # Handle NaN values
    erosion_data = np.nan_to_num(erosion_data, nan=0.0)
    cluster_data = np.nan_to_num(cluster_data, nan=0.0)
    
    # Normalize erosion data
    print("Normalizing data...")
    scaler = StandardScaler()
    original_shape = erosion_data.shape
    erosion_flat = erosion_data.reshape(-1, 1)
    erosion_normalized = scaler.fit_transform(erosion_flat).reshape(original_shape)
    
    # Create dataset with shorter sequences
    print("Creating dataset...")
    dataset = CliffFailureDataset(erosion_normalized, cluster_data, 
                                sequence_length=3, prediction_horizon=1)
    
    # TEMPORAL SPLIT - chronological order maintained
    total_samples = len(dataset)
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))
    
    train_indices = list(range(0, train_end))
    val_indices = list(range(train_end, val_end))
    test_indices = list(range(val_end, total_samples))
    
    if rank == 0:  # Only print from main process
        print(f"Temporal split:")
        print(f"  Train: samples 0-{train_end-1} ({len(train_indices)} samples)")
        print(f"  Val:   samples {train_end}-{val_end-1} ({len(val_indices)} samples)")
        print(f"  Test:  samples {val_end}-{total_samples-1} ({len(test_indices)} samples)")
    
    # Create datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # Create samplers for distributed training
    train_sampler = val_sampler = test_sampler = None
    shuffle = True
    
    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        shuffle = False  # Sampler handles shuffling
    
    # Create data loaders with parallel processing
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else 2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else 2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else 2
    )
    
    # Force garbage collection
    del erosion_data, cluster_data, erosion_normalized, dataset
    gc.collect()
    
    return train_loader, val_loader, test_loader, scaler

def train_model_distributed(model, train_loader, val_loader, num_epochs=50, lr=0.001, 
                          save_path=None, rank=0, world_size=1, distributed=False):
    """Train the U-Net ConvLSTM model with distributed training support"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if rank == 0:
        print(f"Training on device: {device}")
        if distributed:
            print(f"Distributed training with {world_size} processes")
    
    model = model.to(device)
    
    # Wrap model for distributed training
    if distributed and torch.cuda.is_available():
        model = DDP(model, device_ids=[rank])
    elif distributed:
        model = DDP(model)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=(rank==0))
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Set epoch for distributed sampler
        if distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            output = model(data)
            
            # Reshape for loss computation
            output = output.squeeze(1)  # Remove channel dim
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            # Clear cache periodically
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                output = model(data).squeeze(1)
                val_loss += criterion(output, target).item()
                val_batches += 1
        
        # Average losses across all processes if distributed
        if distributed:
            train_loss_tensor = torch.tensor(train_loss).to(device)
            val_loss_tensor = torch.tensor(val_loss).to(device)
            num_batches_tensor = torch.tensor(num_batches).to(device)
            val_batches_tensor = torch.tensor(val_batches).to(device)
            
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_batches_tensor, op=dist.ReduceOp.SUM)
            
            avg_train_loss = train_loss_tensor.item() / num_batches_tensor.item()
            avg_val_loss = val_loss_tensor.item() / val_batches_tensor.item()
        else:
            avg_train_loss = train_loss / num_batches
            avg_val_loss = val_loss / val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        # Save best model (only from main process)
        if avg_val_loss < best_val_loss and save_path and rank == 0:
            best_val_loss = avg_val_loss
            model_to_save = model.module if distributed else model
            torch.save(model_to_save.state_dict(), save_path)
            print(f'Saved best model at epoch {epoch} with val loss: {avg_val_loss:.4f}')
        
        if epoch % 5 == 0 and rank == 0:
            print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            print(f'Memory usage: {psutil.virtual_memory().percent:.1f}%')
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, device='cpu', distributed=False, rank=0):
    """Evaluate model on test set"""
    model.eval()
    model = model.to(device)
    
    criterion = nn.BCELoss()
    test_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data).squeeze(1)
            test_loss += criterion(output, target).item()
            num_batches += 1
    
    # Average across processes if distributed
    if distributed:
        test_loss_tensor = torch.tensor(test_loss).to(device)
        num_batches_tensor = torch.tensor(num_batches).to(device)
        
        dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
        
        avg_test_loss = test_loss_tensor.item() / num_batches_tensor.item()
    else:
        avg_test_loss = test_loss / num_batches
    
    if rank == 0:
        print(f'Final Test Loss: {avg_test_loss:.4f}')
    
    return avg_test_loss

def setup_distributed(rank, world_size):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", 
                          rank=rank, world_size=world_size)

def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()

def train_worker(rank, world_size, args):
    """Worker function for distributed training"""
    
    # Setup distributed training
    setup_distributed(rank, world_size)
    
    # Set device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
    else:
        device = torch.device('cpu')
    
    try:
        # Create data loaders
        if rank == 0:
            print("\nCreating data loaders...")
        
        train_loader, val_loader, test_loader, scaler = create_data_loaders_temporal(
            args.data_path, 
            batch_size=args.batch_size,
            downsample_factor=args.downsample,
            max_time_steps=args.max_time_steps,
            distributed=True,
            world_size=world_size,
            rank=rank
        )
        
        # Initialize model
        if rank == 0:
            print("\nInitializing model...")
        model = UNetConvLSTM(input_channels=2, hidden_dims=[16, 32])
        
        # Train model
        if rank == 0:
            print("\nStarting distributed training...")
        
        model_path = os.path.join(args.output_dir, f'{args.location}_best_model.pth')
        train_losses, val_losses = train_model_distributed(
            model, train_loader, val_loader, 
            num_epochs=args.epochs, 
            lr=args.lr,
            save_path=model_path,
            rank=rank,
            world_size=world_size,
            distributed=True
        )
        
        # Only main process handles plotting and evaluation
        if rank == 0:
            # Plot training curves
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title(f'Training Progress - {args.location} (Distributed)')
            plt.grid(True)
            plot_path = os.path.join(args.output_dir, f'{args.location}_training_curves.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to: {plot_path}")
            
            # Load best model and evaluate
            print("\nEvaluating best model...")
            model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Evaluate on test set
        test_loss = evaluate_model(model, test_loader, device, distributed=True, rank=rank)
        
        # Save results (only from main process)
        if rank == 0:
            results = {
                'location': args.location,
                'final_test_loss': test_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'config': vars(args),
                'distributed': True,
                'world_size': world_size,
                'timestamp': datetime.now().isoformat()
            }
            
            results_path = os.path.join(args.output_dir, f'{args.location}_results_distributed.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nDistributed training completed!")
            print(f"Best model saved to: {model_path}")
            print(f"Results saved to: {results_path}")
            print(f"Final test loss: {test_loss:.4f}")
    
    finally:
        cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description='Cliff Failure Prediction using U-Net ConvLSTM with Parallel Processing')
    parser.add_argument('--location', type=str, default='Delmar', 
                       help='Location name (default: Delmar)')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to data cubes (default: auto-generated from location)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size (default: 2)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--downsample', type=int, default=4,
                       help='Spatial downsampling factor (default: 4)')
    parser.add_argument('--max_time_steps', type=int, default=100,
                       help='Maximum time steps to use (default: 100)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory (default: ./outputs)')
    parser.add_argument('--distributed', action='store_true',
                       help='Use distributed training across multiple processes')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of data loader workers (default: cpu_count//4)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set data path
    if args.data_path is None:
        args.data_path = f"/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs/results/{args.location}/data_cubes"
    
    # Calculate number of workers
    if args.num_workers is None:
        args.num_workers = max(1, mp.cpu_count() // 4)
    
    print(f"Starting cliff failure analysis for {args.location}")
    print(f"Configuration:")
    print(f"  Data path: {args.data_path}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Downsample factor: {args.downsample}")
    print(f"  Max time steps: {args.max_time_steps}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  CPU cores: {mp.cpu_count()}")
    print(f"  Data loader workers: {args.num_workers}")
    print(f"  Distributed training: {args.distributed}")
    
    if args.distributed:
        # Distributed training
        world_size = args.num_workers
        print(f"Spawning {world_size} processes for distributed training...")
        mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        # Single process training with parallel data loading
        print("\nCreating data loaders with parallel processing...")
        train_loader, val_loader, test_loader, scaler = create_data_loaders_temporal(
            args.data_path, 
            batch_size=args.batch_size,
            downsample_factor=args.downsample,
            max_time_steps=args.max_time_steps,
            num_workers=args.num_workers
        )
        
        # Initialize model
        print("\nInitializing model...")
        model = UNetConvLSTM(input_channels=2, hidden_dims=[16, 32])
        
        # Train model
        print("\nStarting training with parallel data loading...")
        model_path = os.path.join(args.output_dir, f'{args.location}_best_model.pth')
        train_losses, val_losses = train_model_distributed(
            model, train_loader, val_loader, 
            num_epochs=args.epochs, 
            lr=args.lr,
            save_path=model_path,
            rank=0,
            world_size=1,
            distributed=False
        )
        
        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Training Progress - {args.location}')
        plt.grid(True)
        plot_path = os.path.join(args.output_dir, f'{args.location}_training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {plot_path}")
        
        # Load best model and evaluate
        print("\nEvaluating best model...")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_loss = evaluate_model(model, test_loader, device)
        
        # Save results
        results = {
            'location': args.location,
            'final_test_loss': test_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'config': vars(args),
            'distributed': False,
            'num_workers': args.num_workers,
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = os.path.join(args.output_dir, f'{args.location}_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nTraining completed!")
        print(f"Best model saved to: {model_path}")
        print(f"Results saved to: {results_path}")
        print(f"Final test loss: {test_loss:.4f}")

if __name__ == '__main__':
    main()