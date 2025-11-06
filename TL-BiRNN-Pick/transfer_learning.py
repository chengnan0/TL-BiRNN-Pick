"""
Seismic Phase Picking with Transfer Learning for Mining Seismic Data

This module implements a deep learning-based approach for automatic P-wave phase picking
in mining-induced seismicity using transfer learning techniques.

Author: [Your Name]
Institution: [Your Institution]
Date: [Date]
"""

import h5py
import numpy as np
import pandas as pd
import configs
from torch.utils.data import random_split, Dataset, DataLoader
import torch
import random
from inital_model import BRNN, Loss
import matplotlib.pyplot as plt
import os
import time


def save_dataset_splits(train_set, val_set, test_set, save_dir="dataset_splits"):
    """
    Save dataset splits to text files for reproducibility.

    Args:
        train_set: Training dataset identifiers
        val_set: Validation dataset identifiers
        test_set: Test dataset identifiers
        save_dir: Directory to save split files
    """
    os.makedirs(save_dir, exist_ok=True)

    with open(f"{save_dir}/train_split.txt", "w") as f:
        f.write("\n".join(train_set))
    with open(f"{save_dir}/val_split.txt", "w") as f:
        f.write("\n".join(val_set))
    with open(f"{save_dir}/test_split.txt", "w") as f:
        f.write("\n".join(test_set))


def initialize_dataset():
    """
    Initialize and split the seismic dataset.

    Returns:
        train_set: Training dataset identifiers
        val_set: Validation dataset identifiers
        test_set: Test dataset identifiers
    """
    df = pd.read_csv(configs.h5py_csv)
    print(f'Total events in CSV file: {len(df)}')

    ev_list = list(set(df['dataset_name'].to_list()))
    print(f'Number of unique events: {len(ev_list)}')

    # Split dataset with fixed random seed for reproducibility
    train_set, val_set, test_set = random_split(
        ev_list,
        [configs.data_for_train, configs.data_for_val,
         len(ev_list) - configs.data_for_train - configs.data_for_val],
        generator=torch.Generator().manual_seed(configs.random_seed))

    save_dataset_splits(train_set, val_set, test_set)
    return train_set, val_set, test_set


class SeismicPhaseDataset(Dataset):
    """
    Dataset class for seismic phase picking.

    Handles data loading, preprocessing, and augmentation for seismic waveforms.
    """

    def __init__(self, ev_list, sequence_length, n_channels=3, shuffle=True,
                 norm_mode='max', augmentation=False):
        """
        Initialize seismic dataset.

        Args:
            ev_list: List of event identifiers
            sequence_length: Length of seismic sequences
            n_channels: Number of seismic channels
            shuffle: Whether to shuffle data
            norm_mode: Normalization mode ('max' or 'std')
            augmentation: Whether to apply data augmentation
        """
        self.ev_list = ev_list
        self.dtfl = h5py.File(configs.h5py_path, 'r')
        self.sequence_length = sequence_length
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.norm_mode = norm_mode
        self.augmentation = augmentation
        self.indexes = np.arange(len(self.ev_list))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Return dataset length (doubled if augmentation is enabled)."""
        return len(self.ev_list) * 2 if self.augmentation else len(self.ev_list)

    def __getitem__(self, index):
        """
        Get a single data sample.

        Args:
            index: Sample index

        Returns:
            stream: Normalized seismic waveform tensor
            label: Phase picking label tensor
        """
        # Determine if using augmented data
        use_augmentation = self.augmentation and (index >= len(self.ev_list))
        if use_augmentation:
            index = index % len(self.ev_list)  # Map back to original index

        name = self.ev_list[self.indexes[index]]
        dataset = self.dtfl[str(name)]
        data, p_position = self._process_data(dataset)

        # Apply data augmentation if enabled
        if use_augmentation:
            data, p_position = self._shift_and_extract_window(data, p_position)
            data = self._add_noise(data, dataset.attrs['SNR'])
            data = self._scale_amplitude(data, rate=configs.sca_amp_rate,
                                         snr=dataset.attrs['SNR'])

        stream = data.reshape(1, -1)
        label = self._generate_triangular_label(stream[0, :], p_position)
        stream = self._normalize_data(stream, mode=configs.nor_mod)

        return (torch.from_numpy(stream.astype(np.float32)),
                torch.from_numpy(label.astype(np.float32)))

    def _shift_and_extract_window(self, data, p_position):
        """
        Apply random time shift for data augmentation.

        Args:
            data: Seismic waveform data
            p_position: P-wave arrival position

        Returns:
            shifted_data: Time-shifted data
            shifted_p_position: Updated P-wave position
        """
        coda_end = p_position + 800
        org_len = len(data)

        data = np.array(data)
        shifted_data = np.copy(data)

        # Random rotation for time shift
        nrotate = int(np.random.uniform(1, int(org_len - coda_end)))
        shifted_data[:, 0] = (list(data[:, 0])[-nrotate:] +
                              list(data[:, 0])[:-nrotate])

        # Update P-wave position
        if p_position + nrotate >= 0 and p_position + nrotate < org_len:
            shifted_p_position = p_position + nrotate
            return shifted_data, shifted_p_position

        return data, p_position

    def _add_noise(self, data, snr):
        """
        Add Gaussian noise to seismic data.

        Args:
            data: Input seismic data
            snr: Signal-to-noise ratio

        Returns:
            data_noisy: Noisy seismic data
        """
        if snr >= 10.0:
            data_noisy = np.empty(data.shape)
            noise_scale = (np.random.uniform(0.01, 0.15) *
                           np.max(np.abs(data[:, 0])))
            noise_scale = max(noise_scale, 0.0001)

            data_noisy[:, 0] = (data[:, 0] +
                                np.random.normal(0, noise_scale, data.shape[0]))
        else:
            data_noisy = data

        return data_noisy

    def _scale_amplitude(self, data, rate, snr):
        """
        Randomly scale amplitude for data augmentation.

        Args:
            data: Input seismic data
            rate: Probability of applying scaling
            snr: Signal-to-noise ratio

        Returns:
            data_amp: Amplitude-scaled data
        """
        tmp = np.random.uniform(0, 1)
        data_amp = np.empty(data.shape)

        if snr < 10.0:
            if tmp < rate:
                data_amp[:, 0] = data[:, 0] * np.random.uniform(1, 3)
            elif tmp < 2 * rate:
                data_amp[:, 0] = data[:, 0] / np.random.uniform(1, 3)
        else:
            data_amp = data

        return data_amp

    def _normalize_data(self, data, mode='max'):
        """
        Normalize seismic data.

        Args:
            data: Input seismic data
            mode: Normalization mode ('max' or 'std')

        Returns:
            normalized_data: Normalized seismic data
        """
        data = data.astype(np.float32)
        mean_data = np.mean(data, axis=1, keepdims=True)
        data = data - mean_data

        if mode == 'max':
            max_data = np.max(data, axis=1, keepdims=True)
            max_data = np.where(max_data == 0, 1, max_data)
            data /= max_data
        elif mode == 'std':
            std_data = np.std(data, axis=1, keepdims=True)
            std_data = np.where(std_data == 0, 1, std_data)
            data /= std_data

        return data

    def _process_data(self, dataset):
        """
        Process raw dataset to extract relevant segments.

        Args:
            dataset: HDF5 dataset object

        Returns:
            data: Processed seismic data
            p_position: P-wave arrival position
        """
        p_time = dataset.attrs['p_arrival_sample']
        diff = p_time - self.sequence_length + configs.data_cut
        random_start = random.randint(diff, p_time - configs.data_cut)
        end_time = random_start + self.sequence_length

        if diff < 0:
            data = dataset[0:self.sequence_length]
            return data, p_time
        else:
            if end_time > len(dataset):
                data = dataset[random_start:len(dataset)]
                if len(data) < self.sequence_length:
                    padding_length = self.sequence_length - len(data)
                    data = np.concatenate((data, dataset[:padding_length]), axis=0)
                    p_position = p_time - random_start
                return data, p_position
            else:
                data = dataset[random_start:end_time]
                p_position = p_time - random_start
                return data, p_position

    def _generate_triangular_label(self, t, p_index, std=0.1):
        """
        Generate triangular probability labels for phase picking.

        Args:
            t: Time array
            p_index: P-wave arrival index
            std: Standard deviation for label smoothing

        Returns:
            label: Binary label array [noise_prob, p_wave_prob]
        """
        p = np.zeros_like(t)
        seq_len = len(t)
        label = np.zeros((2, seq_len))

        start_idx = np.max([0, p_index - configs.tri_middle])
        end_idx = np.min([seq_len, p_index + configs.tri_middle + 1])
        segment_length = np.abs(end_idx - start_idx)

        p[start_idx:end_idx] = self._triangular_function()[:segment_length]
        label[1, :] = p  # P-wave probability
        label[0, :] = np.clip(1 - label[1, :], 0, 1)  # Noise probability

        return label

    def _triangular_function(self, a=0, b=configs.tri_middle, c=configs.tri_right):
        """
        Generate triangular function for label creation.

        Args:
            a: Start point
            b: Peak point
            c: End point

        Returns:
            y: Triangular function values
        """
        z = np.linspace(a, c, num=2 * (b - a) + 1)
        y = np.zeros(z.shape)

        y[z <= a] = 0
        y[z >= c] = 0

        # Increasing segment
        first_half = np.logical_and(a < z, z <= b)
        y[first_half] = (z[first_half] - a) / (b - a)

        # Decreasing segment
        second_half = np.logical_and(b < z, z < c)
        y[second_half] = (c - z[second_half]) / (c - b)

        return y


def create_transfer_learning_model():
    """
    Create model with transfer learning from pre-trained weights.

    Returns:
        model: Initialized model with transfer learning setup
    """
    model = BRNN()

    # Load pre-trained weights
    pretrained = torch.jit.load('./china.rnn.single.jit').state_dict()

    # Load compatible parameters
    model_dict = model.state_dict()
    pretrained = {k: v for k, v in pretrained.items()
                  if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)

    # Freeze encoder and first RNN layer
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.rnns[0].parameters():
        param.requires_grad = False

    # Unfreeze second RNN layer and decoder
    for param in model.rnns[1].parameters():
        param.requires_grad = True
    for param in model.decoder.parameters():
        param.requires_grad = True

    # Unfreeze BatchNorm layers
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            for param in module.parameters():
                param.requires_grad = True

    # Initialize final layer
    torch.nn.init.kaiming_normal_(model.decoder.layers[-1].weight,
                                  mode='fan_out', nonlinearity='relu')
    torch.nn.init.zeros_(model.decoder.layers[-1].bias)

    return model.to(device)


def train_model():
    """Main training function for seismic phase picking model."""
    # Initialize data and model
    train_set, val_set, test_set = initialize_dataset()
    train_dataset = SeismicPhaseDataset(train_set, dim=configs.data_length,
                                        augmentation=True)
    val_dataset = SeismicPhaseDataset(val_set, dim=configs.data_length)

    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=configs.batch_size,
                            num_workers=4)

    model = create_transfer_learning_model()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=configs.learning_rate, weight_decay=configs.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=configs.scheduler_t_max,
        eta_min=configs.scheduler_eta_min)

    criterion = Loss()
    best_val_loss = float('inf')
    start_time = time.time()

    print("Starting model training...")
    for epoch in range(configs.num_epochs):
        # Training phase
        model.train()
        total_grad_norm = 0.0
        train_loss = 0.0

        for batch_idx, (streams, labels) in enumerate(train_loader):
            streams, labels = streams.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(streams)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            max_norm = max(1.0, 10 + 5 * (1 - epoch / configs.num_epochs))
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=max_norm, norm_type=2)
            total_grad_norm += grad_norm.item()

            optimizer.step()
            train_loss += loss.item() * streams.size(0)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for streams_v, labels_v in val_loader:
                streams_v, labels_v = streams_v.to(device), labels_v.to(device)
                outputs_v = model(streams_v)
                val_loss += criterion(outputs_v, labels_v).item() * streams_v.size(0)

        # Calculate metrics
        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)
        avg_grad_norm = total_grad_norm / len(train_loader)
        scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.eval()
            scripted_model = torch.jit.script(model)
            torch.jit.save(scripted_model, configs.model_save_path)
            print(f"Best model saved at epoch {epoch + 1} "
                  f"(Validation Loss: {val_loss:.4f})")

        # Log training progress
        elapsed_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        print(f'Epoch {epoch + 1}/{configs.num_epochs} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Learning Rate: {current_lr:.2e} | '
              f'Gradient Norm: {avg_grad_norm:.4f} | '
              f'Elapsed Time: {elapsed_time:.2f}s')


def evaluate_model(model_path, test_dataset, output_path, device, phase_model=True):
    """
    Evaluate model performance on test dataset.

    Args:
        model_path: Path to trained model
        test_dataset: Test dataset
        output_path: Output file path for results
        device: Computation device
        phase_model: Whether using phase detection model
    """
    from inital_model import BRNN

    with open(output_path, "w", encoding="utf-8") as outfile:
        if phase_model:
            model = torch.load(model_path, map_location=device)

            for index, data in enumerate(test_dataset):
                input_data, label = data
                label = label.numpy()
                true_p_position = np.argmax(label[1, :])

                input_data = input_data.unsqueeze(0).to(device)

                with torch.no_grad():
                    predictions = model(input_data)
                    predictions = predictions.detach().cpu().numpy()

                # Detect phase arrivals
                detected_phases = find_phase_point2point(predictions,
                                                         height=0.25, dist=10)

                outfile.write(f"#true_phase_position,{true_p_position}\n")

                if len(detected_phases[0]) > 0:
                    phase_info = detected_phases[0][0]
                    phase_type = phase_info[0]
                    detected_position = phase_info[1]
                    confidence = phase_info[2]
                    outfile.write(f"{phase_type},{detected_position},{confidence}\n")

        else:
            original_model = torch.jit.load(model_path)

            for index, data in enumerate(test_dataset):
                input_data, label = data
                label = label.numpy()
                true_p_position = np.argmax(label[1, :])

                outfile.write(f"#true_phase_position,{true_p_position}\n")
                input_data = input_data.transpose(0, 1).squeeze()

                with torch.no_grad():
                    predictions = original_model(input_data)
                    predictions = predictions.cpu().numpy()

                if len(predictions) > 0 and predictions[0, 0] == 0:
                    phase_type = int(predictions[0, 0]) + 1
                    detected_position = int(predictions[0, 1])
                    outfile.write(f"{phase_type},{detected_position},{predictions[0, 2]}\n")

                outfile.flush()


def load_dataset_splits(save_dir="dataset_splits"):
    """
    Load pre-saved dataset splits.

    Args:
        save_dir: Directory containing split files

    Returns:
        train_set, val_set, test_set: Dataset splits
    """
    with open(f"{save_dir}/train_split.txt", "r") as f:
        train_set = f.read().splitlines()
    with open(f"{save_dir}/val_split.txt", "r") as f:
        val_set = f.read().splitlines()
    with open(f"{save_dir}/test_split.txt", "r") as f:
        test_set = f.read().splitlines()
    return train_set, val_set, test_set


if __name__ == '__main__':

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.enabled = False
    # Load dataset splits
    train_set, val_set, test_set = load_dataset_splits()
    test_dataset = SeismicPhaseDataset(test_set, dim=configs.data_length)
    # Model evaluation
    model_path = configs.trained_model_path
    output_path = os.path.splitext(model_path)[0] + '_predictions.txt'

    evaluate_model(model_path, test_dataset, output_path, device, phase_model=True)