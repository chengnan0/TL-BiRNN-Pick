import queue
import datetime
import h5py
import numpy as np
import time
import multiprocessing
import threading
import matplotlib.pyplot as plt

plt.switch_backend("agg")
import os
import random
from google.colab import drive
from inital_model import BRNN, Loss
import torch


def _label(a=0, b=20, c=40):
    """
    Generate triangular labeling function for phase detection.

    Args:
        a: Start point
        b: Peak point
        c: End point

    Returns:
        y: Triangular label array
    """
    z = np.linspace(a, c, num=2 * (b - a) + 1)
    y = np.zeros(z.shape)
    y[z <= a] = 0
    y[z >= c] = 0
    first_half = np.logical_and(a < z, z <= b)
    y[first_half] = (z[first_half] - a) / (b - a)
    second_half = np.logical_and(b < z, z < c)
    y[second_half] = (c - z[second_half]) / (c - b)
    return y


class MiningSeismicDataLoader_P():
    """
    Data loader for mining seismic phase picking with P-wave focus.
    """

    def __init__(self, file_name="h5data/diting/DiTing.v2.0.h5", n_length=10240, stride=16, padlen=256):
        self.file_name = file_name
        self.length = n_length
        self.stride = stride
        self.padlen = padlen
        self.n_thread = 50
        self.phase_dict = {
            "Pg": 0,
            "P": 0,  # Focus only on P-wave phases
        }
        fqueue = queue.Queue(10)
        self.dqueue = queue.Queue(10)
        self.epoch = 0
        self.file_name = file_name

        # Start data feeding thread
        self.p1 = threading.Thread(target=self.feed_data, args=(fqueue, self.epoch))
        self.p1.daemon = True
        self.p1.start()

        # Start data processing threads
        self.p2s = []
        for _ in range(self.n_thread):
            p = threading.Thread(target=self.process, args=(fqueue, self.dqueue))
            p.daemon = True
            p.start()
            self.p2s.append(p)

    def get_epoch(self):
        """Get current epoch count."""
        return self.epoch

    def feed_data(self, fqueue, epoch):
        """
        Feed raw seismic data from HDF5 file to queue.

        Args:
            fqueue: Input queue for raw data
            epoch: Current epoch count
        """
        while True:
            try:
                with h5py.File(self.file_name, "r") as h5file:
                    train = h5file["train"]
                    for ekey in train:
                        event = train[ekey]
                        for skey in event:
                            station = event[skey]
                            data = station[:, 2]  # Use specific channel
                            data = data.reshape(-1, 1)

                            pt = station.attrs["Pg"]  # Get P-wave arrival time
                            fqueue.put([data, [int(pt), -1]])  # -1 indicates ignore S-wave
            except Exception as e:
                print(f"Error reading HDF5 file: {e}")
                time.sleep(1)  # Wait before retry
            self.epoch += 1

    def process(self, fqueue, dqueue):
        """
        Process raw seismic data into training samples.

        Args:
            fqueue: Input queue with raw data
            dqueue: Output queue with processed data
        """
        count = 0
        llen = self.length // self.stride

        while True:
            try:
                data, pidx = fqueue.get()
                pdic = {"P": pidx[0]}  # Process only P-wave

                # Randomly select window position for data augmentation
                bidx = np.random.choice([pidx[0]]) - np.random.randint(self.padlen, self.length - self.padlen)
                eidx = bidx + self.length
                rdata = np.zeros([self.length, 3])
                len_data = len(data)

                # Handle different window positioning cases
                if bidx >= 0 and eidx < len_data:
                    rdata = data[bidx:eidx, :]
                elif bidx < 0 and eidx < len_data:
                    before = -bidx
                    rdata = np.pad(data[:eidx], ((before, 0), (0, 0)))
                elif bidx > 0 and eidx >= len_data:
                    after = eidx - len_data
                    rdata = np.pad(data[bidx:], ((0, after), (0, 0)))
                elif bidx < 0 and eidx >= len_data:
                    after = eidx - len_data
                    before = -bidx
                    rdata = np.pad(data, ((before, after), (0, 0)))

                # Normalize data
                rdata = rdata.astype(np.float32)
                rdata -= np.mean(rdata, axis=0, keepdims=True)
                rdata /= (np.max(np.abs(rdata)) + 1e-6)
                rdata *= np.random.uniform(0.5, 2)  # Amplitude augmentation
                rdata = rdata.T

                if rdata.shape[1] != self.length:
                    continue

                # Generate LPPN labels (P-wave detection)
                label1 = np.zeros([1, 2, llen])
                for pkey in pdic:
                    pid = self.phase_dict[pkey]
                    idx = (pdic[pkey] - bidx) // self.stride
                    if idx - 1 > 0:
                        label1[0, :, idx - 1:idx + 2] = -1
                    if idx > 0 and idx < llen:
                        label1[0, 0, idx] = pid + 1
                        label1[0, 1, idx] = (pdic[pkey] - bidx) % self.stride

                def tri(t, mu, std=0.1):
                    """Generate triangular probability distribution."""
                    midx = int(mu * 100)
                    p = np.zeros_like(t)
                    bidx = np.max([0, midx - 20])
                    eidx = np.min([self.length, midx + 21])
                    lent = np.abs(eidx - bidx)
                    p[bidx:eidx] = _label()[:lent]
                    return p

                t = np.arange(self.length) * 0.01

                # Generate probability labels for noise and P-wave
                label2 = np.zeros([1, 2, self.length])  # Noise and P-wave
                label3 = np.zeros([1, 2, self.length])  # Noise and P-wave

                for pkey in pdic:
                    pid = self.phase_dict[pkey]
                    idx = (pdic[pkey] - bidx)
                    if idx > 0 and idx < self.length:
                        label2[0, 1, :] = tri(t, idx * 0.01, 0.1)  # P-wave probability
                        label3[0, 1, :] = tri(t, idx * 0.01, 0.1)  # P-wave probability

                # Noise probabilities
                label2[0, 0, :] = np.clip(1 - label2[0, 1, :], 0, 1)  # Noise
                label3[0, 0, :] = 1 - label3[0, 1, :]  # Noise

                dqueue.put([rdata.astype(np.float32), label1, label2, label3])
                count += 1

            except Exception as e:
                print(f"Error processing data: {e}")
                continue

    def batch_data(self, batch_size=32):
        """
        Get batch of processed data.

        Args:
            batch_size: Number of samples per batch

        Returns:
            Tuple of (waveform_data, label1, label2, label3)
        """
        x1, x2, x3, x4 = [], [], [], []
        for _ in range(batch_size):
            data, label1, label2, label3 = self.dqueue.get()
            x1.append(data)
            x2.append(label1)
            x3.append(label2)
            x4.append(label3)
        x1 = np.stack(x1, axis=0)
        x2 = np.concatenate(x2, axis=0)
        x3 = np.concatenate(x3, axis=0)
        x4 = np.concatenate(x4, axis=0)
        return x1, x2, x3, x4


def main():
    """Main training function."""
    # Mount Google Drive (for Colab)
    drive.mount('/content/drive')

    # Set file paths
    file_name = r'/content/drive/MyDrive/Diting2/natural_v2.hdf5'
    model_save_path = r'/content/drive/MyDrive/Diting2/mining_seismic_phase_picker.pt'

    # Initialize model and loss
    model = BRNN()
    lossfn = Loss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loader configuration
    stride = 1
    data_loader = MiningSeismicDataLoader_P(
        file_name=file_name,
        stride=stride,
        n_length=6144,
        padlen=512
    )

    # Model setup
    model.to(device)
    model.train()
    lossfn.to(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=0e-3)

    # Training loop
    acc_time = 0
    for step in range(500000):
        # Get batch data
        wave_data, _, prob_labels, _ = data_loader.batch_data()

        # Convert to tensors
        wave_tensor = torch.tensor(wave_data, dtype=torch.float32).to(device)
        label_tensor = torch.tensor(prob_labels, dtype=torch.float32).to(device)

        # Forward pass
        output = model(wave_tensor)
        loss = lossfn(output, label_tensor)

        # Backward pass
        loss.backward()

        # Check for NaN values
        if torch.isnan(loss):
            print("NaN loss detected, skipping update")
            optimizer.zero_grad()
            continue

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Logging
        loss_value = loss.detach().cpu().numpy()

        if step % 100 == 0:
            print(f"Step: {step:8}, Loss: {loss_value:6.3f}")
            # Save model checkpoint
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")


if __name__ == '__main__':
    main()