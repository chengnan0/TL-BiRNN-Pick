### Waveform Plot
import torch
import os
import h5py
import configs
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams['font.family'] = 'Times New Roman'

# Set random seed
random.seed(42)  # Set random seed for reproducibility


def process_data(dataset):
    """
    Process seismic waveform data by extracting relevant segments.

    Parameters:
    - dataset: input seismic data

    Returns:
    - processed_data: extracted data segment
    - p_start: P-wave arrival position in the segment
    """
    p_time = dataset.attrs['p_arrival_sample']
    diff = p_time - configs.data_length + configs.data_cut
    random_start = random.randint(diff, p_time - configs.data_cut)
    end_time = random_start + configs.data_length
    dataset = dataset - np.median(dataset)

    if diff < 0:
        data = dataset[0:configs.data_length]
        return data, p_time
    else:
        if end_time > len(dataset):
            data = dataset[random_start:len(dataset)]
            if len(data) < configs.data_length:
                padding_length = configs.data_length - len(data)
                data1 = np.concatenate((data, dataset[:padding_length]), axis=0)
                p_start = p_time - random_start
            return data1, p_start
        else:
            data1 = dataset[random_start:end_time]
            p_start = p_time - random_start
            return data1, p_start


def _normalize(data):
    """Normalize data by subtracting mean and scaling by maximum value."""
    data = data.astype(np.float32)  # Ensure data is float type

    # Calculate overall mean and maximum
    mean_data = np.mean(data)  # Calculate mean of all data
    data = data - mean_data  # Subtract mean

    max_data = np.max(data)  # Calculate maximum value
    max_data = np.where(max_data == 0, 1, max_data)  # Avoid division by zero
    data /= max_data  # Normalize

    return data


def plot_waveform_and_probability(data, pro, pick, snr):
    """
    Plot waveform data and probability distribution in two rows.

    Parameters:
    - data: waveform data (numpy array)
    - pro: probability data (numpy array)
    - pick: manual picking time
    - snr: signal-to-noise ratio
    """
    # Create subplots with 2 rows and 1 column
    fig, axs = plt.subplots(2, 1, figsize=(6, 4), sharex=True, gridspec_kw={'hspace': 0.0})
    max_index = np.argmax(pro)
    pro[max_index] += 0.0

    # First row: plot waveform data
    axs[0].plot(data, label='Waveform', color='black')
    axs[0].set_ylabel('Amplitude (m/s)', color='black')
    axs[0].axvline(x=max_index, color='red', linestyle='-', label='Phase-P')  # Phase-P marker
    axs[0].legend(loc="upper left")
    axs[0].tick_params(axis='y', labelcolor='black')

    # Add SNR annotation to top right corner
    axs[0].text(0.98, 0.95, f'SNR={snr:.2f} dB', transform=axs[0].transAxes,
                fontsize=12, color='black', ha='right', va='top')

    # Second row: plot probability distribution
    axs[1].plot(pro, label='Probability', color='red', linestyle='--')
    axs[1].set_xlabel('Sample')
    axs[1].set_ylabel('Probability')
    axs[1].legend(loc="upper left")
    axs[1].tick_params(axis='y')

    # Define custom formatter for probability axis
    def custom_formatter(y, pos):
        if np.isclose(y, 0.0):
            return "0"
        else:
            return f"{y + 0.3:.2f}"

    # Apply formatter to axs[1] y-axis
    axs[1].yaxis.set_major_formatter(FuncFormatter(custom_formatter))

    print('Picking error: {:.2f} ms'.format((pick - max_index) / 500 * 1000))

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Display figure
    plt.show()


def calculate_snr(data, p_position, window_size=100):
    """
    Calculate Signal-to-Noise Ratio (SNR) of seismic data.

    Parameters:
    - data: seismic waveform data
    - p_position: P-wave arrival position index
    - window_size: analysis window size

    Returns:
    - snr: calculated SNR value in dB
    """
    # Extract P-wave signal
    p_wave_signal = data[p_position - window_size: p_position + window_size]

    # Calculate signal energy
    E_signal = np.sum(p_wave_signal ** 2)

    # Extract noise signal (before P-wave arrival)
    noise_signal = data[:p_position - window_size]
    E_noise = np.sum(noise_signal ** 2)

    # Calculate SNR in dB
    snr = 10 * np.log10(E_signal / E_noise)
    return snr


def adjust_snr_to_target(data, p_position, target_snr_range=(-5, 0), window_size=100):
    """
    Adjust SNR of seismic data to target range by adding noise.

    Parameters:
    - data: original seismic data
    - p_position: P-wave arrival position
    - target_snr_range: target SNR range in dB
    - window_size: analysis window size

    Returns:
    - noisy_signal: data with adjusted SNR
    - target_snr: actual SNR value after adjustment
    """
    # Calculate current signal energy
    p_wave_signal = data[p_position - window_size: p_position + window_size]
    E_signal = np.sum(p_wave_signal ** 2)

    # Randomly select target SNR
    target_snr = np.random.uniform(target_snr_range[0], target_snr_range[1])

    # Calculate target noise energy
    E_noise_target = E_signal / (10 ** (target_snr / 10))

    # Generate noise signal with same shape as data
    noise = np.random.randn(*data.shape)
    noise_energy = np.sum(noise ** 2)

    # Adjust noise amplitude to match target energy
    scaling_factor = np.sqrt(E_noise_target / noise_energy)
    noise = noise * scaling_factor

    # Add noise to original signal
    noisy_signal = data + noise

    return noisy_signal, target_snr


def run_model(data2, T_model, device='cpu'):
    """
    Run model inference on input data.

    Parameters:
    - data2: input data (numpy array)
    - T_model: loaded PyTorch model
    - device: computation device ('cpu' or 'cuda')

    Returns:
    - pro: probability output [0, 1, :] (numpy array)
    """
    # Normalize data
    data = _normalize(data2)

    # Prepare input tensor for model
    stream = np.stack([data], axis=0).squeeze(axis=-1)
    stream = torch.from_numpy(stream.astype(np.float32))
    input_tensor = stream.unsqueeze(0).to(device)  # Add batch dimension

    # Model inference
    with torch.no_grad():
        output = T_model(input_tensor)

    # Extract probability output
    output_np = output.detach().cpu().numpy()
    pro = output_np[0, 1, :]

    return pro


def plot_four_waveforms_and_probabilities(data_list, pro_list, pick_list, snr_list):
    """
    Plot four waveform and probability pairs in 4x2 subplot layout.

    Parameters:
    - data_list: list of 4 waveform data arrays
    - pro_list: list of 4 probability data arrays
    - pick_list: list of manual picking times
    - snr_list: list of SNR values
    """
    # Create 4x2 subplot layout
    fig, axes = plt.subplots(4, 2, figsize=(12, 16), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})

    # Subplot titles
    subplot_titles = ['(a)', '(b)', '(c)', '(d)']

    for i in range(4):
        # Get current data
        data = data_list[i]
        pro = pro_list[i]
        pick = pick_list[i] if isinstance(pick_list, list) else pick_list
        snr = snr_list[i]

        # Find probability peak index
        max_index = np.argmax(pro)

        # Calculate picking error in milliseconds
        error_ms = (pick - max_index) / 500 * 1000

        # First column: waveform data
        axes[i, 0].plot(data, color='black', linewidth=1)
        axes[i, 0].set_ylabel('Amplitude (m/s)', fontsize=10)
        axes[i, 0].axvline(x=max_index, color='red', linestyle='-', linewidth=2, label='AI Pick')
        if isinstance(pick_list, list):  # If manual picks are provided
            axes[i, 0].axvline(x=pick, color='green', linestyle='--', linewidth=2, label='Manual Pick')

        # Add SNR annotation
        axes[i, 0].text(0.98, 0.95, f'SNR={snr:.1f} dB',
                        transform=axes[i, 0].transAxes,
                        fontsize=10, color='black', ha='right', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Add error information
        axes[i, 0].text(0.02, 0.15, f'Error: {error_ms:.1f} ms',
                        transform=axes[i, 0].transAxes,
                        fontsize=10, color='blue', ha='left', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

        axes[i, 0].grid(True, alpha=0.3)

        # Second column: probability data
        axes[i, 1].plot(pro, color='red', linestyle='-', linewidth=2, label='P-wave Probability')
        axes[i, 1].set_ylabel('Probability', fontsize=10)
        axes[i, 1].legend(loc="upper right", fontsize=9)
        axes[i, 1].grid(True, alpha=0.3)

        # Set x-axis labels for bottom row
        if i == 3:
            axes[i, 0].set_xlabel('Sample', fontsize=10)
            axes[i, 1].set_xlabel('Sample', fontsize=10)

        # Add subplot numbering
        axes[i, 0].text(0.02, 0.85, subplot_titles[i],
                        transform=axes[i, 0].transAxes,
                        fontsize=12, color='black', ha='left', va='bottom',
                        weight='bold')

        axes[i, 1].set_ylim(0, 0.8)

    # Adjust layout and save figure
    plt.tight_layout()
    fig.savefig("./example_data/waveform_probability_plot.png", dpi=300, bbox_inches='tight')
    plt.show()
    return fig


if __name__ == '__main__':
    # Initialize device and model
    device = 'cpu'
    torch.backends.cudnn.enabled = False
    T_model = torch.load(r'./model/mining_seismic_phase_picker.pt')
    T_model.to(device='cpu')


    with h5py.File(r'./example_data/example_data.hdf5', 'r') as f:
        # Get saved "waveform" dataset
        ds = f['waveform']
        # Process data
        data1, p_pick = process_data(ds)

    # Calculate original SNR
    target_snr_1 = calculate_snr(data1, p_pick, window_size=200)

    # Generate data with SNR in [10, 15] range
    data2, target_snr_2 = adjust_snr_to_target(data1, p_pick, target_snr_range=[10, 15], window_size=200)
    print('Original signal SNR: {:.2f} dB'.format(target_snr_1))
    print('Data2 SNR: {:.2f} dB'.format(target_snr_2))

    # Generate data with SNR in [0, 5] range
    data3, target_snr_3 = adjust_snr_to_target(data1, p_pick, target_snr_range=[0, 5], window_size=200)
    print('Data3 SNR: {:.2f} dB'.format(target_snr_3))

    # Generate data with SNR in [-10, -7] range
    data4, target_snr_4 = adjust_snr_to_target(data1, p_pick, target_snr_range=[-10, -7], window_size=200)
    print('Data4 SNR: {:.2f} dB'.format(target_snr_4))

    # Run model inference
    pro_1 = run_model(data1, T_model)
    pro_2 = run_model(data2, T_model)
    pro_3 = run_model(data3, T_model)
    pro_4 = run_model(data4, T_model)

    # Prepare data for plotting
    data_list = [data1[0:1750], data2[0:1750], data3[0:1750], data4[0:1750]]
    pro_list = [pro_1[0:1750], pro_2[0:1750], pro_3[0:1750], pro_4[0:1750]]
    pick_list = [p_pick, p_pick, p_pick, p_pick]
    snr_list = [target_snr_1, target_snr_2, target_snr_3, target_snr_4]

    # Generate combined plot
    fig = plot_four_waveforms_and_probabilities(data_list, pro_list, pick_list, snr_list)