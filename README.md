# TL-BiRNN-Pick

**TL-BiRNN-Pick** is an automated microseismic phase picking framework designed specifically for underground mining environments. It utilizes a transfer learning approach, leveraging features learned from a large-scale natural earthquake dataset (Diting 2.0) and fine-tuning them for mining-specific seismic signals.

## Core Features

* **Pre-training Support**: Includes scripts to utilize the Diting 2.0 dataset for initial model training.
* **Transfer Learning Optimization**: Fine-tuned to accommodate the unique requirements and noise characteristics of mining seismic data.
* **Ready-to-Use Model**: Provides a final transfer-learned model for high-precision P-wave.

## Repository Structure

The repository is organized as follows:

* `initial_training.py`: Script for the initial model training process.
* `transfer_learning.py`: Script for adapting the pre-trained model to mining-specific tasks.
* `Phase_pick.py`: The main script for seismic phase picking and prediction.
* `model/`: Contains the final transfer-learned weights (`mining_seismic_phase_picker.pt`).
* `example_data/`: Contains anonymized example mining seismic data (`example_data.hdf5`) for testing.
* `requirements.txt`: List of Python dependencies required to run the project.

## Installation

Ensure you have **Python 3.8 or higher** installed. You can install the required dependencies using:

```bash
pip install -r TL-BiRNN-Pick/requirements.txt
