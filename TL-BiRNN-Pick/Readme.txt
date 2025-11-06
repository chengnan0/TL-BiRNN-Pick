## Code Description

This code implements phase picking for mining seismic data, optimized through transfer learning. The following describes the code organization and usage instructions:

1. **Initial Training Code**:
   - Includes data preprocessing, feature extraction, and model training processes. Please refer to the `initial_training.py` file.

2. **Transfer Learning Code**:
   - Contains transfer learning steps that adapt the initially trained model to the phase picking task. Please refer to the `transfer_learning.py` file.
   - The transfer learning approach utilizes a pre-trained model and performs fine-tuning to accommodate phase picking requirements.

3. **Phase Picking Code**:
   - Processes and predicts phases in mining seismic data. Please refer to the `phase_pick.py` file.

4. **Transfer-Learned Model**:
   - We provide the complete model after transfer learning, stored in the `mining_seismic_phase_picker.pt` file.

### Usage Instructions:

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt