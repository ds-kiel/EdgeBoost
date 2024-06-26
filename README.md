
# EdgeBoost: Confidence Boosting for Resource Constrained Inference via Selective Offloading

Welcome to EdgeBoost's repository. This repository contains the code for the work presented in the paper titled "EdgeBoost: Confidence Boosting for Resource Constrained Inference via Selective Offloading".

## About EdgeBoost

EdgeBoost is a selective input offloading system designed to overcome the challenges of limited
computational resources on edge devices. EdgeBoost trains and calibrates a lightweight model for deployment on the edge and, in addition, deploys a large, complex model on the cloud. During inference, the edge model makes initial predictions for input samples, and if the confidence of the prediction is low, the sample is sent to the cloud model for further processing otherwise, we accept the local prediction.

![EdgeBoost System Diagram](images/img.png)

## Installation Instructions

Follow these instructions to set up the required Python environment for running EdgeBoost. Make sure you have the following prerequisites installed on your system:

- [Python](https://www.python.org/downloads/) (Python 3.x recommended)
- [pip](https://pip.pypa.io/en/stable/installation/) (Python package manager)
- [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html) (Python environment creator)
### Installation

1. Clone this Git repository to your local machine using the following command:

   ```bash
   git clone https://github.com/ds-kiel/EdgeBoost
   cd EdgeBoost
   ```
2. Create a virtual Python environment to isolate the project dependencies. You can do this using venv if you're using Python 3.3 or newer, or virtualenv if you're using an older version of Python. Replace <env_name> with your preferred environment name:

    ```bash
    virtualenv edgeboost
    source edgeboost/bin/activate
    ``` 
3. Install the necessary Python packages by running:

   ```bash
   pip install -r requirements.txt
   ```


## Usage

### Train the Models

To train the edge/cloud model, use train.py. This repo trains MobileNetV3 Small (as edge model) and EfficientV2L(as cloud model) on CIFAR-100 dataset. But the same can be used for other datasets or models

To train a model, execute the following command:

```bash
python train.py --model_name <model_name> --data_dir <path_to_data>
```
Replace <model_name> with mobilenet_v3 or efficientnet_v2_l and <path_to_data> to store the CIFAR-100 dataset. After training, the models are saved as <model_name>.pth.

### Model Evaluation

After training your models with `train.py`, you can evaluate their performance using the `evaluate.py` script. This evaluation will output the accuracy of the chosen model on the CIFAR-100 test dataset and will calculate the Expected Calibration Error (ECE).

To evaluate a model, run the `evaluate.py` script with the required arguments:

- `--model_name`: Name of the model you wish to evaluate. Choices are `mobilenet_v3` or `efficientnet_v2_l`.
- `--model_path`: Path to the saved weights of the trained model.
- `--data_dir`: Directory path for the CIFAR-100 dataset.

Example command

```bash
python evaluate.py --model_name mobilenet_v3 --model_path path/to/mobilenet_v3_cifar100.pth --data_dir ./data
```
This command initializes the model, loads the saved weights, and performs an evaluation to report the accuracy and ECE. It also saves the uncalibrated probabilities of the test set and the test labels in .npy format which are then used in the offloading process.

### Model Calibration

Use the following command to calibrate the trained model

```bash
python calibrate.py --model_path PATH_TO_YOUR_MODEL --model_name NAME_OF_YOUR_MODEL --data_dir PATH_TO_CIFAR100_DATASET
```
The output will display the test loss, accuracy on the test set, and the Expected Calibration Error (ECE) after scaling. The model is calibrated on calibration set which is the subset of the test set. The script will also generate a .npy file containing the calibrated model's probabilities:


### Selective Offloading


To execute the offload script, use the following command:

```bash
python offload.py --mobnet_cal PATH_TO_CALIBRATED_MOBNET_PROBS --mobnet_uncal PATH_TO_UNCALIBRATED_MOBNET_PROBS --effnet PATH_TO_EFFICIENTNET_PROBS --labels PATH_TO_LABELS
```

- `--mobnet_cal` is the path to the .npy file containing the calibrated MobileNet probabilities.
- `--mobnet_uncal` is the path to the .npy file containing the uncalibrated MobileNet probabilities.
- `--effnet` is the path to the .npy file containing the EfficientNet probabilities.
- `--labels` is the path to the .npy file containing the actual labels for the test data.

The script will output:

- Calibration curve plot saved as `cifar-reliability.pdf`.
- Console output detailing the combined accuracy and offloading metrics at various thresholds.

## Trained Models

We provide pre-trained models and probability files for three datasets used in our paper. You can download them from the following Google Drive link:

[Trained Models and Probabilities](https://drive.google.com/drive/folders/1f4fM1NMzThR_pv9G7w75pkJYEbISJhjx?usp=sharing)






