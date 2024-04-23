
# EdgeBoost: Confidence Boosting for Resource Constrained Inference via Selective Offloading

Welcome to EdgeBoost's repository. This repository contains the code for the work presented in the paper titled "EdgeBoost: Confidence Boosting for Resource Constrained Inference via Selective Offloading"

## About EdgeBoost

EdgeBoost is a selective input offloading system designed to overcome the challenges of limited
computational resources on edge devices. EdgeBoost trains and calibrates a lightweight model for deployment on the edge and, in addition, deploys a large, complex model on the cloud. During inference, the edge model makes initial predictions for input samples, and if the confidence of the prediction is low, the sample is sent to the cloud model for further processing otherwise, we accept the local prediction.

![EdgeBoost System Diagram](images/image.png)

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

### Train from Scratch








