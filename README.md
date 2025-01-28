
# **Federated Dynamic Model for Spatiotemporal Data Forecasting in Transportation**

This repository contains the implementation of the **Federated-LSTM-DSTGCRN** model, proposed in the manuscript:

> **"Federated Dynamic Model for Spatiotemporal Data Forecasting in Transportation"**, submitted to IEEE Transactions on Intelligent Transportation Systems (Jan 2025).

## **Overview**

This project brings Federated Learning to spatiotemporal forecasting, making it more accurate, and privacy-friendly. To keep things robust and efficient, we introduce a Client-Side Validation mechanism, in which, clients check updates before applying them, so only the best improvements make it into the model. This means better accuracy, faster convergence, and no junk updates. 

This model is built for real-world spatiotemporal forecasting tasks like multimodal transport demand and OD matrix prediction, all while maintaining data privacy and decentralization.

Due to privacy constraints, we are unable to share OD data in this repository. However, we do provide datasets for multimodal transport demand, including Chicago taxi, New York taxi, and New York bike data, as well as weather data for the corresponding locations.


## **Installation**

### **1. Clone the Repository**
```bash
git clone https://github.com/nhat-thien/Federated-LSTM-DSTGCRN
cd Federated-LSTM-DSTGCRN  
```

### **2. Install Dependencies**
Ensure you have Python 3.8+ installed. Then, install the required dependencies:

```bash
pip install -r requirements.txt  
```


## **Usage**

### **1. Prepare Your Data**
- Place your dataset in the `DATA/` folder.
- Ensure it follows the required format for training.

### **2. Configure Your Experiment**
Modify the following files to set up your experiment:

- **`TestCase.py`** → Define the test case and dataset configurations.
- **`Hyperparameters.py`** → Set the base model and federated learning scheme.

### **3. Run the Model**
Execute the training script:

```bash
python Experiments.py  
```


## **Repository Structure**
```
📂 Federated-LSTM-DSTGCRN  
│── 📂 DATA/                    # Folder for datasets  
│── 📂 FL_HELPERS/              # Handle the federated learning
│── 📂 MODELS/                  # The models
│── Hyperparameters.py          # Configuration for base model & FL scheme  
│── TestCase.py                 # Define test cases  
│── Experiments.py              # Implement the experiment(s)  
│── requirements.txt            # Dependencies  
│── README.md                   # Project documentation  
```


## **Citing This Work**
If you use this repository in your research, please cite:

```bibtex
@article{ToBeUpdated,  
  title={Federated Dynamic Model for Spatiotemporal Data Forecasting in Transportation},  
  author={Names},  
  journal={IEEE Transactions on Intelligent Transportation Systems},  
  year={2025},  
  status={Submitted}  
}
```

