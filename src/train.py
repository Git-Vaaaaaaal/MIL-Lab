
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from builder_utils import MILDataset
from builder import create_model
from function import train_model

import os
""" import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dataset_path", type=str)
parser.add_argument("csv_file", type=str)
parser.add_argument("output_path", type=str)
args = parser.parse_args() """

#Variable
dataset_path = "src/features"
print("Working directory:", os.getcwd())
csv_path = "src/clinical_data.csv"
features_list = os.listdir(dataset_path)
model_names = ["abmil", "clam", "dsmil", "transmil", "dftd", "ilra", "rrt", "wikg", "transformer"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 80

output_path = "src/output"
output_path = os.path.join(os.getcwd(), output_path)


#"abmil.base.uni.op-108", "abmil.base.uni.op-109", "abmil.base.uni.op-110", "abmil.base.conch_v15.pc108-24k", "abmil.base.uni_v2.pc108-24k",
#"abmil.base.uni.pc108-24k", 


for dataset in features_list: 
    for model_name in model_names:
            print(f"\n--- Training model: {model_name} ---")
            
            model = create_model(
                model_name=model_name,
                num_classes=2,
                from_pretrained=False
            )
            
            train_model(model, dataset_path, device, epochs, str(model_names), output_path)