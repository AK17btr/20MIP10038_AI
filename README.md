# AI Trademarkia Classification

This project involves building an AI model to classify trademark classes based on descriptions of goods and services. The model is trained using data from the USPTO ID Manual and leverages BERT for classification. The project includes a REST API for inference and is dockerized for deployment.

# Table of Contents
Project Overview
Setup
Data
Model Training
API
Docker
Contributing
License
Project Overview
The AI Trademarkia Classification project aims to develop a machine learning model that predicts trademark classes from textual descriptions of goods and services. The project includes:

Data preprocessing and model training using BERT.
Logging and monitoring with tools like WandB, MLFlow, and TensorBoard.
A REST API for model inference.
Dockerization for easy deployment.
Setup
Clone the Repository

bash
Copy code
git clone https://github.com/<your-github-username>/<REGISTER_NUMBER>_AI.git
cd <REGISTER_NUMBER>_AI
Create and Activate a Virtual Environment

bash
Copy code
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
Install Dependencies

bash
Copy code
pip install -r requirements.txt
Data
The dataset is a JSON file containing the following columns:

id_tx: Transaction ID
class_id: Trademark class ID
description: Description of goods/services
status: Status of the item
The data file is located at: C:\Users\sendm\Downloads\idmanual.json

Model Training
The model is built using BERT for sequence classification. Training is performed using PyTorch.

Training Script
To train the model, run:

bash
Copy code
python train_model.py
The script performs the following steps:

Loads and preprocesses the data.
Encodes labels.
Trains the BERT model.
Saves the trained model and tokenizer.
API
The REST API is built using Django and provides endpoints for classifying descriptions.

API Endpoints
POST /classify/
Description: Classify a given description into trademark classes.
Request Body:
json
Copy code
{
  "user_id": "<user_id>",
  "description": "<description>"
}
Response:
json
Copy code
{
  "class_id": "<predicted_class_id>",
  "class_description": "<description>"
}


