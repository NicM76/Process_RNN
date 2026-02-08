# Containerized Predictive Process Monitor

## Project Overview
This project builds a Deep Learning microservice capable of predicting the next event in a business process trace. It leverages **Process Mining** concepts combined with **LSTM (Long Short-Term Memory)** neural networks.

Unlike static analysis, this tool is designed as a deployed **MLOps** solution, containerized with **Docker** and served via a **FastAPI** endpoint.

## Tech Stack
* **Deep Learning:** PyTorch (LSTM/RNN)
* **Data Processing:** Pandas, Numpy
* **Backend:** FastAPI
* **Containerization:** Docker
* **Process Mining:** BPI Challenge Datasets

## Architecture
1.  **Data Ingestion:** Preprocessing event logs (XES/CSV) into temporal sequences.
2.  **Training:** An LSTM model trained to minimize CrossEntropyLoss on event prediction.
3.  **Inference:** A REST API that accepts partial traces and returns the predicted next activity.

## Setup
(To be added: Instructions on how to build the Docker image)

## Data
https://data.4tu.nl/articles/dataset/BPI_Challenge_2012/12689204
