# Process Predictor: Deep Learning for Business Process Monitoring

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

## Project Overview
This repository contains an end-to-end implementation of a predictive process monitoring system. Utilizing a **Long Short-Term Memory (LSTM)** neural network, the system forecasts the most probable subsequent activities in a business process. The model is trained and validated on the **BPI Challenge 2012** dataset, which consists of real-world loan application event logs.

## Technical Features
* **Architecture:** Multi-layer LSTM network featuring an Embedding layer for categorical activity representation.
* **Inference Engine:** High-performance REST API developed with FastAPI, providing the best prediction and its confidence score.
* **Data Integrity:** Implements case-based data splitting to prevent data leakage and ensures robust validation.
* **Deployment:** Containerized via Docker to provide a consistent execution environment across various infrastructures.

## System Architecture


1.  **Data Processing:** Transforms raw event logs into fixed-length sliding-window sequences.
2.  **Modeling:** Implements a sequence-to-label architecture using PyTorch.
3.  **Serving:** A production-grade API layer that manages model lifespan and handles concurrent inference requests.
4.  **Containerization:** Optimized Docker image for scalable deployment.

## Installation and Usage

### Prerequisites
* Docker installed on the host machine.

### Deployment via Docker
1.  **Build the Image:**
    ```bash
    docker build -t process-predictor:v2 .
    ```

2.  **Initialize the Container:**
    ```bash
    docker run -p 8000:8000 process-predictor:v2
    ```

3.  **Access Documentation:**
    Interactive API documentation (Swagger UI) is available at [http://localhost:8000/docs](http://localhost:8000/docs).

## Model Evaluation
Performance metrics are derived from a 20% hold-out test set, ensuring the model's ability to generalize to unseen traces.

| Metric | Value |
| :--- | :--- |
| **Training Accuracy** | 84.67% |
| **Test Accuracy** | 84.00% |



## Methodology and Acknowledgments
* **Dataset:** BPI Challenge 2012, provided by the Business Process Intelligence Challenge.
https://data.4tu.nl/articles/dataset/BPI_Challenge_2012/12689204
* **Development Workflow:** This project was developed using industry-standard machine learning practices, including comprehensive data preprocessing, model evaluation using precision/recall metrics, and containerized deployment. 
* **AI Assistance:** Generative AI tools were utilized during development for rapid prototyping of boilerplate code, environment configuration, and architectural review.

## License
This project is licensed under the MIT License.


