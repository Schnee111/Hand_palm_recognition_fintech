# Hand Palm Recognition for Financial and technology 

A project for hand palm image-based prediction and verification experiments. This repository contains the source code for implementing a biometric security system using hand palm features, specifically for fintech applications.

## About the Project
This project explores the use of hand palm recognition as a biometric authentication method. The primary goal is to build and evaluate models for both prediction (identifying a user from a database) and verification (confirming a user's identity). This technology has significant potential for enhancing security in the financial technology (fintech) sector.

## Key Features
- Data Preprocessing: Scripts to prepare and augment hand palm images.

- Feature Extraction: Techniques to extract distinctive features from the images.

- Prediction Model: A model to identify individuals based on their hand palm.

- Verification Model: A model to verify the claimed identity of an individual.

## Dataset
The dataset for this project consists of hand palm images collected for training and evaluation.

Note: The data/ directory, which contains the dataset, is not included in this repository due to its size. You will need to add the data/ folder and structure it as follows to run the training scripts:

```
data/
├── train/
│   ├── user_1/
│   │   ├── image_1.jpg
│   │   └── ...
│   └── user_2/
└── test/
    ├── user_1/
    └── user_2/
```
