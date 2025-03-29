# Fraud Detection Using Autoencoders

This project implements an end-to-end fraud detection system using an autoencoder neural network. The goal is to identify fraudulent transactions by learning the underlying patterns of non-fraudulent behavior and flagging transactions with unusually high reconstruction errors.

## Project Overview

- **Objective:** Detect fraudulent transactions in financial datasets by leveraging advanced feature engineering and deep learning techniques.
- **Data Sources:** Two datasets are used:
  - **fraudTrain.csv:** Contains historical transaction data for training.
  - **fraudTest.csv:** Used for validating the model performance.
- **Approach:** 
  - **Exploratory Data Analysis (EDA):** Initial data exploration, visualization of transaction patterns, and investigation of class distribution.
  - **Feature Engineering:** Creation of temporal features (day of week, hour of day, time between transactions), geospatial features (haversine distance between customer and merchant), and rolling statistics (e.g., average transaction amount over the last 5 transactions).
  - **Preprocessing:** Handling missing values, standardizing numerical features, and encoding categorical variables.
  - **Modeling:** Training an autoencoder neural network to learn the typical patterns of legitimate transactions. Anomalies (i.e., potential frauds) are detected based on the reconstruction error.

## Data Preprocessing & Feature Engineering

- **Temporal Features:** Converted transaction timestamps to extract day-of-week and hour-of-day. Calculated the time elapsed since the last transaction per card.
- **Geospatial Features:** Computed the haversine distance between customer and merchant locations. Clustered merchant locations using KMeans to create regional features.
- **Rolling Statistics:** Computed a rolling mean of transaction amounts for each card to capture recent spending behavior.
- **Normalization and Encoding:** Numerical features are standardized, while categorical features are label encoded to ensure compatibility with the autoencoder.

## Autoencoder Architecture

- **Encoder:** 
  - First Dense layer with 64 neurons, followed by batch normalization, LeakyReLU activation, and dropout.
  - Second Dense layer with 32 neurons and similar normalization and activation steps.
  - A bottleneck layer with 16 neurons that captures the compressed representation of the input data.
- **Decoder:** 
  - Mirrors the encoder with layers of 32 and 64 neurons, respectively, and uses batch normalization and LeakyReLU activation functions.
  - The final output layer reconstructs the original feature space using a linear activation function.
- **Training:** The autoencoder is trained using Mean Squared Error (MSE) as the loss function, with early stopping and learning rate reduction strategies to prevent overfitting.
  - **Training Loss:** `0.2362`
  - **Validation Loss:** `0.1302`
  - **Significance:** The lower validation loss compared to training loss suggests that the model generalizes well to unseen data and is not overfitting. The small difference between the two values indicates that the model has learned useful representations without memorizing the training data.

## Evaluation Metrics

While traditional classification metrics are informative, in the context of fraud detection using an autoencoder, the model’s performance is evaluated by comparing the reconstruction error against a defined threshold. Here are some of the key metrics and their significance:

- **Reconstruction Error:**  
  - **Definition:** The average absolute difference between the original input and its reconstruction.
  - **Usage:** Transactions with reconstruction errors above a threshold (typically set as the mean error plus two standard deviations) are flagged as potentially fraudulent.
  - **Reconstruction Error Threshold:** `0.380976166091332`
  
- **Accuracy:**  
  - **Definition:** The ratio of correctly classified transactions (both fraudulent and non-fraudulent) to the total transactions.
  - **Consideration:** Given the imbalanced nature of fraud detection, accuracy alone may not be sufficient.
  
- **Precision:**  
  - **Definition:** The proportion of transactions flagged as fraud that are actually fraudulent.
  - **Importance:** High precision indicates a lower false positive rate, which is critical to avoid unnecessary alerts.
  
- **Recall (Sensitivity):**  
  - **Definition:** The proportion of actual fraudulent transactions that are correctly identified.
  - **Importance:** High recall ensures that most fraudulent transactions are detected, minimizing potential losses.
  
- **F1-Score:**  
  - **Definition:** The harmonic mean of precision and recall.
  - **Importance:** Provides a balanced metric that is especially useful when dealing with imbalanced classes.
  
*Note:* When testing on the validation set, you can compute these metrics by comparing the predicted labels (based on the reconstruction error threshold) with the ground truth (is_fraud). Due to the nature of fraud detection (high class imbalance), metrics like precision, recall, and F1-score are often more indicative of model performance than overall accuracy.

## How to Run the Project

1. **Install Dependencies:**  
   Ensure you have Python 3.x installed along with the necessary libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
   ```
2. **Clone the Repository:**
   ```bash
   git clone https://github.com/abhay-2108/Credit-Card-Fraud-Detection-Using-Autoencoders.git
   ```
3. **Navigate to the Project Directory:**
   ```bash
   cd Credit-Card-Fraud-Detection-Using-Autoencoders
   ```
4. **Data Preparation:**  
   Place the fraudTrain.csv and fraudTest.csv files in the project directory.
5. **Execute Notebooks/Scripts:**  
   Run the provided Jupyter notebook or Python scripts in sequential order to perform EDA, preprocess the data, train the autoencoder, and evaluate the results.
6. **Model Evaluation:**  
   After training, adjust the reconstruction error threshold as needed and compute performance metrics to assess the effectiveness of the model.

## Conclusion

This project demonstrates a comprehensive approach to fraud detection using deep learning techniques. By combining thorough EDA, advanced feature engineering, and an autoencoder architecture, the system is able to effectively flag anomalous transactions. Further improvements can include refining the threshold selection, incorporating additional features, and testing the model on larger datasets.

