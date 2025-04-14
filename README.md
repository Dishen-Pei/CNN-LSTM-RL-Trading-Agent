# CNN-LSTM-RL-Trading-Agent
A real-time cryptocurrency trading bot powered by Proximal Policy Optimization (PPO) and integrated with the Binance Testnet API. This project uses deep reinforcement learning and LSTM-enhanced market signals to dynamically adjust trading positions.

This Agent is Combined with 7 parts:
  1. Set up Data Interface with Polygon.io
  2. Data
  3. CNN-LSTM Model
  4. Model Evaluation & Optimization
  5. Reinforcement Learning Trading Model
  6. CNN-LSTM-RL Model
  7. Trading-Set



Part 1. Set up Data Interface with Polygon.io

🔧 Main Functionalities

1. Real-Time Data Fetching
   
· Retrieves current exchange rate data for currency pairs (e.g., EURUSD, USDCAD) via Polygon.io API.

· Extracts rate values and timestamps for further analysis.

2. Statistical Analysis
   
· Calculates basic metrics such as maximum, minimum, mean, and volatility.

· Computes Keltner Channel boundaries to support technical analysis.

3. Database Integration
   
· Supports dual storage: structured (SQLite) and unstructured (MongoDB).

· Includes automated connection, deletion, and insertion procedures for database updates.

💡 Technical Highlights

· Modular design with reusable functions (fetch_fx_rate, compute_stats, etc.).

· Real-time data processing with database persistence.

· Easily extensible to support more indicators, data sources, or output formats.

📈 Use Cases

· Quantitative trading research and backtesting.

· Real-time forex analytics and dashboard development.

· Academic assignments or fintech project foundations.



Part 2. Data Preprocessing

Objective: This notebook focuses on cleaning and preprocessing raw forex rate data to prepare it for further analysis and modeling. Since the input data from polygon.io has already processed, I just simply rechecked the data value and standardize the data frame.

Processed Data Link: https://drive.google.com/file/d/17V9UVHAW8-2ISfODsBHK5bWiU2WlbQIf/view?usp=sharing

🔧 Data Processing Overview

1. Data Loading

· Multiple CSV files representing different currency pairs (e.g., EURUSD, USDCHF) are loaded into the environment.

· Timestamps are parsed and converted to a consistent datetime format.

2. Handling Missing Values

· Missing or null values are identified using isnull() and dropna() functions.

· Some rows with incomplete or corrupted entries are removed to maintain data integrity.

3. Datetime Normalization

· The raw timestamp data is converted to hourly intervals using pd.to_datetime() and resample('H').

·Ensures all data points are aligned on a unified hourly timeline for consistency across currency pairs.

4. Feature Engineering

· New features are computed, including moving averages, standard deviations, and rate differentials.

· These derived columns enhance the data’s predictive quality for future modeling.

5. Exporting Cleaned Data

· Final cleaned datasets are saved into new CSV files for each currency pair.

· These outputs are structured for direct use in machine learning pipelines or financial analysis tools.

💡 Key Strengths

· Modular and repeatable preprocessing pipeline for multi-currency datasets.

· Time-series aware transformation ensures consistent temporal alignment.

· Optimized for integration with tools like PyCaret or backtesting frameworks.


 

Step3 & 4. CNN-LSTM Model and Optimization

Objective:
This project aims to build a deep learning model using a combination of Convolutional and Recurrent Neural Networks to predict the directional movement (up/down) of cryptocurrency prices based on historical price data and technical indicators.

🔧 Model Construction Process
1. 📥 Data Loading & Feature Engineering

· The dataset (cleaned_crypto_data.csv) contains cryptocurrency price and volume data.

· Technical indicators such as RSI, MACD, EMA, Bollinger Bands, and momentum were computed using the ta library.

· A binary label is created where:
 
    1 = future price > current price

    0 = otherwise.

2. 🧹 Data Preprocessing

· Missing values are handled and dropped after computing indicators.

· Features are standardized using StandardScaler.

· Sequences of past 24 hours of data (window=24) are created as model inputs.

3. 📊 Train/Test Split

· The data is split into 80% training and 20% testing sets.

· DataLoader is used to handle batching and shuffling.

4. 🧠 Model Variants Implemented

· Basic LSTM: A two-layer LSTM followed by a fully connected output layer.
![image](https://github.com/user-attachments/assets/64d11626-81cd-4b56-9bed-eb026e6e9b09)

· Bidirectional LSTM: Enhances temporal understanding using both forward and backward sequences.
![image](https://github.com/user-attachments/assets/088b87b0-20b8-4246-9618-30423d272e2c)


· Regularized LSTM: Adds dropout and layer normalization to reduce overfitting.
![image](https://github.com/user-attachments/assets/287372a8-3c30-4b25-a936-f44f4a7dc189)


· Note: While CNN was mentioned in the filename, the model is purely LSTM-based in the current version.

📈 Training and Evaluation

Models are trained using CrossEntropyLoss and the Adam optimizer.

Training loss curves are visualized to monitor overfitting.

Accuracy on the test set is computed to assess performance.

📉 Early Stopping and Validation

Regularized LSTM includes an early stopping mechanism based on validation loss.

Both training and validation losses are plotted for diagnostic analysis.

✅ Outcome
The best-performing model (Regularized LSTM) achieved stable accuracy with good generalization.

It is suitable for binary classification (up/down price prediction) and can be expanded for multi-class or regression tasks in future iterations.
  
