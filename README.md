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

***************************************************************************************************

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


***************************************************************************************************


Part 2. Data Preprocessing

Objective: 

This notebook focuses on cleaning and preprocessing raw forex rate data to prepare it for further analysis and modeling. Since the input data from polygon.io has already processed, I just simply rechecked the data value and standardize the data frame.

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


***************************************************************************************************

 
Step 3 & 4. CNN-LSTM Model and Optimization

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


· Note: From the training loss and validation loss plot, we can found out tha the Regularized LSTM performs the best. Thus, we use it as the final LSTM combined with CNN to process into next step.

5. 📈 Training and Evaluation

· Models are trained using CrossEntropyLoss and the Adam optimizer.

· Training loss curves are visualized to monitor overfitting.

· Accuracy on the test set is computed to assess performance.

6. 📉 Early Stopping and Validation

· Regularized LSTM includes an early stopping mechanism based on validation loss.

· Both training and validation losses are plotted for diagnostic analysis.

7. ✅ Outcome

· The best-performing model (Regularized LSTM) achieved stable accuracy with good generalization.

· It is suitable for binary classification (up/down price prediction) and can be expanded for multi-class or regression tasks in future iterations.
  

***************************************************************************************************

Step 5. Reinforcement Trading Model

Objective: 

This project implements a reinforcement learning (RL) trading agent using the Proximal Policy Optimization (PPO) algorithm from the stable-baselines3 library. It is designed to learn and optimize trading strategies based on technical indicators extracted from historical cryptocurrency price data.

🔧 Key Components

1. rl_trading_agent.py
   
✅ Custom Trading Environment

Defines a custom TradingEnv class compatible with OpenAI Gym.

· Inputs:

—— Technical indicators: RSI, MACD, EMA, Bollinger Bands, momentum, etc.

—— Observation: A sliding window (default 24 steps) of historical data.

· Action space:

—— Continuous between [0.0, 1.0], representing portfolio position (no position to full long).

· Reward:

—— Proportional to the asset return based on the current position.

· Output:

—— At each step, the environment returns the updated observation, reward, asset value, and done flag.

2. ppo_trainer.py
   
✅ Model Training & Logging

· Loads pre-cleaned crypto price data from CSV.

· Wraps the TradingEnv in a DummyVecEnv (vectorized environment) required by stable-baselines3.

· Initializes a PPO model with MLP policy and defined hyperparameters (e.g., n_steps=2048, learning_rate=3e-4).

· Trains the agent for 50,000 timesteps.

· Saves:

—— The trained model: ppo_trading_agent.zip

—— Trading logs during testing: ppo_trading_log.csv


***************************************************************************************************

Step 6. CNN-LSTM-RL Model

Objective:

To develop a deep reinforcement learning trading agent that combines CNN-LSTM-based feature extraction with policy optimization using PPO, aimed at predicting market movements and optimizing portfolio returns in the cryptocurrency market.

🔧 Core Components

1. 📥 Data Preprocessing

· Loads cleaned crypto market data (OHLCV and technical indicators).

· Standardizes features and constructs sequences with a window size (e.g., 24 time steps).

2. 🧠 CNN-LSTM Feature Extractor

· Implements a hybrid deep learning model:

· CNN: Captures local temporal patterns within each time window.

· LSTM: Captures long-term sequential dependencies across windows.

· The extracted features are used to inform the trading decision logic.

3. 🎮 Custom Gym Environment

· Integrates the CNN-LSTM as an internal signal generator or state transformer.

· Uses continuous action space (0–1) to represent asset allocation.

· Reward is based on portfolio return over each time step.

4. 🤖 PPO Training Loop

· Utilizes stable-baselines3.PPO to train the agent within a custom TradingEnv.

· Saves both the trained agent and trading logs for backtesting/evaluation.

5. 📈 Evaluation and Logging

· Tracks agent performance including asset value, reward per step, and position decisions.

· Exports results to CSV for further visualization and analysis.

✅ Model Strengths

· Deep Feature Representation: Combines CNN and LSTM to enhance pattern recognition.

· Policy Learning via PPO: Learns trading strategy through reinforcement signals.

· Custom RL Environment: Tailored for single-asset crypto trading with realistic settings.

Pipline:
![6a73cc0c-b3b8-4e7f-9907-6013fcccf86e](https://github.com/user-attachments/assets/bd2ed517-7351-43f4-8cb5-2b23053edf54)

📊 Comparison: CNN-LSTM-PPO vs Traditional LSTM-PPO Agent
![image](https://github.com/user-attachments/assets/20de00a2-bd5c-48b1-a478-1602eea50fc7)


***************************************************************************************************


