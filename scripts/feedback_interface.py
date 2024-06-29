import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# Load the synthetic data
data = pd.read_csv('../data/bitcoin_futures.csv', parse_dates=['Date'])
data.set_index('Date', inplace=True)

# Define a custom trading environment for Bitcoin futures
import gymnasium as gym
from gym_anytrading.envs import StocksEnv

class BitcoinFuturesEnv(StocksEnv):
    _process_data = lambda self: (data['Close'], data['Volume'])

# Initialize the environment
env = BitcoinFuturesEnv(df=data, window_size=10, frame_bound=(10, len(data)))

# Integrate FinGPT
from fingpt import FinGPT

# Initialize FinGPT model
fingpt_model = FinGPT()

def get_fingpt_predictions(observation):
    """Generate multiple responses using FinGPT based on the observation."""
    responses = fingpt_model.predict(observation)
    return responses

def plot_trade_data(data):
    """Plot trade data to visualize market synchrony."""
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], label='Close Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

def main():
    st.title("RL Model Feedback")

    # Simulate multiple responses using FinGPT
    obs = env.reset()
    responses = get_fingpt_predictions(obs)
    
    # Show trade data and charts for each response
    for i, response in enumerate(responses):
        st.subheader(f"Response {i+1}")
        st.write(response)
        
        # Plot trade data (simulated for demonstration)
        plot_trade_data(data)
        
    chosen_response = st.radio("Select the best response based on market synchrony:", [f"Response {i+1}" for i in range(len(responses))])

    if st.button("Submit"):
        st.write(f"You selected: {chosen_response}")
        # Save feedback for training
        with open("../data/feedback.txt", "a") as f:
            f.write(f"{chosen_response}\n")

if __name__ == "__main__":
    main()