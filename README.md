# next-word-prediction_GRU-RNN
Prediction next word using a Gated Recurrent Unit Recurrent Neural Network

Dataset used: Shakespeare's Hamlet

Steps taken to implement the LSTM (GRU RNN):
1. Data Collection
2. Data Processing:
    * Tokenizing the data
    * Converting into sequences
    * Padding to ensure a uniform input size
    * Train/Test splits
3. Model Training:
    * Uses LSTM with an Embedding layer
    * Two LSTM layers
    * Dense output layer with softmax activation (multi-classification)
    * Uses early stop mechanism to prevent overfitting --> Monitors validation loss metric
4. Deployed using Streamlit: [Check it out here!](https://next-word-predictiongru-rnn-5pwr2zkncyzeekupkmpzhu.streamlit.app/)
