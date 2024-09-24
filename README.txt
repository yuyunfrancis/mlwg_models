### Instructions to Run the Code

1. **Install Dependencies**:
   Ensure you have the required Python libraries installed. You can install them using `pip`:
   ```bash
   pip install torch nltk numpy matplotlib seaborn
   ```

2. **Prepare the Dataset**:
   - Place your dataset in the appropriate directory structure as described in the `data/README` file.
   - Ensure the pretrained word embeddings file `all.review.vec.txt` is in the correct location.

3. **Run the Preprocessing Script**:
   Execute the preprocessing script to clean and tokenize the data:
   ```python
   # Example preprocessing script
   from preprocess import preprocess_string, tokenize, padding_
   ```

4. **Train the Models**:
   - Run the CNN model training script:
     ```bash
     python CNN.py
     ```
   - Run the LSTM model training script:
     ```bash
     python LSTM.py
     ```

5. **Evaluate the Models**:
   Use the provided functions to evaluate the models on test data and predict sentiments for sample texts.

6. **Visualize Results**:
   The training scripts will generate plots for training and validation losses and accuracies. Ensure you have `matplotlib` and `seaborn` installed to view these plots.

6. **Predict Sentiments**:
   Use the `predict_sentiment` function to predict the sentiment of new text samples:
   ```python
   sample_text = "Your text here"
   sentiment = predict_sentiment(model, sample_text)
   print(f"Sentiment: {'Positive' if sentiment > 0.5 else 'Negative'} (Score: {sentiment:.2f})")
   ```