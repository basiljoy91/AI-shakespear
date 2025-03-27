import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Input
from tensorflow.keras.optimizers import RMSprop

# Download dataset
filepath = tf.keras.utils.get_file(
    'shakespeare.txt',
    origin='https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
)

# Preprocess text
text = open(filepath, 'rb').read().decode('utf-8').lower()
text = text[100000:600000]  # Use a subset

# Create character mappings
chars = sorted(set(text))
char_to_idx = {c:i for i,c in enumerate(chars)}
idx_to_char = {i:c for i,c in enumerate(chars)}

# Prepare training data
SEQ_LENGTH = 60
STEP_SIZE = 1

sentences = []
next_chars = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i:i+SEQ_LENGTH])
    next_chars.append(text[i+SEQ_LENGTH])

# Vectorize data
x = np.zeros((len(sentences), SEQ_LENGTH, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i,t,char_to_idx[char]] = 1
    y[i,char_to_idx[next_chars[i]]] = 1

# Build model (with corrected input specification)
model = Sequential([
    Input(shape=(SEQ_LENGTH, len(chars))),  # Explicit input layer
    LSTM(256),
    Dense(len(chars)),
    Activation('softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(learning_rate=0.005),
    metrics=['accuracy']
)

# Train model
model.fit(x, y, batch_size=256, epochs=20)

# Save with proper extension
model.save('textgenerator.keras')  # or .h5

print("Model trained and saved successfully!")


model = tf.keras.models.load_model('textgenerator.keras')

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(lenght, temperature):
    start_index=random.randint(0, len(text) - SEQ_LENGTH - 1)
    generate = []
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generate += sentence
    for i in range(lenght):
        x = np.zeros((1, SEQ_LENGTH, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_to_idx[char]] = 1

            predictions = model.predict(x, verbose=0)[0]
            next_index = sample(predictions, temperature)
            next_chars =  idx_to_char[next_index]

            generate += next_chars
            sentence = sentence[1:] + next_chars
        return generate
print('------0.2------')
print(generate_text(300, 0.2))
print('------0.4------')
print(generate_text(300, 0.4))
print('------0.6------')
print(generate_text(300, 0.6))
print('------0.8------')
print(generate_text(300, 0.8))
print('------1------')
print(generate_text(300, 1.0))
