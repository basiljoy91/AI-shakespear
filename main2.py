import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import re

# Download dataset
filepath = tf.keras.utils.get_file(
    'shakespeare.txt',
    origin='https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
)

# Preprocess text
text = open(filepath, 'rb').read().decode('utf-8').lower()
text = text[100000:600000]  # Use a focused subset
text = re.sub(r'[^\w\s]', '', text)  # Remove special chars
text = text.replace('\n', ' \n ')  # Treat newlines as separate tokens

# Create character mappings
chars = sorted(set(text))
char_to_idx = {c:i for i,c in enumerate(chars)}
idx_to_char = {i:c for i,c in enumerate(chars)}

# Prepare training data
SEQ_LENGTH = 60  # Increased sequence length
STEP_SIZE = 1    # More overlapping sequences

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

# Enhanced model architecture
model = Sequential([
    Input(shape=(SEQ_LENGTH, len(chars))),
    LSTM(256, return_sequences=True),
    LSTM(256),
    Dense(len(chars)*2, activation='relu'),
    Dense(len(chars), activation='softmax')
])

# Improved training configuration
model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(learning_rate=0.001),
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    ModelCheckpoint('best_model.keras', save_best_only=True),
    EarlyStopping(patience=3)
]

# Train model
history = model.fit(
    x, y,
    batch_size=128,
    epochs=20,
    validation_split=0.1,
    callbacks=callbacks
)

# Save model
model.save('textgenerator.keras')

# Generation functions
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def format_output(text):
    sentences = text.split('. ')
    sentences = [s.capitalize() for s in sentences]
    return '. '.join(sentences)

def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    
    for _ in range(length):
        x_pred = np.zeros((1, SEQ_LENGTH, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_idx[char]] = 1

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = idx_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char
    
    return format_output(generated)

# Beam search alternative
def beam_search_generate(seed, length=300, beam_width=3):
    sequences = [[seed, 1.0]]  # [text, score]
    
    for _ in range(length):
        all_candidates = []
        for seq, score in sequences:
            x = np.zeros((1, SEQ_LENGTH, len(chars)))
            for t, char in enumerate(seq[-SEQ_LENGTH:]):
                x[0, t, char_to_idx[char]] = 1
            
            preds = model.predict(x, verbose=0)[0]
            top_indices = np.argsort(preds)[-beam_width:]
            
            for idx in top_indices:
                candidate = [seq + idx_to_char[idx], 
                            score * preds[idx]]
                all_candidates.append(candidate)
        
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        sequences = ordered[:beam_width]
    
    return format_output(sequences[0][0])

# Load best model
model = tf.keras.models.load_model('best_model.keras')

# Generate samples
print('\n------ Temperature 0.2 ------')
print(generate_text(300, 0.2))
print('\n------ Temperature 0.4 ------')
print(generate_text(300, 0.4))
print('\n------ Temperature 0.6 ------')
print(generate_text(300, 0.6))
print('\n------ Beam Search (width=3) ------')
print(beam_search_generate('ROMEO:', 300))