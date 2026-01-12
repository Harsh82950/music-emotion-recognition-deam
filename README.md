# music-emotion-recognition-deam
CNN-based music emotion recognition system using mel spectrograms and the DEAM dataset, implemented in PyTorch.
This is a compact README that recruiters actually read in 30–40 seconds.
You can replace your current README with this, or keep it as a shorter version.

🎵 Music Emotion Recognition (DEAM Dataset)
🔍 Overview

A CNN-based Music Emotion Recognition system that predicts continuous valence and arousal values from music audio using mel spectrograms and PyTorch.

⚙️ Approach

Extracted mel spectrograms from audio using Librosa

Built a custom PyTorch Dataset & DataLoader

Designed a CNN regression model for emotion prediction

Trained and evaluated the model on the DEAM dataset using MSE loss

🧠 Model

Input: Mel spectrogram (128 × time)

Architecture: 3 Conv blocks + regression head

Output: Valence & Arousal

🛠️ Tech Stack

Python · PyTorch · Librosa · NumPy · Pandas

▶️ Run
python -m src.train
python -m src.evaluate

🎯 Use Cases

Music recommendation · Mood detection · Affective computing
