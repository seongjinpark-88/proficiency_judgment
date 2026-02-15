# proficiency_judgment

Automatic proficiency judgment of non-native speech using neural networks. Evaluates accentedness, fluency, and comprehensibility using multi-task learning with acoustic and text features.

## Overview

This project implements several neural network architectures for automatic speech proficiency assessment:
- Single-input models (acoustic features only)
- Multi-input models (acoustic + text features)
- Multi-task learning models (joint prediction of multiple proficiency dimensions)
- Attention-based models for improved feature weighting

## Related Publications

- Park, S. & Culnan, J. (2019). "Automatic perceptual judgment using neural networks." *JASA* 146(4_Supplement), 2957.
- Park, S. & Culnan, J. (2021). "Automatic proficiency judgments: Accentedness, fluency, and comprehensibility." *JASA* 150(4_Supplement), A357.
- Park, S. (2021). "Human and Machine Judgment of Non-Native Speakers' Speech Proficiency." PhD Thesis, The University of Arizona.

## Project Structure

```
├── bin/                        # Entry-point scripts
│   ├── train_torch.py          # PyTorch training script
│   └── user_distripution.py    # User distribution analysis
├── data_prep/                  # Data preparation
│   ├── acoustic_extraction.py  # Acoustic feature extraction
│   ├── audio_to_w2v.py         # Wav2Vec feature extraction
│   ├── data_prep.py            # General data preparation
│   ├── roberta_prep.py         # RoBERTa text feature extraction
│   └── w2v_prep.py             # Wav2Vec preparation
├── models/                     # Model architectures
│   ├── attn_models.py          # Attention-based models
│   ├── input_models.py         # Input processing models
│   ├── train_and_test_models.py
│   └── parameters/             # Model hyperparameters
├── train_and_test_models/      # Training scripts
│   ├── train_single_input_cv.py
│   ├── train_multi_single_cv.py
│   ├── train_multi_multi_cv.py
│   ├── train_rnn.py
│   └── train_rnn_mtl.py        # Multi-task learning
└── keras_test/                 # Baseline models
    └── SimpleFNN.py
```

## Requirements

- Python 3.7+
- PyTorch
- Transformers (for RoBERTa features)
- Wav2Vec 2.0
- NumPy, pandas, scikit-learn

## Usage

1. Prepare data: `python data_prep/data_prep.py`
2. Extract features: `python data_prep/acoustic_extraction.py`
3. Train model: `python bin/train_torch.py`

See individual scripts for configuration options.

## Author

Seongjin Park — [seongjinpark.com](https://seongjinpark.com)

## License

MIT
