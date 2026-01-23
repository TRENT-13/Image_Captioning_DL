# Image Captioning with EfficientNet and LSTM

## Overview

This project implements an end-to-end image captioning system that automatically generates natural language descriptions for images. The model combines a convolutional neural network encoder for visual feature extraction with a recurrent neural network decoder for sequential text generation. The system is trained on the Flickr8k dataset and evaluated using the BLEU metric.

## Architecture

### High-Level Design

The architecture follows the encoder-decoder paradigm commonly used in sequence-to-sequence tasks:

1. **Encoder**: A frozen pretrained EfficientNet-B2 extracts high-level visual features from input images
2. **Projection Layer**: A trainable linear layer maps the visual features to the embedding space
3. **Decoder**: An LSTM-based recurrent network generates captions word by word, conditioned on the visual features

The key insight is treating image captioning as a translation task where we translate from the visual domain (images) to the linguistic domain (text sequences).

### Encoder: EfficientNet-B2

The choice of EfficientNet-B2 as the visual encoder is deliberate and based on several considerations:

**Why EfficientNet over VGG?**

Traditional approaches often use VGG networks, primarily because many existing tutorials and GitHub repositories default to it. However, VGG is outdated by modern standards. EfficientNet offers significant advantages:

- **Efficiency**: EfficientNet-B2 has substantially fewer parameters and lower computational requirements (FLOPs) compared to VGG
- **Performance**: Despite being more compact, EfficientNet achieves better accuracy on ImageNet classification tasks
- **Modern Architecture**: EfficientNet uses compound scaling, optimally balancing network depth, width, and resolution
- **Resource-Friendly**: Lower memory footprint and faster inference times make it more practical for deployment

**Implementation Details**

The encoder consists of:
- The complete EfficientNet-B2 feature extraction backbone (frozen)
- Global average pooling layer
- A trainable projection head that maps the 1408-dimensional feature vector to a 256-dimensional embedding space

Freezing the backbone parameters is crucial because:
1. The pretrained ImageNet features already capture robust visual representations
2. Training only the projection layer prevents overfitting on the relatively small Flickr8k dataset
3. Significantly reduces training time and computational requirements
4. Allows the model to leverage rich visual features learned from millions of images

### Decoder: LSTM Network

The decoder generates captions using a Long Short-Term Memory (LSTM) network, chosen for its ability to model long-range dependencies in sequential data.

**Architecture Components**

1. **Embedding Layer**: Maps word indices to dense 256-dimensional vectors
2. **LSTM Layer**: Single-layer LSTM with 512 hidden units processes the sequence
3. **Linear Output Layer**: Projects LSTM hidden states to vocabulary-sized logits
4. **Dropout**: 50% dropout applied to embeddings and between LSTM layers for regularization

**Caption Generation Process**

During training, the model uses teacher forcing:
1. The visual feature vector acts as the first input to the LSTM
2. Each subsequent LSTM input is the embedding of the previous ground truth word
3. The model predicts the next word at each time step
4. Loss is computed by comparing predictions against the ground truth caption

During inference, the model generates captions autoregressively:
1. Start with the visual features
2. Generate the first word using the special `<start>` token
3. Feed each predicted word back as input to generate the next word
4. Continue until the `<end>` token is generated or maximum length is reached

### Vocabulary Design

The vocabulary system is carefully designed to handle text processing:

**Special Tokens**
- `<pad>` (index 0): Padding token for batch processing
- `<start>` (index 1): Marks caption beginning
- `<end>` (index 2): Marks caption end
- `<unk>` (index 3): Represents out-of-vocabulary words

**Tokenization Strategy**

The tokenizer implements a simple but effective word-level approach:
1. Convert text to lowercase for case-insensitivity
2. Remove all punctuation and special characters
3. Normalize whitespace
4. Split on spaces to obtain word tokens

This word-level tokenization is appropriate for this task because:
- Caption vocabulary is relatively limited and well-defined
- Subword tokenization would add unnecessary complexity
- Training data is in English with standard vocabulary

**Frequency Threshold**

Only words appearing at least 5 times in the training set are included in the vocabulary. This threshold serves multiple purposes:
- Reduces vocabulary size and model complexity
- Filters out rare words and potential typos
- Prevents overfitting to infrequent terms
- Improves generalization by focusing on common descriptive language

Words below the threshold are mapped to `<unk>`, ensuring the model handles novel words gracefully during inference.

## Data Processing

### Dataset Structure

The Flickr8k dataset consists of 8,000 images, each annotated with 5 human-written captions. The `captions.txt` file contains image filenames paired with their descriptions.

### Image Transformations

**Training Augmentation**

The training pipeline applies several augmentations to improve model robustness and prevent overfitting:

1. **Resize to 260x260**: Standardizes input dimensions while slightly larger than the model's expected size to allow for cropping
2. **Random Horizontal Flip (50% probability)**: Doubles the effective dataset size and helps the model learn viewpoint invariance
3. **Color Jitter**: Randomly adjusts brightness (±20%), contrast (±20%), saturation (±20%), and hue (±10%)
   - Makes the model robust to different lighting conditions
   - Helps generalize across various camera settings and time of day
4. **Random Grayscale (2% probability)**: Occasionally converts images to grayscale
   - Forces the model to rely on shape and structure, not just color
   - Applied sparingly because color is often important for accurate descriptions
5. **Normalization**: Uses ImageNet mean and standard deviation
   - Required because EfficientNet was pretrained on normalized ImageNet data
   - Ensures input distribution matches what the encoder expects

**Validation Transformation**

Validation images receive minimal preprocessing:
1. Resize to 260x260
2. Convert to tensor
3. Normalize using ImageNet statistics

No augmentation is applied during validation to ensure consistent, reproducible evaluation.

**Why These Transformations?**

Each transformation serves a specific purpose:
- We avoid aggressive cropping because it might remove important objects that should be described
- Horizontal flipping is safe because most scene semantics are preserved when mirrored
- We use minimal grayscale conversion (2%) rather than excluding it entirely because a small amount prevents over-reliance on color
- Color jitter is moderate to maintain realism while improving robustness
- Affine transformations are avoided because rotating or scaling could distort objects in ways that change their description

### Data Splitting Strategy

The dataset is split using a stratified approach based on images rather than captions:

- **Training Set**: 80% of unique images
- **Validation Set**: 10% of unique images
- **Test Set**: 10% of unique images (implicit, remaining data)

**Critical Design Choice**: Each image has 5 captions, but we ensure all captions of a single image stay in the same split. This prevents data leakage where the model might memorize specific images seen during training and artificially inflate validation performance.

The vocabulary is built exclusively from training captions to prevent test-time information leakage.

### Batch Collation

Variable-length captions require custom batch processing. The `MyCollate` class handles this by:

1. Sorting batch samples by caption length (descending)
2. Stacking images into a tensor
3. Padding captions to the length of the longest caption in the batch
4. Tracking original caption lengths for packed sequence processing

This approach is essential for computational efficiency. PyTorch's packed sequences allow the LSTM to skip computations on padding tokens, significantly speeding up training.

## Training Configuration

### Hyperparameters

The model is trained with the following configuration:

- **Embedding Size**: 256 dimensions
  - Balances expressiveness with computational efficiency
  - Matches the projected visual feature dimension
- **Hidden Size**: 512 dimensions
  - Provides sufficient capacity for the LSTM to model complex caption patterns
  - Twice the embedding size gives the model room to learn rich representations
- **Number of Layers**: 1 LSTM layer
  - Single-layer architecture is sufficient for this task
  - Prevents overfitting on the moderate-sized dataset
- **Dropout**: 50%
  - High dropout rate acts as strong regularization
  - Applied to embeddings and the projection layer
- **Batch Size**: 64
  - Balances gradient stability with memory constraints
  - Provides good GPU utilization on typical hardware
- **Learning Rate**: 3e-4
  - Standard learning rate for Adam optimizer
  - Works well for this scale of model and dataset
- **Weight Decay**: 1e-5
  - L2 regularization prevents parameter growth
  - Light regularization since we already use dropout
- **Gradient Clipping**: Maximum norm of 5.0
  - Prevents exploding gradients in the LSTM
  - Essential for stable recurrent network training
- **Epochs**: 20
  - Sufficient for convergence on Flickr8k
  - Model typically converges within 15 epochs

### Loss Function

Cross-entropy loss is used with an important modification: the padding token is ignored when computing loss. This ensures the model only learns to predict actual words, not padding.

The loss is computed by:
1. Flattening the output tensor from (batch, sequence_length, vocab_size) to (batch × sequence_length, vocab_size)
2. Flattening the target captions similarly
3. Computing cross-entropy while ignoring positions with the padding token index

### Optimization Strategy

Adam optimizer is selected for its adaptive learning rate properties and good performance on this class of problems. The optimizer updates only the trainable parameters:
- Projection layer weights
- Embedding layer
- LSTM weights
- Output linear layer

The frozen EfficientNet backbone is excluded from optimization, significantly reducing the parameter count from millions to hundreds of thousands.

### Training Procedure

Training follows a standard supervised learning loop:

1. Load a batch of images and captions
2. Forward pass: encoder extracts features, decoder generates predictions
3. Compute cross-entropy loss between predictions and ground truth
4. Backward pass: compute gradients
5. Clip gradients to prevent instability
6. Update parameters using Adam
7. Repeat for all batches in an epoch

Checkpoints are saved every 5 epochs to enable recovery and model selection. The final model after 20 epochs is saved as `final_model.pth.tar`.

## Inference and Caption Generation

### Greedy Decoding

The basic inference method uses greedy decoding:
1. Extract visual features from the input image
2. Initialize LSTM state with the visual features
3. Start with the `<start>` token
4. At each step, predict the most likely next word
5. Feed the predicted word back as input
6. Continue until `<end>` is predicted or maximum length is reached

Greedy decoding is fast but can be suboptimal because it makes locally optimal choices without considering future consequences.

### Beam Search

An improved inference method implements beam search with width k=5:

**How Beam Search Works**

Instead of greedily selecting the single best word at each step, beam search maintains k=5 candidate sequences (beams):

1. Start with k=5 candidates initialized with `<start>`
2. At each step, for each candidate:
   - Generate predictions for all possible next words
   - Compute log probabilities for each extension
3. Keep the top k=5 sequences with highest cumulative log probability
4. Repeat until all beams end with `<end>` or maximum length is reached
5. Return the sequence with the highest overall score

**Why Beam Search?**

Beam search explores a broader portion of the sequence space, often finding better captions than greedy decoding. The log probability scoring naturally handles the multiplicative nature of sequence probabilities while avoiding numerical underflow.

The width k=5 balances exploration (considering multiple hypotheses) with computational cost. Larger beam widths provide diminishing returns while significantly increasing inference time.

## Evaluation

### BLEU Score

The model is evaluated using BLEU (Bilingual Evaluation Understudy), a metric originally designed for machine translation but widely adopted for image captioning.

**What BLEU Measures**

BLEU compares generated captions against reference captions by counting n-gram overlaps:
- Unigrams (1-grams): Individual word matches
- Bigrams (2-grams): Two-word phrase matches
- Trigrams (3-grams): Three-word phrase matches
- 4-grams: Four-word phrase matches

The score is computed as the geometric mean of n-gram precisions, weighted equally by default.

**Why BLEU for Image Captioning?**

Traditional metrics like accuracy or F1 score are inappropriate for caption generation because:
- Word order matters: "dog chasing cat" vs "cat chasing dog" have identical word sets but opposite meanings
- Multiple valid captions exist for the same image
- Semantic similarity is more important than exact word matches

BLEU addresses these issues by considering n-grams, which capture both word choice and ordering.

**Key BLEU Components**

1. **Modified Precision**: Counts how many n-grams from the generated caption appear in reference captions
   - Clipping prevents gaming the metric by repeating words
   - Maximum count in references caps the credit given for each n-gram

2. **Brevity Penalty**: Penalizes captions that are too short
   - Without this, very short captions would artificially achieve high precision
   - Exponential penalty based on the length ratio between candidate and reference

The implementation uses NLTK's corpus-level BLEU, which aggregates statistics across all test images before computing the score, providing a more stable metric than averaging individual image scores.

### Evaluation Process

To calculate validation BLEU:
1. Sample images from the validation set
2. Generate captions using beam search
3. Tokenize both generated and reference captions using the same tokenizer used during training
4. Compute corpus-level BLEU-4 score across all samples
5. Report the score as a measure of caption quality

The BLEU score provides an objective measure of how well the generated captions match human-written descriptions, both in content and structure.

## Implementation Details

### Reproducibility

Random seeds are set for PyTorch, NumPy, and CUDA to ensure reproducible results across runs:
```python
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
```

### Memory Optimization

Several techniques optimize memory usage:
- Frozen encoder parameters are not stored in optimizer state
- Packed sequences reduce unnecessary computation on padding tokens
- Gradient checkpointing could be added for larger models
- Efficient batch collation minimizes padding overhead

### Training on Google Colab

The notebooks are designed for Google Colab with GPU acceleration:
- Data is stored in Google Drive for persistence
- T4 GPU training takes approximately 3 hours for 20 epochs
- Checkpoints are saved to Google Drive to prevent data loss
- Persistent workers in DataLoader improve data loading efficiency when using GPUs

## Files Description

- `data_and_training.ipynb`: Complete training pipeline including data preprocessing, model definition, training loop, and checkpoint saving
- `inference.ipynb`: Model loading, caption generation, beam search implementation, and BLEU evaluation
- `caption_data/Images/`: Directory containing the 8,000 Flickr images
- `caption_data/captions.txt`: CSV file mapping image filenames to their 5 reference captions
- `final_model.pth.tar`: Trained model checkpoint containing state dict, optimizer state, configuration, and vocabulary
- `val_images.json`: List of validation image filenames for consistent evaluation

## Results and Observations

The model successfully learns to generate descriptive captions that typically include:
- Main objects in the scene (people, animals, objects)
- Actions being performed (running, jumping, playing)
- Contextual information (location, color, relative positions)
- Occasionally relational information (interactions between entities)


## Usage

### Training

1. Upload the caption dataset to Google Drive
2. Open `data_and_training.ipynb` in Google Colab
3. Mount Google Drive and navigate to the data directory
4. Run all cells to train the model
5. The trained model will be saved as `final_model.pth.tar`

### Inference

1. Ensure the trained model checkpoint is in Google Drive
2. Open `inference.ipynb` in Google Colab
3. Mount Google Drive
4. Update paths in the configuration if necessary
5. Run cells to load the model and generate captions
6. Use the demo section to test on validation images
7. Run BLEU evaluation to quantify performance

## Technical Requirements

- Python 3.7+
- PyTorch 1.9+
- torchvision
- PIL (Pillow)
- pandas
- numpy
- matplotlib
- tqdm
- nltk (for BLEU score calculation)

## Conclusion
This project demonstrates a complete image captioning pipeline from data processing through training to evaluation. The architecture combines proven techniques (CNN encoder, LSTM decoder) with modern, efficient components (EfficientNet) to achieve good performance on the Flickr8k dataset. The implementation prioritizes clarity and educational value while maintaining competitive results, making it suitable both for understanding image captioning fundamentals and as a starting point for more advanced experiments such as real starting mass surveilance company
This project demonstrates a complete image captioning pipeline from data processing through training to evaluation. The architecture combines proven techniques (CNN encoder, LSTM decoder) with modern, efficient components (EfficientNet) to achieve good performance on the Flickr8k dataset. The implementation prioritizes clarity and educational value while maintaining competitive results, making it suitable both for understanding image captioning fundamentals and as a starting point for more advanced experiments.
