# AG News Text Classification using TensorFlow and TensorFlow Hub

This project demonstrates how to build a binary text classification model using the AG News Subset dataset with pre-trained text embeddings from TensorFlow Hub.

## 🧠 Model Overview

- **Embedding Layer**: Uses Swivel word embeddings from TensorFlow Hub.
- **Architecture**:
  - Text embedding
  - Dense Layer with ReLU activation
  - Output Dense Layer with Sigmoid for binary classification
- **Loss**: Binary Crossentropy
- **Optimizer**: Adam

## 📦 Dependencies

- Python ≥ 3.8
- TensorFlow ≥ 2.x
- TensorFlow Hub
- TensorFlow Datasets
- `tf_keras` (custom alias for `tensorflow.keras`)

Install all dependencies via pip:

```bash
pip install tensorflow tensorflow-hub tensorflow-datasets

📚 Dataset
We use the ag_news_subset from TensorFlow Datasets which includes four classes of news articles. However, this script assumes binary classification, so you may want to adjust the labels accordingly.

Training Split: 60%

Validation Split: 40% of train

Test Split: Separate test set

🚀 How to Run
python ag_news_classifier.py

📈 Results
After training, the script prints:

Test loss

Test accuracy

Modify the code to visualize history.history using matplotlib if needed.

💡 Note
If you're seeing label mismatch issues (because AG News is multi-class), update the last layer and loss to support multi-class classification (SparseCategoricalCrossentropy and softmax) instead of binary.

📜 License
MIT License

✍️ Author
Siddharth Parmar
