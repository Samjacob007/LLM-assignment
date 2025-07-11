
# BERT Fine-Tuning for News Topic Classification

This project fine-tunes a pre-trained BERT model (`bert-base-uncased`) for multi-class news topic classification using the AG News dataset. The model is trained to classify short news texts into four categories: **World**, **Sports**, **Business**, and **Sci/Tech**.

## 📚 Dataset

- **Source**: [wangrongsheng/ag_news](https://huggingface.co/datasets/wangrongsheng/ag_news) via Hugging Face Datasets
- **Classes**:
  - World
  - Sports
  - Business
  - Sci/Tech
- **Subset used**: 10,000 training samples, 2,000 test samples (to reduce compute time)

## 🧠 Model

- **Pre-trained model**: `bert-base-uncased` from Hugging Face Transformers
- **Architecture**: 12-layer BERT encoder with ~110M parameters
- **Modified for**: 4-class sequence classification

## 🛠️ Training Configuration

- Optimizer: AdamW
- Learning Rate: `2e-5`
- Epochs: `5`
- Batch Size: `8`
- Weight Decay: `0.01`
- Evaluation Strategy: `per epoch`
- Metrics: Accuracy, Precision, Recall, F1 (weighted average)

## 📈 Results

### Epoch-wise Performance

| Epoch | Train Loss | Val. Loss | Accuracy | Precision | Recall | F1 Score |
|-------|------------|-----------|----------|-----------|--------|----------|
| 1     | 0.3311     | 0.3537    | 90.45%   | 90.65%    | 90.45% | 90.42%   |
| 2     | 0.2289     | 0.2989    | 92.65%   | 92.64%    | 92.65% | 92.62%   |
| 3     | 0.1330     | 0.3790    | 92.75%   | 92.80%    | 92.75% | 92.74%   |
| 4     | 0.0740     | 0.3988    | 93.35%   | 93.35%    | 93.35% | 93.34%   |
| 5     | 0.0374     | 0.4571    | 92.60%   | 92.59%    | 92.60% | 92.58%   |

### Final Classification Report

```
              precision    recall  f1-score   support

       World       0.92      0.96      0.94       511
      Sports       0.98      0.97      0.98       526
    Business       0.88      0.87      0.87       449
    Sci/Tech       0.92      0.90      0.91       514

    accuracy                           0.93      2000
   macro avg       0.92      0.92      0.92      2000
weighted avg       0.93      0.93      0.93      2000
```

## 🔍 Example Predictions

```python
predict("NASA launches new rocket to explore Mars.")  # Output: Sci/Tech
predict("The stock market saw significant drops today.")  # Output: Business
```

## 📊 Visualizations

- Training vs Validation Loss
- Validation Accuracy per Epoch

These plots help assess overfitting and generalization.

## 💾 How to Run

```bash
pip install transformers datasets evaluate scikit-learn
```

Then run the Python script in a notebook or as a standalone script. Training will fine-tune the model and generate predictions, plots, and metrics.

## 🧪 Requirements

- Python 3.7+
- Transformers >= 4.0
- Datasets >= 2.0
- Evaluate
- PyTorch
- scikit-learn
- Matplotlib

## 📂 Output

- Fine-tuned model and tokenizer saved locally
- Metrics printed and plots generated
- Classification report for test set

## 📎 License

This project is for academic and research purposes under a fair use clause.
