import numpy as np
import torch
from data_loader import MovieDataLoader
from models import BaselineModel, MovieDataset
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

def run_experiment():
    # 1. Prepare Data (Sampled to 10k for speed, set to None for full run)
    loader = MovieDataLoader()
    train_df, val_df, test_df, solution_df, le = loader.load_all_data(sample_size=10000)

    # 2. Baseline Model Execution
    print("\n[Phase 1] Training Baseline (TF-IDF + Logistic Regression)...")
    baseline = BaselineModel()
    baseline.train(train_df['DESCRIPTION'], train_df['LABEL'])
    
    # Internal baseline check
    b_train_acc = accuracy_score(train_df['LABEL'], baseline.predict(train_df['DESCRIPTION']))
    b_val_acc = accuracy_score(val_df['LABEL'], baseline.predict(val_df['DESCRIPTION']))

    # 3. Transformer Model Execution (DistilBERT)
    print("\n[Phase 2] Training Transformer (DistilBERT)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(le.classes_)).to(device)

    # Tokenize data
    def tokenize(texts):
        return tokenizer(texts.tolist(), truncation=True, padding=True, max_length=128)

    train_dataset = MovieDataset(tokenize(train_df['DESCRIPTION']), train_df['LABEL'].tolist())
    val_dataset = MovieDataset(tokenize(val_df['DESCRIPTION']), val_df['LABEL'].tolist())
    test_dataset = MovieDataset(tokenize(test_df['DESCRIPTION']))

    # use the following paramters, can fine tune, but keep it for now
    train_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    t_val_acc = trainer.evaluate()['eval_accuracy']

    # 4. Final Evaluation on Unseen Test Solution
    print("\n[Phase 3] Final Evaluation against Hidden Solution...")
    
    # Baseline Test Acc
    b_test_preds = baseline.predict(test_df['DESCRIPTION'])
    b_test_acc = accuracy_score(solution_df['LABEL'], b_test_preds)

    # Transformer Test Acc
    t_test_logits = trainer.predict(test_dataset).predictions
    t_test_preds = np.argmax(t_test_logits, axis=1)
    t_test_acc = accuracy_score(solution_df['LABEL'], t_test_preds)

    # Final Comparison Table
    print("\n" + "="*60)
    print(f"{'Metric':<20} | {'Baseline':<15} | {'DistilBERT':<15}")
    print("-" * 60)
    print(f"{'Training Acc':<20} | {b_train_acc:.4f:<15} | {'N/A':<15}")
    print(f"{'Internal Val Acc':<20} | {b_val_acc:.4f:<15} | {t_val_acc:.4f:<15}")
    print(f"{'Final Test Acc':<20} | {b_test_acc:.4f:<15} | {t_test_acc:.4f:<15}")
    print("="*60)

    print("\nDetailed Report (DistilBERT):")
    print(classification_report(solution_df['LABEL'], t_test_preds, target_names=le.classes_))

if __name__ == "__main__":
    run_experiment()