import numpy as np
import torch
from data_loader import MovieDataLoader
from models import BaselineModel, MovieDataset
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

def run_experiment():
    # 1. SCALE UP DATA
    loader = MovieDataLoader()
    # Use all samples about 30mins training time for distilbert
    train_df, val_df, test_df, solution_df, le = loader.load_all_data(sample_size=None)

    # 2. Phase 1: Baseline Model
    print("\n" + "="*50)
    print("PHASE 1: TRAINING BASELINE (TF-IDF + LOGISTIC REGRESSION)")
    print("="*50)
    baseline = BaselineModel()
    baseline.train(train_df['DESCRIPTION'], train_df['LABEL'])
    
    b_val_preds = baseline.predict(val_df['DESCRIPTION'])
    b_val_acc = accuracy_score(val_df['LABEL'], b_val_preds)
    
    print(f"Baseline Internal Validation Accuracy: {b_val_acc:.4f}")
    
    # NEW: Baseline Validation Classification Report
    print("\n--- Baseline: Internal Validation Classification Report ---")
    print(classification_report(val_df['LABEL'], b_val_preds, target_names=le.classes_, zero_division=0))

    # 3. Phase 2: Optimized Transformer (DistilBERT)
    print("\n" + "="*50)
    print("PHASE 2: TRAINING OPTIMIZED DISTILBERT")
    print("="*50)
    
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(le.classes_)
    ).to(device)

    # Tokenization: max_length=128 for efficiency on local hardware
    def tokenize(texts):
        return tokenizer(texts.tolist(), truncation=True, padding=True, max_length=128)

    print("Tokenizing datasets...")
    train_dataset = MovieDataset(tokenize(train_df['DESCRIPTION']), train_df['LABEL'].tolist())
    val_dataset = MovieDataset(tokenize(val_df['DESCRIPTION']), val_df['LABEL'].tolist())
    test_dataset = MovieDataset(tokenize(test_df['DESCRIPTION']))

    train_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=16,
        eval_strategy="epoch", 
        save_strategy="epoch",
        report_to="none",
        logging_steps=50,
        metric_for_best_model="accuracy",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    print("Starting Fine-tuning...")
    trainer.train()

    # DistilBERT Validation Classification Report
    print("\n--- DistilBERT: Internal Validation Classification Report ---")
    val_logits = trainer.predict(val_dataset).predictions
    val_preds = np.argmax(val_logits, axis=1)
    print(classification_report(val_df['LABEL'], val_preds, target_names=le.classes_, zero_division=0))

    # 4. Phase 3: Final Evaluation (Test Phase)
    print("\n" + "="*50)
    print("PHASE 3: FINAL EVALUATION AGAINST HIDDEN SOLUTION (TEST PHASE)")
    print("="*50)
    
    # Baseline Test Predictions
    b_test_preds = baseline.predict(test_df['DESCRIPTION'])
    b_test_acc = accuracy_score(solution_df['LABEL'], b_test_preds)

    print("\n--- Baseline: Final Test Solution Classification Report ---")
    print(classification_report(solution_df['LABEL'], b_test_preds, target_names=le.classes_, zero_division=0))
    
    # Transformer Test Predictions
    t_test_logits = trainer.predict(test_dataset).predictions
    t_test_preds = np.argmax(t_test_logits, axis=1)
    t_test_acc = accuracy_score(solution_df['LABEL'], t_test_preds)

    # DistilBERT Test Classification Report
    print("\n--- DistilBERT: Final Test Solution Classification Report ---")
    print(classification_report(solution_df['LABEL'], t_test_preds, target_names=le.classes_, zero_division=0))

    # Final Comparison Summary
    print("\n" + "="*60)
    print(f"{'Final Performance Summary':^60}")
    print("-" * 60)
    print(f"{'Metric':<20} | {'Baseline':<15} | {'DistilBERT':<15}")
    print("-" * 60)
    
    # Get the best evaluation accuracy from the trainer state
    t_val_metrics = trainer.evaluate()
    
    # Corrected f-string format: {value:<width.precisionf}
    print(f"{'Internal Val Acc':<25} | {b_val_acc:<15.4f} | {t_val_metrics['eval_accuracy']:<15.4f}")
    print(f"{'Final Test Acc':<25} | {b_test_acc:<15.4f} | {t_test_acc:<15.4f}")
    print("="*65)

if __name__ == "__main__":
    run_experiment()