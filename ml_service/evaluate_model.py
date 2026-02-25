"""
Prediction Evaluation Module
=============================
Evaluates demand predictions using holdout validation.
For each product's 30-day sales_history:
  - Input:  days 1-23  (fed to model as padded 30-day window)
  - Target: days 24-30 (actual ground truth)
  - Output: 7-day forecast from model

Metrics:
  - MAE, RMSE, MAPE, R², Accuracy%, Direction Accuracy
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import ast
import pandas as pd


def evaluate_predictions(model, csv_path: str) -> dict:
    """
    Run holdout evaluation on a CSV dataset.

    Args:
        model: AdvancedMultimodalModel instance
        csv_path: Path to CSV with columns [description, sales_history]

    Returns:
        dict with aggregate metrics and per-product breakdown
    """
    df = pd.read_csv(csv_path)

    if 'sales_history' not in df.columns:
        return {"error": "CSV must have 'sales_history' column"}

    per_product = []
    all_actuals = []
    all_preds = []

    for idx, row in df.iterrows():
        # Parse history
        hist_raw = row['sales_history']
        try:
            hist = ast.literal_eval(hist_raw) if isinstance(hist_raw, str) else list(hist_raw)
        except Exception:
            continue

        hist = [float(v) for v in hist]

        # Need at least 30 days (23 input + 7 ground truth)
        if len(hist) < 30:
            continue

        # Holdout split
        input_window = hist[:23]  # days 1-23
        ground_truth = hist[23:30]  # days 24-30 (actual)

        desc = str(row.get('description', 'Product')) if 'description' in df.columns else 'Product'
        product_id = row.get('product_id', f"P{idx+1}") if 'product_id' in df.columns else f"P{idx+1}"

        # Get model prediction (model pads to 30 internally)
        result = model.predict_single(desc, input_window, image_url=None)

        if 'error' in result or 'daily_forecast' not in result:
            continue

        predicted = result['daily_forecast'][:7]

        if len(predicted) != 7 or len(ground_truth) != 7:
            continue

        actual = np.array(ground_truth)
        pred = np.array(predicted, dtype=float)

        # Per-product metrics
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))

        # MAPE (guard against zero actuals)
        nonzero_mask = actual > 0
        if nonzero_mask.sum() > 0:
            mape = np.mean(np.abs((actual[nonzero_mask] - pred[nonzero_mask]) / actual[nonzero_mask])) * 100
        else:
            mape = 0.0

        accuracy_pct = max(0, 100 - mape)

        # R² (needs at least 2 samples — we have 7)
        r2 = r2_score(actual, pred) if len(actual) >= 2 else 0.0

        # Direction accuracy: does predicted change match actual change?
        actual_dirs = np.sign(np.diff(actual))
        pred_dirs = np.sign(np.diff(pred))
        direction_matches = np.sum(actual_dirs == pred_dirs)
        direction_accuracy = (direction_matches / len(actual_dirs)) * 100 if len(actual_dirs) > 0 else 0.0

        product_result = {
            "product_id": str(product_id),
            "description": desc[:50],
            "actual_7day": [round(v, 1) for v in actual.tolist()],
            "predicted_7day": [round(v, 1) for v in pred.tolist()],
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2),
            "accuracy_pct": round(accuracy_pct, 2),
            "r2_score": round(r2, 4),
            "direction_accuracy": round(direction_accuracy, 2)
        }

        per_product.append(product_result)
        all_actuals.extend(actual.tolist())
        all_preds.extend(pred.tolist())

    if len(all_actuals) == 0:
        return {"error": "No valid products found for evaluation (need 30+ days of history)"}

    # Aggregate metrics
    all_actual = np.array(all_actuals)
    all_pred = np.array(all_preds)

    agg_mae = mean_absolute_error(all_actual, all_pred)
    agg_rmse = np.sqrt(mean_squared_error(all_actual, all_pred))

    nonzero = all_actual > 0
    agg_mape = np.mean(np.abs((all_actual[nonzero] - all_pred[nonzero]) / all_actual[nonzero])) * 100 if nonzero.sum() > 0 else 0.0
    agg_accuracy = max(0, 100 - agg_mape)
    agg_r2 = r2_score(all_actual, all_pred)

    # Aggregate direction accuracy
    all_dir_actual = np.sign(np.diff(all_actual))
    all_dir_pred = np.sign(np.diff(all_pred))
    agg_direction = (np.sum(all_dir_actual == all_dir_pred) / len(all_dir_actual)) * 100 if len(all_dir_actual) > 0 else 0.0

    summary = {
        "total_products_evaluated": len(per_product),
        "total_datapoints": len(all_actuals),
        "mae": round(agg_mae, 2),
        "rmse": round(agg_rmse, 2),
        "mape": round(agg_mape, 2),
        "accuracy_pct": round(agg_accuracy, 2),
        "r2_score": round(agg_r2, 4),
        "direction_accuracy": round(agg_direction, 2)
    }

    return {
        "success": True,
        "summary": summary,
        "per_product": per_product
    }


def print_evaluation_report(results: dict):
    """Pretty-print the evaluation results to console."""
    if 'error' in results:
        print(f"\n❌ Evaluation Error: {results['error']}")
        return

    s = results['summary']
    print("\n" + "=" * 60)
    print("   DEMAND PREDICTION — EVALUATION REPORT")
    print("=" * 60)
    print(f"  Products Evaluated : {s['total_products_evaluated']}")
    print(f"  Total Datapoints   : {s['total_datapoints']}")
    print("-" * 60)
    print(f"  MAE  (Mean Abs Error)    : {s['mae']:.2f} units")
    print(f"  RMSE (Root Mean Sq Err)  : {s['rmse']:.2f} units")
    print(f"  MAPE (Mean Abs % Error)  : {s['mape']:.2f}%")
    print(f"  Accuracy (100 - MAPE)    : {s['accuracy_pct']:.2f}%")
    print(f"  R² Score                 : {s['r2_score']:.4f}")
    print(f"  Direction Accuracy       : {s['direction_accuracy']:.2f}%")
    print("=" * 60)

    print(f"\n{'Product':<35} {'MAE':>8} {'MAPE%':>8} {'Acc%':>8} {'R²':>8} {'Dir%':>8}")
    print("-" * 83)
    for p in results['per_product']:
        print(f"  {p['description'][:33]:<33} {p['mae']:>8.2f} {p['mape']:>8.2f} {p['accuracy_pct']:>8.2f} {p['r2_score']:>8.4f} {p['direction_accuracy']:>8.2f}")
    print()
