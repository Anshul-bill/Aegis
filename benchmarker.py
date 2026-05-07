import evaluate
import numpy as np
from typing import List, Dict, Any

class AegisBenchmarker:
    def __init__(self):
        # Loading SOTA metrics from Hugging Face evaluate
        self.accuracy_metric = evaluate.load("accuracy")
        self.precision_metric = evaluate.load("precision")
        self.recall_metric = evaluate.load("recall")
        self.f1_metric = evaluate.load("f1")
        self.rouge_metric = evaluate.load("rouge")
        self.cer_metric = evaluate.load("cer")
        self.wer_metric = evaluate.load("wer")

    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) for two boxes [ymin, xmin, ymax, xmax].
        """
        if not box1 or not box2 or len(box1) < 4 or len(box2) < 4:
            return 0.0
            
        y1_min, x1_min, y1_max, x1_max = box1
        y2_min, x2_min, y2_max, x2_max = box2

        # Calculate intersection
        inter_ymin = max(y1_min, y2_min)
        inter_xmin = max(x1_min, x2_min)
        inter_ymax = min(y1_max, y2_max)
        inter_xmax = min(x1_max, x2_max)

        inter_w = max(0, inter_xmax - inter_xmin)
        inter_h = max(0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h

        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def calculate_metrics(self, predictions: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]):
        """
        Calculates Accuracy, Precision, Recall, F1, ROUGE, CER, WER, and IoU.
        """
        if not predictions or not ground_truth:
            return {
                "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "rouge1": 0.0, "rougeL": 0.0, "cer": 1.0, "wer": 1.0, "iou": 0.0
            }

        # 1. Classification Metrics (Status)
        pred_status = [1 if p['status'] == "COMPLIANT" else 0 for p in predictions]
        gt_status = [1 if g['status'] == "COMPLIANT" else 0 for g in ground_truth]
        
        # Clip indices to match length (safety)
        min_len = min(len(pred_status), len(gt_status))
        pred_status = pred_status[:min_len]
        gt_status = gt_status[:min_len]

        acc = self.accuracy_metric.compute(predictions=pred_status, references=gt_status)['accuracy']
        prec = self.precision_metric.compute(predictions=pred_status, references=gt_status)['precision']
        rec = self.recall_metric.compute(predictions=pred_status, references=gt_status)['recall']
        f1 = self.f1_metric.compute(predictions=pred_status, references=gt_status)['f1']
        
        # 2. Text Quality Metrics (Extracted Value)
        pred_values = [str(p.get('extracted_value', "")) for p in predictions][:min_len]
        gt_values = [str(g.get('expected_value', "")) for g in ground_truth][:min_len]
        
        # Handle empty strings for CER/WER
        valid_pairs = [(p, g) for p, g in zip(pred_values, gt_values) if g.strip()]
        if valid_pairs:
            p_v, g_v = zip(*valid_pairs)
            cer = self.cer_metric.compute(predictions=p_v, references=g_v)
            wer = self.wer_metric.compute(predictions=p_v, references=g_v)
        else:
            cer, wer = 1.0, 1.0

        rouge = self.rouge_metric.compute(predictions=pred_values, references=gt_values)
        
        # 3. Localization Metrics (IoU)
        ious = []
        for i in range(min_len):
            p_box = predictions[i].get('evidence_bbox', [0,0,0,0])
            g_box = ground_truth[i].get('expected_bbox', [0,0,0,0])
            ious.append(self.calculate_iou(p_box, g_box))
        
        mean_iou = np.mean(ious) if ious else 0.0
        
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "rouge1": rouge['rouge1'],
            "rougeL": rouge['rougeL'],
            "cer": cer,
            "wer": wer,
            "iou": mean_iou
        }

if __name__ == "__main__":
    benchmarker = AegisBenchmarker()
    
    # Mock data for demonstration
    mock_predictions = [
        {"status": "COMPLIANT", "extracted_value": "Rs. 10,000/-"},
        {"status": "COMPLIANT", "extracted_value": "Total Rs. 590.00"},
        {"status": "NON_COMPLIANT", "extracted_value": "12.05.2026"} # Partial match/error
    ]
    
    mock_ground_truth = [
        {"status": "COMPLIANT", "expected_value": "Rs. 10,000/- Per Cluster"},
        {"status": "COMPLIANT", "expected_value": "Total Rs. 590.00"},
        {"status": "COMPLIANT", "expected_value": "14.05.2026"} # Mis-extracted date
    ]
    
    results = benchmarker.calculate_metrics(mock_predictions, mock_ground_truth)
    print("\n--- Pipeline Performance Metrics ---")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"ROUGE-1 (Extraction Similarity): {results['rouge1']:.4f}")
