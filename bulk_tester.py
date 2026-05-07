import os
import json
from orchestrator import AegisOrchestrator

def run_stress_test(parent_dir: str):
    orchestrator = AegisOrchestrator()
    summary_report = []
    
    # Get all subdirectories
    subdirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    
    print(f"==========================================")
    print(f"   PROJECT AEGIS: BULK STRESS TEST        ")
    print(f"   Target: {len(subdirs)} Datasets        ")
    print(f"==========================================\n")
    
    for i, folder in enumerate(subdirs, 1):
        folder_path = os.path.join(parent_dir, folder)
        print(f"[{i}/{len(subdirs)}] Processing: {folder}...")
        
        try:
            result = orchestrator.run(folder_path)
            
            metrics = result.get("metrics", {})
            summary_report.append({
                "folder": folder,
                "rules": len(result.get("golden_rules", [])),
                "accuracy": metrics.get("accuracy", 0),
                "f1": metrics.get("f1", 0),
                "status": "SUCCESS"
            })
            print(f"  - Rules Found: {len(result.get('golden_rules', []))}")
            print(f"  - Accuracy: {metrics.get('accuracy', 0):.4f}")
            
        except Exception as e:
            print(f"  - FAILED: {str(e)}")
            summary_report.append({
                "folder": folder,
                "status": f"FAILED: {str(e)}"
            })
            
    # Final Summary Table
    print("\n" + "="*50)
    print(f"{'FOLDER':<15} | {'RULES':<5} | {'ACCURACY':<8} | {'STATUS'}")
    print("-" * 50)
    for entry in summary_report:
        acc_str = f"{entry.get('accuracy', 0):.4f}" if 'accuracy' in entry else "N/A"
        rules_str = str(entry.get('rules', 0))
        print(f"{entry['folder']:<15} | {rules_str:<5} | {acc_str:<8} | {entry['status']}")
    print("="*50)

    # Save Results
    with open("STRESS_TEST_RESULTS.json", "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"\nDetailed results saved to STRESS_TEST_RESULTS.json")

if __name__ == "__main__":
    run_stress_test("stress_test")
