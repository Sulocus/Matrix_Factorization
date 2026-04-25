
import torch
import sys

def analyze(path):
    print(f"Loading {path}...")
    try:
        data = torch.load(path)
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    print(f"Top-level Keys: {list(data.keys())}")
    
    alpha_values = data.get('alpha_values', [])
    metrics = data.get('metrics', {})
    
    if not alpha_values:
        print("No alpha_values found.")
        return
        
    print(f"Metrics Keys: {list(metrics.keys())}")

    print(f"{'Alpha':<10} {'Q_Y':<10} {'Q_Full':<10}")
    print("-" * 35)
    
    # Try to identify keys. Common keys: 'A.y', 'Q_Y', 'Q_Y_mean'
    # 'Q_Full_Tensor' might not be in runner output unless requested.
    
    print(f"{'Alpha':<10} {'Q_Y':<10} {'Q_Y_Std':<10} {'Q_Full':<10}")
    print("-" * 45)

    for i, alpha in enumerate(alpha_values):
        # Key in metrics is likely string representation of alpha
        # Runner usually uses generated keys provided by step_callback or aggregation
        # Based on keys printed above: '0.1', '0.300...', etc.
        
        # Fuzzy match key
        alpha_key = None
        for k in metrics.keys():
            try:
                if abs(float(k) - alpha) < 1e-4:
                    alpha_key = k
                    break
            except:
                pass
        
        q_y = 0.0
        q_std = 0.0
        q_full = "N/A"
        
        if alpha_key and alpha_key in metrics:
            m = metrics[alpha_key]
            # m is likely a dict { 'Q_Y_mean': ..., 'Q_Y_std': ..., ... }
            if isinstance(m, dict):
                q_y = m.get('Q_Y_mean', 0.0)
                q_std = m.get('Q_Y_std', 0.0)
                
                # Check for Q_Full_Tensor
                # Might be 'Q_Full_Tensor' key with list of values
                if 'Q_Full_Tensor' in m:
                    val = m['Q_Full_Tensor']
                    if isinstance(val, list) and val:
                        val = sum(val) / len(val)
                    if isinstance(val, (int, float)):
                         q_full = f"{val:.4f}"
            
        print(f"{alpha:<10.2f} {q_y:<10.4f} {q_std:<10.4f} {q_full:<10}")

if __name__ == "__main__":
    path = "runs/_latest/artifacts/results.pt"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    
    analyze(path)
