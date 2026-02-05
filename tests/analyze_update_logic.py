
import torch
import math

def run_simulation():
    print("=== Tensor Update Logic Simulation ===")
    
    # 1. Setup Toy Problem
    N = 1000
    M = 1
    S = 1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    
    # Case: Signal is weak (Alpha small, Tau large)
    # True Signal
    true_x = torch.randn(N, M, device=device)
    
    # Input to denoiser (R) has low SNR
    # Precision tau (low precision -> large noise)
    tau_val = 0.5 
    noise_std = math.sqrt(1.0 / tau_val)
    
    # The "Effective Field" R coming from the graph
    # R = True_X + Noise
    R_eff = true_x + torch.randn_like(true_x) * noise_std
    
    # Students Initial Guess (Random, close to 0)
    x_old = torch.randn(N, M, device=device) * 0.1
    
    # Pre-calculate common terms
    tau_d = torch.full_like(true_x, tau_val)
    # r_d in code roughly corresponds to: (R_eff - x_old) * tau
    # (The residual scaled by precision)
    r_d_sim = (R_eff - x_old) * tau_val
    
    # Posterior Variance (Same for both)
    # P(x) ~ N(0, 1) -> Prior Precision = 1
    # Likelihood Precision = tau
    # Post Precision = 1 + tau -> Post Var = 1/(1+tau)
    new_var = 1.0 / (1.0 + tau_d)

    print(f"True X Mean: {true_x.mean():.4f}, Std: {true_x.std():.4f}")
    print(f"Old X Mean:  {x_old.mean():.4f}, Std: {x_old.std():.4f}")
    print(f"R_eff Mean:  {R_eff.mean():.4f}, Std: {R_eff.std():.4f}")
    
    # --- Logic A: Current (Faulty) ---
    # new = old + new_var * r
    # new = old + (1/(1+tau)) * (R - old)*tau
    # new = old + (tau/(1+tau)) * (R - old)
    # new = (old*(1+tau) + tau*R - tau*old) / (1+tau)
    # new = (old + tau*R) / (1+tau)
    # ^^^ This effectively weights 'old' with 1, and 'R' with tau.
    # But 'old' comes from... nowhere? It's just the previous iteration.
    # If 'old' is huge, 'new' stays huge. Weight Decay is weak.
    
    x_new_wrong = x_old + new_var * r_d_sim
    
    # --- Logic B: Correct (Bayesian) ---
    # new = (tau * old + r) * new_var  <-- Wait, the formula in code was "tau*old + r" ?
    # Legacy: new_factor = new_var * (tau_d * factors[d] + r_d)
    # Here 'factors[d]' is treated as... the prior mean? 
    # NO. In standard AMP, the denoiser input is R, and we estimate X given R.
    # The 'factors[d]' in the formula 'tau*factors[d] + r' is confusing. 
    # Usually r_d IS the term $\sum ...$ 
    
    # Let's perform the algebra on the Legacy Code:
    # new = (1/tau) * (tau*old + r) = old + r/tau
    # This cancels out 'tau'. 
    
    # Let's look at the GAMP Expectation step properly:
    # E[x | R, tau] = (R * tau + PriorMean * PriorPrec) / (tau + PriorPrec)
    # If PriorMean=0, PriorPrec=1
    # E[x] = (R * tau) / (tau + 1)
    
    # Now let's calculate what the code produces.
    # r_d_sim = (R - old) * tau
    
    # Correct Output should be: (R * tau) / (tau + 1)
    # Let's see which formula gives this.
    
    # Try Correct Formula:
    # new_factor = new_var * (tau * old + r)
    # = (1/(1+tau)) * (tau*old + (R-old)*tau)
    # = (1/(1+tau)) * (tau*old + tau*R - tau*old)
    # = (tau*R) / (1+tau)
    # BINGO! This produces exactly the Bayesian Posterior Mean.
    
    x_new_correct = new_var * (tau_d * x_old + r_d_sim)
    
    # Try Wrong Formula:
    # new_factor = old + new_var * r
    # = old + (1/(1+tau)) * (R-old)*tau
    # = old + (tau/(1+tau)) * (R-old)
    # = (old*(1+tau) + tau*R - tau*old) / (1+tau)
    # = (old + tau*R) / (1+tau)
    # ERROR FOUND: The 'old' term has coefficient 1 instead of 0!
    # It adds 'old / (1+tau)' to the result.
    # This is "Memory" or "Momentum" that shouldn't be there.
    # It drags the previous iteration's value along.
    
    # Verify Correlations
    def calc_q(x, truth):
        return (x * truth).mean().item()
    
    q_old = calc_q(x_old, true_x)
    q_wrong = calc_q(x_new_wrong, true_x)
    q_correct = calc_q(x_new_correct, true_x)
    
    print(f"\n--- Correlations (Q) with Truth ---")
    print(f"Q_Old:     {q_old:.4f}")
    print(f"Q_Wrong:   {q_wrong:.4f}")
    print(f"Q_Correct: {q_correct:.4f}")
    
    print("\n--- Interpretation ---")
    if abs(q_wrong) > abs(q_correct):
         print("!! Wrong update yields HIGHER correlation.")
    else:
         print("Correct update yields higher correlation.")

if __name__ == "__main__":
    run_simulation()
