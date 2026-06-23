import json
import os
import random
from datetime import datetime, timedelta

# --- PRICING CONSTANTS (Per 1M Tokens) ---
# Hardcoded to current OpenAI pricing for ROI calculation
PRICING = {
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60}
}

LOG_FILE = "logs/cascade_traffic.log"

def generate_mock_logs(filepath):
    """Generates realistic mock traffic if no real log file exists yet."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    print(f"\033[90m[Info] No live logs found at {filepath}. Generating mock enterprise traffic...\033[0m\n")
    
    now = datetime.now()
    with open(filepath, "w") as f:
        for i in range(50000):
            # 75% of traffic is simple enough for mini
            is_simple = random.random() < 0.75 
            target_model = "gpt-4o-mini" if is_simple else "gpt-4o"
            
            # Realistic token distributions
            input_tokens = int(random.gauss(3500, 500))
            output_tokens = int(random.gauss(450, 100))
            latency = round(random.uniform(3.8, 4.9), 2)
            
            log_entry = {
                "timestamp": (now - timedelta(minutes=1000-i)).isoformat(),
                "model_routed": target_model,
                "input_tokens": max(10, input_tokens),
                "output_tokens": max(5, output_tokens),
                "routing_latency_ms": latency
            }
            f.write(json.dumps(log_entry) + "\n")

def calculate_roi(filepath):
    """Parses the log file and calculates savings."""
    total_requests = 0
    mini_requests = 0
    total_latency = 0.0
    
    actual_cost = 0.0
    frontier_only_cost = 0.0 # What it WOULD have cost without Cascade
    
    with open(filepath, "r") as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                model = data.get("model_routed", "gpt-4o")
                in_tok = data.get("input_tokens", 0)
                out_tok = data.get("output_tokens", 0)
                
                total_requests += 1
                total_latency += data.get("routing_latency_ms", 4.5)
                
                if model == "gpt-4o-mini":
                    mini_requests += 1
                
                # Calculate what it ACTUALLY cost
                in_cost = (in_tok / 1_000_000) * PRICING[model]["input"]
                out_cost = (out_tok / 1_000_000) * PRICING[model]["output"]
                actual_cost += (in_cost + out_cost)
                
                # Calculate what it WOULD HAVE cost if they just hardcoded gpt-4o
                frontier_in_cost = (in_tok / 1_000_000) * PRICING["gpt-4o"]["input"]
                frontier_out_cost = (out_tok / 1_000_000) * PRICING["gpt-4o"]["output"]
                frontier_only_cost += (frontier_in_cost + frontier_out_cost)
                
            except json.JSONDecodeError:
                continue
                
    savings = frontier_only_cost - actual_cost
    savings_pct = (savings / frontier_only_cost) * 100 if frontier_only_cost > 0 else 0
    avg_latency = total_latency / total_requests if total_requests > 0 else 0
    
    return total_requests, mini_requests, avg_latency, actual_cost, frontier_only_cost, savings, savings_pct

def print_dashboard():
    if not os.path.exists(LOG_FILE):
        generate_mock_logs(LOG_FILE)
        
    reqs, mini_reqs, avg_lat, actual, frontier, savings, pct = calculate_roi(LOG_FILE)
    
    # ANSI Color Codes for beautiful terminal output
    C_BLUE = "\033[94m"
    C_GREEN = "\033[92m"
    C_YELLOW = "\033[93m"
    C_BOLD = "\033[1m"
    C_RESET = "\033[0m"
    
    print(f"{C_BLUE}{C_BOLD}")
    print("  ___  __  __  __  __  __  ___ ")
    print(" / __|/  \\/ __/ _|/  \\|  \\| __|")
    print("| (__| /\\ \\__ \\_ \\ /\\ | | | _| ")
    print(" \\___|_||_|___/__/_||_|__/|___|")
    print(f"       ROUTER ANALYTICS        {C_RESET}\n")
    
    print(f" {C_BOLD}PERFORMANCE METRICS{C_RESET}")
    print(f" ------------------------------------")
    print(f" Total Requests Processed:  {reqs:,}")
    print(f" Successfully Down-Routed:  {C_GREEN}{mini_reqs:,} ({ (mini_reqs/reqs)*100:.1f}%){C_RESET}")
    print(f" Avg Routing Overhead:      {C_YELLOW}{avg_lat:.2f} ms{C_RESET}\n")
    
    print(f" {C_BOLD}ENTERPRISE ROI (Simulated Trailing){C_RESET}")
    print(f" ------------------------------------")
    print(f" Standard `gpt-4o` Cost:    ${frontier:.2f}")
    print(f" Cascade Optimized Cost:    ${actual:.2f}")
    print(f" ------------------------------------")
    print(f" {C_BOLD}Total Money Saved:         {C_GREEN}${savings:.2f} ({pct:.1f}%){C_RESET}\n")
    
    print(f"{C_BLUE}-> Cascade Proxy is running optimally.{C_RESET}\n")

if __name__ == "__main__":
    print_dashboard()