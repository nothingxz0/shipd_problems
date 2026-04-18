import time
import solution

def levenshtein(a, b):
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if ai == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = tmp
    return dp[m]

def ned(pred, true):
    if not pred and not true:
        return 1.0
    d = levenshtein(pred, true)
    return 1.0 - d / max(len(pred), len(true))

def main():
    print("Loading data...")
    data_dir = solution.resolve_data_dir()
    all_rows = solution.read_csv_rows(data_dir / "train.csv")
    
    # 85/15 Split (6800 train, 1200 validation)
    train_rows = all_rows[:6800]
    val_rows = all_rows[6800:]
    
    print(f"Training on {len(train_rows)} rows, validating on {len(val_rows)} rows...")
    t0 = time.time()
    
    # Run your exact solution pipeline
    predictions = solution.build_predictions(train_rows, val_rows)
    
    # Calculate scores
    scores = []
    for row, pred in zip(val_rows, predictions):
        true_seq = tuple(solution.parse_target_sequence(row.get("target_sequence", "[]")))
        scores.append(ned(pred, true_seq))
        
    final_score = sum(scores) / len(scores)
    
    print(f"\nCompleted in {time.time() - t0:.2f}s")
    print("========================================")
    print(f"🏆 Local Validation NED Score: {final_score:.5f}")
    print("========================================")

if __name__ == "__main__":
    main()