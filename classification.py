import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

def calculate_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    return accuracy, precision, recall

def calculate_exact_match_ratio(true_labels, predicted_labels):
    exact_matches = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    return exact_matches / len(true_labels)

def read_and_evaluate(file_path):
    df = pd.read_csv(file_path)
    
    main_role_true = df['main_role']
    max_main_role = df['max_main_role']
    avg_main_role = df['avg_main_role']
    
    fg_grained_roles_true = df['fine_grained_roles']
    max_fg_role = df['max_fg_role']
    avg_fg_role = df['avg_fg_role']
    
    max_main_role_metrics = calculate_metrics(main_role_true, max_main_role)
    avg_main_role_metrics = calculate_metrics(main_role_true, avg_main_role)

    max_fg_role_exact_match_ratio = calculate_exact_match_ratio(fg_grained_roles_true, max_fg_role)
    avg_fg_role_exact_match_ratio = calculate_exact_match_ratio(fg_grained_roles_true, avg_fg_role)
    
    return max_main_role_metrics, avg_main_role_metrics, max_fg_role_exact_match_ratio, avg_fg_role_exact_match_ratio

def save_results(file_path, results):
    with open(file_path, 'w') as f:
        f.write("File, Max_Acc, Max_Prc, Max_rec, Max_Exact_match_ratio, Avg_Acc, Avg_Prc, Avg_rec, Avg_Exact_match_ratio\n")
        for file_label, metrics in results.items():
            #f.write(f"{file_label},{metrics['accuracy']},{metrics['precision']},{metrics['recall']},{metrics['exact_match_ratio']}\n")
            f.write(
                f"{file_label},{metrics['max_accuracy']},{metrics['max_precision']},{metrics['max_recall']},{metrics['max_exact_match_ratio']},{metrics['avg_accuracy']},{metrics['avg_precision']},{metrics['avg_recall']},{metrics['avg_exact_match_ratio']}\n")

def main():
    results = {}
    files = os.listdir("./results")  # Add your file paths here
    files = [f"./results/{file}" for file in files]
    
    for file in files:
        max_main_role_metrics, avg_main_role_metrics, max_fg_role_exact_match_ratio, avg_fg_role_exact_match_ratio = read_and_evaluate(file)
        results[file] = {
            'max_accuracy': max_main_role_metrics[0],
            'max_precision': max_main_role_metrics[1],
            'max_recall': max_main_role_metrics[2],

            'avg_accuracy': avg_main_role_metrics[0],
            'avg_precision': avg_main_role_metrics[1],
            'avg_recall': avg_main_role_metrics[2],

            'max_exact_match_ratio': max_fg_role_exact_match_ratio,

            'avg_exact_match_ratio': avg_fg_role_exact_match_ratio
        }
    
    save_results('evaluation_results.csv', results)

if __name__ == "__main__":
    main()
