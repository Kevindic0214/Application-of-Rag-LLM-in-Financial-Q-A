import json

def calculate_precision_at_1(predicted_answers, ground_truth_answers, qid_range):
    """Calculate Precision@1 for the specified QID range and record QIDs with different answers"""
    correct_count = 0
    total_count = 0
    mismatched_qids = []  # Used to record QIDs with different answers

    for gt in ground_truth_answers['ground_truths']:
        qid = gt['qid']
        # Only calculate for QIDs within the specified range (Insurance range is 51 to 100)
        if qid in qid_range:
            correct_doc = gt['retrieve']
            # Find the retrieve in predicted answers with the same QID
            predicted_doc = next((pred['retrieve'] for pred in predicted_answers['answers'] if pred['qid'] == qid), None)
            # Calculate Precision@1
            if predicted_doc == correct_doc:
                correct_count += 1
            else:
                mismatched_qids.append(qid)  # Add different QIDs to the list
            total_count += 1

    # Precision@1 calculation
    precision_at_1_score = correct_count / total_count if total_count > 0 else 0
    return precision_at_1_score, mismatched_qids

def load_json_file(file_path):
    """Read JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    # Specify the file paths for predicted results and ground truth
    predicted_file_path = "dataset/preliminary/finance_pred_retrieve.json"  # JSON file for predicted results
    ground_truth_file_path = "dataset/preliminary/ground_truths_example.json"  # JSON file for Insurance ground truths

    # Read JSON data
    predicted_answers = load_json_file(predicted_file_path)
    ground_truth_answers = load_json_file(ground_truth_file_path)

    # Set the QID range for the Insurance part (QIDs 51 to 100)
    insurance_qid_range = range(51, 100)

    # Calculate Precision@1 for the Insurance part and get different QIDs
    avg_precision_at_1, mismatched_qids = calculate_precision_at_1(predicted_answers, ground_truth_answers, insurance_qid_range)

    # Display the Precision@1 result for the Insurance part
    print(f"The Insurance Average Precision@1: {avg_precision_at_1:.7f}")
    if mismatched_qids:
        print("Mismatched QIDs:", mismatched_qids)
    else:
        print("All QIDs predictions are correct")
