import json

def calculate_precision_at_1(predicted_answers, ground_truth_answers):
    """計算 Precision@1 和 Average Precision@1"""
    correct_count = 0
    total_count = len(ground_truth_answers['ground_truths'])

    for gt in ground_truth_answers['ground_truths']:
        qid = gt['qid']
        correct_doc = gt['retrieve']

        # 找到預測答案中相同 qid 的 retrieve
        predicted_doc = next((pred['retrieve'] for pred in predicted_answers['answers'] if pred['qid'] == qid), None)

        # 計算 Precision@1
        if predicted_doc == correct_doc:
            correct_count += 1

    # Precision@1 計算方式
    precision_at_1_score = correct_count / total_count
    return precision_at_1_score

def load_json_file(file_path):
    """讀取 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    # 指定預測結果和真實標記的文件路徑
    predicted_file_path = "dataset/preliminary/pred_retrieve.json"  # 替換成您的預測結果文件路徑
    ground_truth_file_path = "dataset/preliminary/ground_truths_example.json"  # 替換成您的真實標記文件路徑

    # 讀取 JSON 資料
    predicted_answers = load_json_file(predicted_file_path)
    ground_truth_answers = load_json_file(ground_truth_file_path)

    # 計算 Precision@1 和 Average Precision@1
    avg_precision_at_1 = calculate_precision_at_1(predicted_answers, ground_truth_answers)

    # 顯示結果
    print(f"Average Precision@1: {avg_precision_at_1:.7f}")