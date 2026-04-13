from datasets import load_dataset
import pandas as pd

# Load SQuAD 2.0
dataset = load_dataset("squad_v2", split='train', streaming=True)

data = []
ans_count = 0
unans_count = 0

for item in dataset:
    # Get 50 answerable
    if len(item['answers']['text']) > 0 and ans_count < 50:
        data.append({
            "context": item['context'],
            "question": item['question'],
            "label": "Answerable"
        })
        ans_count += 1
    # Get 50 unanswerable (The 'Scary' Agent test)
    elif len(item['answers']['text']) == 0 and unans_count < 50:
        data.append({
            "context": item['context'],
            "question": item['question'],
            "label": "Unanswerable"
        })
        unans_count += 1
    
    if ans_count == 50 and unans_count == 50:
        break

df = pd.DataFrame(data)
df.to_csv("squad_t1_test.csv", index=False)
print("Done! You have squad_t1_test.csv with 100 rows.")