
import json

f = open('results.json')

data = json.load(f)

summary = {"invalid": 0, "true_pos": 0,
           "false_pos": 0, "true_neg": 0, "false_neg": 0}
for i in data:
    if i.get('status') == 'invalid':
        summary['invalid'] += 1
    else:
        winner = max(i, key=i.get)
        if winner == 'companyA.txt':
            summary['true_pos'] += 1
            summary['true_neg'] += 2
        else:
            summary['false_neg'] += 1
            summary['false_pos'] += 1
            summary['true_neg'] += 1

f.close()
print(summary)
