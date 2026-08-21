import json

with open('sellers.json') as f:
    data = json.load(f)

data.sort(key=lambda x: x['username'].lower())

with open('sellers.json', 'w') as f:
    f.write('[\n')
    for i, entry in enumerate(data):
        line = '  ' + json.dumps(entry)
        if i < len(data) - 1:
            line += ','
        f.write(line + '\n')
    f.write(']\n')