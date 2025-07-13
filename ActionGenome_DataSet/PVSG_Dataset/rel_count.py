import json

path = "metadata.json"

with open(path, 'r') as f:
    data = json.load(f)
    
mp = {}    
for dct in data:
    cnt_dct_relations = len(dct['relations'])
    if cnt_dct_relations not in mp:
        mp[cnt_dct_relations] = 1
    else:
        mp[cnt_dct_relations] += 1
        
for key in sorted(mp.keys()):
    print(key, mp[key])
    