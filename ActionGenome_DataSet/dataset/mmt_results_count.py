import os

gemma_path = './gemma_jsons'
dino_path = './dino_results'
sam_path = './sam_results'

gc = 0
for folder in os.listdir(gemma_path):
    gc += len(os.listdir(os.path.join(gemma_path, folder)))
    
print(gc)

dc = 0
for folder in os.listdir(dino_path):
    dc += len(os.listdir(os.path.join(dino_path, folder)))
    
print(dc)

sc = 0
for folder in os.listdir(sam_path):
    f_path = os.path.join(sam_path, folder)
    for sub_f in os.listdir(f_path):
        sc += len(os.listdir(os.path.join(f_path, sub_f))) - 1
        
print(sc)