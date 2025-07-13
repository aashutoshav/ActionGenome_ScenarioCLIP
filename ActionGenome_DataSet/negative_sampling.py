import json
import random
import nltk
from nltk.corpus import wordnet as wn

nltk.download('wordnet')

json_path = "/mnt/MIG_store/Datasets/ActionGenome/results_29_9/gemma_jsons/000000221592_gemma.json"

with open(json_path, 'r') as f:
    data = json.load(f)

relations = data['response']['relations']

def get_synonyms(word):
    synonyms = set()
    
    for syn in wn.synsets(word):
        for l in syn.lemmas():
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
    return list(synonyms)

def get_antonyms(word):
    antonyms = set()
    
    for syn in wn.synsets(word):
        for l in syn.lemmas():
            for lemma in l.antonyms():
                antonyms.add(lemma.name())
    return list(antonyms)

def generate_samples(relations, num_samples, pos_or_neg):
    if pos_or_neg == 'antonyms':

        for relation in relations:
            o1, rel, o2 = relation
            negatives = []

            pred_antonyms = get_antonyms(rel)

            for antonym in pred_antonyms:
                new_rel = [o1, antonym, o2]
                if new_rel not in relations:
                    negatives.append(new_rel)

            unrelated_objects = [r[0] for r in relations if r[0] != o1 and r[0] != o2]
            if unrelated_objects:
                random_unrelated = random.choice(unrelated_objects)
                if [o1, rel, random_unrelated] not in relations:
                    negatives.append([o1, rel, random_unrelated])
                if [random_unrelated, rel, o2] not in relations:
                    negatives.append([random_unrelated, rel, o2])

        return random.sample(negatives, min(num_samples, len(negatives)))

    else:
        for relation in relations:
            o1, rel, o2 = relation
            positives = []

            pred_synonyms = get_synonyms(rel)

            for synonym in pred_synonyms:
                new_rel = [o1, synonym, o2]
                if new_rel not in relations:
                    positives.append(new_rel)
        
        return random.sample(positives, min(num_samples, len(positives)))

negs = generate_samples(relations, 15, 'antonyms')
pos = generate_samples(relations, 15, "synonyms")

print(f"Original:")
print(relations)

print(f"Negatives:")
print(negs)
print(f"Positives:")
print(pos)
