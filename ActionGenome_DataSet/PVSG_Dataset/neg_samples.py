import random
import json
from tqdm import tqdm
import spacy

nlp = spacy.load("en_core_web_sm")

json_path = "metadata.json"

class Mapping:
    def __init__(self):
        self.verbs = {
            # Motion verbs
            "enter": "exit",
            "exit": "enter",
            "advance": "retreat",
            "retreat": "advance",
            "approach": "withdraw",
            "withdraw": "approach",
            "arrive": "depart",
            "depart": "arrive",
            "ascend": "descend",
            "descend": "ascend",
            "rise": "fall",
            "fall": "rise",
            "climb": "descend",
            "dive": "surface",
            "surface": "dive",
            "move": "stop",
            "stop": "move",
            "go": "come",
            "come": "go",
            "leave": "return",
            "return": "leave",
            "land": "take off",
            "take off": "land",
            "jump": "fall",
            "fall": "jump",
            # Placement verbs
            "put": "remove",
            "remove": "put",
            "insert": "extract",
            "extract": "insert",
            "push": "pull",
            "pull": "push",
            "raise": "lower",
            "lower": "raise",
            "attach": "detach",
            "detach": "attach",
            "connect": "disconnect",
            "disconnect": "connect",
            "open": "close",
            "close": "open",
            "expand": "contract",
            "contract": "expand",
            # Position verbs
            "stand": "lie",
            "lie": "stand",
            "face": "back",
            "back": "face",
            "sit": "stand",
            "stand": "sit",
            "kneel": "rise",
            "rise": "kneel",
            # State verbs
            "activate": "deactivate",
            "deactivate": "activate",
            "expand": "reduce",
            "reduce": "expand",
            "grow": "shrink",
            "shrink": "grow",
            "appear": "disappear",
            "disappear": "appear",
            "increase": "decrease",
            "decrease": "increase",
            "fill": "empty",
            "empty": "fill",
            # Interaction verbs
            "unite": "separate",
            "separate": "unite",
            "bond": "break apart",
            "break apart": "bond",
            "fasten": "unfasten",
            "unfasten": "fasten",
            "cover": "expose",
            "expose": "cover",
            "join": "split",
            "split": "join",
            # Other actions
            "rest": "work",
            "work": "rest",
            "capture": "release",
            "release": "capture",
            "adopt": "abandon",
            "abandon": "adopt",
            "collect": "scatter",
            "scatter": "collect",
            "load": "unload",
            "unload": "load",
            "pack": "unpack",
            "unpack": "pack",
            # Additional mappings
            "accelerate": "decelerate",
            "decelerate": "accelerate",
            "begin": "end",
            "end": "begin",
            "build": "demolish",
            "demolish": "build",
            "create": "destroy",
            "destroy": "create",
            "help": "harm",
            "harm": "help",
            "improve": "worsen",
            "worsen": "improve",
            "increase": "decrease",
            "decrease": "increase",
            "learn": "forget",
            "forget": "learn",
            "lose": "gain",
            "gain": "lose",
            "make": "break",
            "break": "make",
            "obtain": "lose",
            "lose": "obtain",
            "offer": "refuse",
            "refuse": "offer",
            "start": "stop",
            "stop": "start",
            "succeed": "fail",
            "fail": "succeed",
        }

        self.prepositions = {
            # Basic spatial prepositions
            "in front of": "behind",
            "behind": "in front of",
            "above": "below",
            "below": "above",
            "over": "under",
            "under": "over",
            "inside": "outside",
            "outside": "inside",
            "left of": "right of",
            "right of": "left of",
            "on top of": "beneath",
            "beneath": "on top of",
            # Additional spatial prepositions
            "within": "outside of",
            "outside of": "within",
            "near": "far from",
            "far from": "near",
            "adjacent to": "away from",
            "away from": "adjacent to",
            "alongside": "away from",
            "atop": "beneath",
            "among": "separate from",
            "separate from": "among",
            "around": "away from",
            "across from": "alongside",
            "beyond": "before",
            "before": "beyond",
            # Directional prepositions
            "toward": "away from",
            "towards": "away from",
            "into": "out of",
            "out of": "into",
            "onto": "off of",
            "off of": "onto",
            "up": "down",
            "down": "up",
            "through": "out of",
            # Symmetric relations (their own opposites)
            "next to": "next to",
            "beside": "beside",
            "between": "between",
            "against": "against",
            "along": "along",
            "parallel to": "parallel to",
            "level with": "level with",
            # Spatial and directional opposites and symmetric relations
            "underneath": "above",
            "beside": "apart from",
            "adjacent to": "distant from",
            "away from": "close to",
            "opposite": "aligned with",
            "close to": "far from",
            "centered in": "decentered from",
            "perpendicular to": "parallel to",
            "within range of": "out of range of",
            "encompassing": "enclosed by",
            "outside of": "enclosed within",
            "inside of": "outside of",
            "below": "above",
            "next to": "apart from",
            "beyond": "within",
            "adjacent": "distant",
            "alongside": "opposite to",
            "surrounding": "apart from",
            "nearby": "remote from",
            "close to": "distant from",
            "in line with": "out of alignment with",
            "near": "far away from",
            "attached to": "detached from",
            "in": "out of",
            "enclosed within": "excluded from",
            "contained in": "outside of",
            "in the vicinity of": "distant from",
            # Additional mappings
            "across": "along",
            "against": "away from",
            "amongst": "separate from",
            "around": "through",
            "at": "away from",
            "by": "away from",
            "from": "to",
            "into": "out",
            "on": "off",
            "onto": "off",
            "over": "underneath",
            "past": "before",
            "since": "until",
            "throughout": "outside",
            "till": "until",
            "to": "from",
            "toward": "away",
            "underneath": "above",
            "until": "since",
            "up": "down",
            "upon": "off",
            "via": "past",
            "with": "without",
            "within": "outside",
        }

    def get_spacy_negative(self, relation):
        if len(relation) != 3:
            return []
        o1, rel, o2 = relation
        rel_token = nlp(rel)[0]

        neg_rel_candidates = []

        all_candidates = list(self.verbs.values()) + list(self.prepositions.values())
        for candidate in all_candidates:
            candidate_token = nlp(candidate)[0]

            if rel_token.similarity(candidate_token) < 0.5:
                neg_rel_candidates.append(candidate)

        if neg_rel_candidates:
            return [
                o1,
                random.choice(neg_rel_candidates),
                o2,
            ]

        return []

    def get_spacy_negative(self, relation):
        if len(relation) != 3:
            return []
        o1, rel, o2 = relation
        rel_token = nlp(rel)[0]

        max_similarity = 0
        most_similar_key = None

        for key in self.verbs.keys() | self.prepositions.keys():
            key_token = nlp(key)[0]
            similarity = rel_token.similarity(key_token)

            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_key = key

        if most_similar_key:
            if most_similar_key in self.verbs:
                neg_rel = self.verbs[most_similar_key]
            else:
                neg_rel = self.prepositions[most_similar_key]

            return [o1, neg_rel, o2]

        return []

    def extract_negative_relation(self, relation):
        """
        Extract the opposite relation for a given relation.
        """
        o1, rel, o2 = relation

        if rel in self.prepositions:
            neg_rel = [o1, self.prepositions[rel], o2]
        elif rel in self.verbs:
            neg_rel = [o1, self.verbs[rel], o2]
        else:
            neg_rel = self.get_spacy_negative(relation)

        return neg_rel

    def extract_object_switch(self, relation, relations):
        obj_list = set()
        for rel in relations:
            if len(rel) == 3:
                obj_list.add(rel[0])
                obj_list.add(rel[2])

        obj_list = list(obj_list)
        neg_rel_list = []
        for _ in range(3):
            if len(relation) == 3:
                obj = random.choice(obj_list)
                if [relation[0], relation[1], obj] not in relations and obj != relation[
                    0
                ]:
                    neg_rel_list.append([relation[0], relation[1], obj])
                if [obj, relation[1], relation[2]] not in relations and obj != relation[
                    2
                ]:
                    neg_rel_list.append([obj, relation[1], relation[2]])

        return neg_rel_list

with open(json_path, "r") as f:
    samples = json.load(f)
    
new_output = []

for sample in tqdm(samples):
    sample_copy = sample.copy()
    focused_images = sample_copy["relations"]
    
    all_rels = []
    all_objs = set()
    
    print(f"SAMPLE:")
    print(sample)
    print(f"\n"*10)
    
    for key in focused_images.keys():
        rel = focused_images[key]['relation']
        all_rels.append(rel)
        objs = [t for t in focused_images[key]['labels']]
        for obj in objs:
            all_objs.add(obj)
    
    
    print(f"all_rels: {all_rels}")
    print("all_objs: ", all_objs)
    
    for key in focused_images.keys():
        rel = focused_images[key]['relation']
        r1 = Mapping().extract_negative_relation(rel)
        r2 = Mapping().extract_object_switch(rel, all_rels)
        neg_samples = [r1] + r2
    
        sample_copy['relations'][key]['neg_samples'] = neg_samples
        
    new_output.append(sample_copy)

with open("metadata_w_neg.json", "w") as f:
    json.dump(new_output, f, indent=4)
