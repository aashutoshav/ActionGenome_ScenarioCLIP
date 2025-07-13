import json

data = json.load(open("file.json"))

actions = data["actions"]

new_actions = set()

mapping = {}

for action in actions:
    if action.endswith("ed"):
        new_actions.add(action[::-1].replace("de", "gni", 1)[::-1])
        mapping[action] = action[::-1].replace("de", "gni", 1)[::-1]
    else:
        new_actions.add(action)

actions = list(new_actions)

t = [
    "taking a photo",
    "taking a photograph",
    "taking a selfie",
    "taking photos",
]

for x in t:
    mapping[x] = "taking photo"
    
actions.append("taking photo")
for s in t:
    actions.remove(s)

json.dump({"actions": actions, "mapping": mapping}, open("action_classes.json", "w"), indent=4)