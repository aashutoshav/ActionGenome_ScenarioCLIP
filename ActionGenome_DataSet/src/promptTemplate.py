GEMMA_PROMPT = """
    Return only the following information:
    1. A list of major unique objects in the image (limit to 10 objects).
    2. The relations between those objects in the form of relation triplets: <object1> <relation> <object2>.
    3. Action should be a single word of a broad class, like "playing", "eating", "running", "talking", "driving" etc. Avoid specific actions like "serving", "gulping", "steering" etc.
    4. The description tag should be one or two sentences long, describing the image in general terms.
    Do not provide any extra information or descriptions.
    Give the output in a json format, example:
    {
        "action": "<some action>"
        "objects": ["object1", "object2", ...],
        "relations": [
            ["object1", "relation", "object2"],
            ...
        ]
        "dense caption": "The image shows a... <description>"
    }
"""

UNIQUE_OBJ_PROMPT = """
    <image>\nYou are provided with an image. Your task is to detect the unique objects in the image.
    
Action Detection:
- Provide a verdict for the image if one or multiple instances of actions are taking place in the image. verdict : "Yes" or "No".
- Provide the type of actions taking place in the image. If there are multiple actions, provide the most prominent one.
- The action names should not be more than four or five words.
- If no actions are detected in the image, i.e the verdict is 'No', then set action : 'None'

Object Detection:
- Provide a list of unique objects in the image.
- DO NOT REPEAT ANY OBJECTS IN THE LIST.
- Do not instantiate the objects, i.e. do not use numbers to differentiate between multiple instances of the same object.
- DO NOT RETURN MORE THAN 10 OBJECTS.
- You do not need to return 10 objects if there are less than 10 unique objects in the image.

Relations Detection:
- Based on the action detected, provide prominent relationships between objects and people in the image that best describe single and multiple actions.
- Do not use collective nouns for objects, so prefer to use 'person 1', 'person 2' instead of 'some people'.
- Use the relations from the relations mentioned above to describe the relationships between the objects.
- The relations should be in the format : ['object1', 'relation', 'object2'] Eg : ['person', 'walking with', 'dog'], ['person', 'reading', 'book'], ['dog', 'walking on', 'road'] etc.
- Ensure that the objects are the first and last elements of the relation triplet, with the relation in between.
- Include multiple relations if applicable.
- Don't return repeating relations, and return no more than 10 relations.

Always generate a JSON as output.

Here are some good examples of the output format:

{
    "verdict": "Yes",
    "action": "catching",
    "unique_objects": ["person", "baseball glove", "baseball"],
    "relations": [["person", "catching", "baseball"], ["person", "wearing", "baseball glove"]]
}

{
    "verdict": "No",
    "action": "None",
    "unique_objects": ["bed", "drawers", "desk", "chair", "books", "pen", "cup"],
    "relations": [
        ["drawers", "attached to", "bed"],
        ["desk", "in front of", "bed"],
        ["chair", "next to", "desk"],
        ["books", "on", "desk"],
        ["pen", "on", "desk"],
        ["cup", "on", "desk"]
    ]
}

{
    "verdict": "Yes",
    "action": "drinking",
    "unique_objects": ["table", "wine glass", "man", "woman"],
    "relations": [["man", "drinking from", "wine glass"], ["woman", "drinking from", "wine glass"]]
}

{
    "verdict": "Yes",
    "action": "holding",
    "unique_objects": ["baby", "toothbrush", "camera", "person"],
    "relations": [["person", "holding", "baby"], ["baby", "holding", "toothbrush"], ['person', 'holding', 'camera']]
}
"""

LLAMA_PROMPT = """
    <image>\nYou are provided with an image. Your task is to detect the action, objects and their relationships in the image.

Action Detection:
- Provide a verdict for the image if one or multiple instances of actions are taking place in the image. verdict : "Yes" or "No".
- Provide the type of actions taking place in the image. If there are multiple actions, provide the most prominent one.
- The action names should not be more than four or five words.
- If no actions are detected in the image, i.e the verdict is 'No', then set action : 'None'

Object Detection:
- Provide only the names of objects in the image.
- It is VERY IMPORTANT for us to label each instance of an object separately
- So there are multiple objects of the same type, you can use numbers to differentiate them.
- For example, instead of saying ['person', 'person', 'person'], you MUST say ['person 1', 'person 2', 'person 3']
- But if there is only one instance of an object, you don't need to add a number to it.
- Always detect the objects even if an action is not detected.
- **Important**: Return only objects with a confidence score of at least '20%' of the object with the highest confidence score.

Relations Detection:
- Based on the action detected, provide prominent relationships between objects and people in the image that best describe single and multiple actions.
- Do not use collective nouns for objects, so prefer to use 'person 1', 'person 2' instead of 'some people'.
- Use the relations from the relations mentioned above to describe the relationships between the objects.
- The relations should be in the format : ['object1', 'relation', 'object2'] Eg : ['person', 'walking with', 'dog'], ['person', 'reading', 'book'], ['person', 'talking to', 'person'] etc.
- Ensure that the objects are the first and last elements of the relation triplet, with the relation in between.
- Include multiple relations if applicable.
- **Important**: IN THE RELATION TRIPLETS ONLY USE THE NAMES OF THE OBJECTS AS DETECTED BY YOU EXACTLY AS THEY ARE IN THE OBJECT LIST.
- You can use the following hint for a basic understanding of a relation in the scene, but can't copy it exactly, as this is one of many possible relations in the image: {hintPlaceholder}.

Always generate a JSON as output.

Here are some good examples of the output format:

{
    "verdict": "Yes",
    "action": "catching",
    "objects": ["one person", "one baseball glove", "one baseball"],
    "relations": [["one person", "catching", "one baseball"], ["one baseball glove", "on the hand of", "one person"]]
}

{
    "verdict": "No",
    "action": "None",
    "objects": ["bed", "drawers", "desk", "chair", "books", "pen", "cup"],
    "relations": [
        ["drawers", "attached to", "bed"],
        ["desk", "in front of", "bed"],
        ["chair", "next to", "desk"],
        ["books", "on", "desk"],
        ["pen", "on", "desk"],
        ["cup", "on", "desk"]
    ]
}

{
    "verdict": "Yes",
    "action": "drinking",
    "objects": ["table", "wine glass 1", "wine glass 2", "one man", "one woman"],
    "relations": [["one man", "drinking from", "wine glass 1"], ["one woman", "drinking from", "wine glass 2"]]
}

{
    "verdict": "Yes",
    "action": "holding",
    "objects": ["baby", "toothbrush", "camera", "person"],
    "relations": [["person", "holding", "baby"], ["baby", "holding", "toothbrush"], ['person', 'holding', 'camera']]
}
"""

TEMPLATE_ACTION_OBJECTS_RELATIONS_LIST_LLAVA = """[INST] <image>\n You are provided with an image.
Your task is to detect the action, objects and their relationships in the image.
Follow the guidelines below:

RELATIONS:
Relations connect two objects in the image. Some examples of relations are:
Here are some relations based on the category of action in the image:

positional_relations = [
    'over', 'in front of', 'beside', 'on', 'in', 'attached to', 'under', 'below', 'above',
    'near', 'between', 'across from', 'adjacent to', 'behind', 'around', 'inside', 'outside',
    'alongside', 'adjacent to', 'far from', 'touching', 'separate from'
]

common_object_object_relations = [
    'hanging from', 'on the back of', 'falling off', 'going down', 'painted on',
    'leaning against', 'touching', 'stacked on', 'supporting', 'wrapped around', 'connected to',
    'covered with', 'suspended over', 'standing on', 'lying on', 'mounted on', 'attached to',
    'suspended from', 'wrapped in', 'bordered by'
]

traffic_relations = [
    'driving', 'riding', 'parked on', 'driving on', 'speeding', 'turning', 'stopping',
    'accelerating', 'reversing', 'overtaking', 'crashing into', 'merging', 'exiting',
    'stuck in traffic', 'avoiding', 'colliding with', 'following', 'preceding', 'being overtaken by'
]

sports_relations = [
    'about to hit', 'kicking', 'swinging', 'throwing', 'catching', 'dribbling', 'passing',
    'shooting', 'serving', 'spiking', 'blocking', 'defending', 'tackling', 'catching up with',
    'falling behind', 'celebrating'
]

interaction_with_background_relations = [
    'entering', 'exiting', 'enclosing', 'approaching', 'leaving', 'hiding behind',
    'emerging from', 'standing against', 'climbing onto', 'jumping off', 'blending into',
    'camouflaged against', 'hiding in', 'obscured by', 'emerging from', 'disappearing into'
]

human_relations = [
    'talking to', 'walking with', 'sitting on', 'standing on', 'cooking', 'throwing', 'slicing',
    'eating', 'drinking', 'reading', 'writing', 'playing with', 'hugging', 'holding', 'pushing',
    'pulling', 'lifting', 'carrying', 'washing', 'watching', 'collaborating with', 'arguing with',
    'assisting', 'observing', 'guarding', 'guiding', 'confronting', 'embracing'
]

Action Detection:
- Provide a verdict for the image if one or multiple instances of actions are taking place in the image. verdict : "Yes" or "No".
- Provide the type of actions taking place in the image. If there are multiple actions, provide the most prominent one.
- The action names should not be more than four or five words.
- If no actions are detected in the image, i.e the verdict is 'No', then set action : 'None'

Object Detection:
- Provide only the names of objects in the image.
- If there are more than one instances of the same object, you can use words like 'one', 'two', 'three' to specify the number of instances.
- It is VERY IMPORTANT for us to label each instance of an object separately
- So there are multiple objects of the same type, you can use numbers to differentiate them.
- For example, instead of saying ['person', 'person', 'person'], you MUST say ['person 1', 'person 2', 'person 3']
- Avoid Duplicate names.
- The object names should be less than four or five words.
- Always detect the objects even if an action is not detected.

Relations Detection:
- Based on the action detected, provide prominent relationships between objects and people in the image that best describe single and multiple actions.
- Use the relations from the relations mentioned above to describe the relationships between the objects.
- The relations should be in the format : ['object1', 'relation', 'object2'] Eg : ['person', 'walking with', 'dog'], ['person', 'reading', 'book'], ['person', 'talking to', 'person'] etc.
- Don't give output relations like ['person', 'person', 'person', 'person']
- Include multiple relations if applicable.
- You can use the following hint for a basic understanding of a relation in the scene, but can't copy it exactly, as this is one of many possible relations in the image: {hintPlaceholder}

Always generate a JSON as output.

Here are some good examples of the output format:

{
    "verdict": "Yes",
    "action": "catching",
    "objects": ["one person", "one baseball glove", "one baseball"],
    "relations": [["one person", "catching", "one baseball"], ["one baseball glove", "on the hand of", "one person"]]
}

{
    "verdict": "No",
    "action": "None",
    "objects": ["bed", "drawers", "desk", "chair", "books", "pen", "cup"],
    "relations": [
        ["drawers", "attached to", "bed"],
        ["desk", "in front of", "bed"],
        ["chair", "next to", "desk"],
        ["books", "on", "desk"],
        ["pen", "on", "desk"],
        ["cup", "on", "desk"]
    ]
}

{
    "verdict": "Yes",
    "action": "drinking",
    "objects": ["table", "some wine glasses", "one man", "one woman"],
    "relations": [["one man", "drinking from", "wine glass"], ["one woman", "drinking from", "wine glass"]]
}

{
    "verdict": "Yes",
    "action": "holding",
    "objects": ["baby", "toothbrush", "camera", "person"],
    "relations": [["person", "holding", "baby"], ["baby", "holding", "toothbrush"]]
}

[/INST]"""

TEMPLATE_ACTION_OBJECTS_RELATIONS_LIST_FLORENCE = """[INST] <image>\n You are provided with an image.
Your task is to detect the action, objects and their relationships in the image.
Follow the guidelines below:

RELATIONS:
Relations connect two objects in the image. Some examples of relations are:
Here are some relations based on the category of action in the image:

positional_relations = [
    'over', 'in front of', 'beside', 'on', 'in', 'attached to', 'under', 'below', 'above',
    'near', 'between', 'across from', 'adjacent to', 'behind', 'around', 'inside', 'outside',
    'alongside', 'adjacent to', 'far from', 'touching', 'separate from'
]

common_object_object_relations = [
    'hanging from', 'on the back of', 'falling off', 'going down', 'painted on',
    'leaning against', 'touching', 'stacked on', 'supporting', 'wrapped around', 'connected to',
    'covered with', 'suspended over', 'standing on', 'lying on', 'mounted on', 'attached to',
    'suspended from', 'wrapped in', 'bordered by'
]

traffic_relations = [
    'driving', 'riding', 'parked on', 'driving on', 'speeding', 'turning', 'stopping',
    'accelerating', 'reversing', 'overtaking', 'crashing into', 'merging', 'exiting',
    'stuck in traffic', 'avoiding', 'colliding with', 'following', 'preceding', 'being overtaken by'
]

sports_relations = [
    'about to hit', 'kicking', 'swinging', 'throwing', 'catching', 'dribbling', 'passing',
    'shooting', 'serving', 'spiking', 'blocking', 'defending', 'tackling', 'catching up with',
    'falling behind', 'celebrating'
]

interaction_with_background_relations = [
    'entering', 'exiting', 'enclosing', 'approaching', 'leaving', 'hiding behind',
    'emerging from', 'standing against', 'climbing onto', 'jumping off', 'blending into',
    'camouflaged against', 'hiding in', 'obscured by', 'emerging from', 'disappearing into'
]

human_relations = [
    'talking to', 'walking with', 'sitting on', 'standing on', 'cooking', 'throwing', 'slicing',
    'eating', 'drinking', 'reading', 'writing', 'playing with', 'hugging', 'holding', 'pushing',
    'pulling', 'lifting', 'carrying', 'washing', 'watching', 'collaborating with', 'arguing with',
    'assisting', 'observing', 'guarding', 'guiding', 'confronting', 'embracing'
]

Action Detection:
- Provide a verdict for the image if one or multiple instances of actions are taking place in the image. verdict : "Yes" or "No".
- Provide the type of actions taking place in the image. If there are multiple actions, provide the most prominent one.
- The action names should not be more than four or five words.
- If no actions are detected in the image, i.e the verdict is 'No', then set action : 'None'

Object Detection:
- Provide only the names of objects in the image.
- It is VERY IMPORTANT for us to label each instance of an object separately
- So there are multiple objects of the same type, you can use numbers to differentiate them.
- For example, instead of saying ['person', 'person', 'person'], you MUST say ['person 1', 'person 2', 'person 3']
- Always detect the objects even if an action is not detected.
- Ensure all of these objects are detected: {objectsPlaceholder}.

Relations Detection:
- Based on the action detected, provide prominent relationships between objects and people in the image that best describe single and multiple actions.
- Use the relations from the relations mentioned above to describe the relationships between the objects.
- The relations should be in the format : ['object1', 'relation', 'object2'] Eg : ['person', 'walking with', 'dog'], ['person', 'reading', 'book'], ['person', 'talking to', 'person'] etc.
- Don't give output relations like ['person', 'person', 'person', 'person']
- Include multiple relations if applicable.
- You can use the following hint for a basic understanding of a relation in the scene, but can't copy it exactly, as this is one of many possible relations in the image: {hintPlaceholder}.

Always generate a JSON as output.

Here are some good examples of the output format:

{
    "verdict": "Yes",
    "action": "catching",
    "objects": ["one person", "one baseball glove", "one baseball"],
    "relations": [["one person", "catching", "one baseball"], ["one baseball glove", "on the hand of", "one person"]]
}

{
    "verdict": "No",
    "action": "None",
    "objects": ["bed", "drawers", "desk", "chair", "books", "pen", "cup"],
    "relations": [
        ["drawers", "attached to", "bed"],
        ["desk", "in front of", "bed"],
        ["chair", "next to", "desk"],
        ["books", "on", "desk"],
        ["pen", "on", "desk"],
        ["cup", "on", "desk"]
    ]
}

{
    "verdict": "Yes",
    "action": "drinking",
    "objects": ["table", "some wine glasses", "one man", "one woman"],
    "relations": [["one man", "drinking from", "wine glass"], ["one woman", "drinking from", "wine glass"]]
}

{
    "verdict": "Yes",
    "action": "holding",
    "objects": ["baby", "toothbrush", "camera", "person"],
    "relations": [["person", "holding", "baby"], ["baby", "holding", "toothbrush"]]
}

[/INST]"""
