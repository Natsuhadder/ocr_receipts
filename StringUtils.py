import numpy as np
import openai
from difflib import SequenceMatcher



def CorrectDigits(text = '') :

    """
    Corrects mispelled digits in a given text based on a dictionary of personal knowledge.
    """
    DigitCorrection = {'@': '0', 'S': '5', 'U': '4', 'T': '7', 'I': '1', 'L': '1', 'O': '0', 'Q': '0', 'B': '8', 'o': '0', 's': '5', ',' : '.', 'i' : '.' , 'e' : '4'}
    corrected_text = ""
    
    for i in range(len(text)):
        if text[i] in DigitCorrection and len(text)>1:
            if (i > 0 and (text[i-1].isdigit() or text[i-1]=='.' )) or (len(text)>1 and i == 0 and text[i+1].isdigit()):
                corrected_text += DigitCorrection[text[i]]
            elif (i < len(text)-1 and (text[i+1].isdigit() or text[i+1]=='.')):
                corrected_text += DigitCorrection[text[i]]
            else:
                corrected_text += text[i]
        else:
            corrected_text += text[i]

    return corrected_text


def AskOpenai(message):

    return openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "user",
        "content": message
        }
    ],
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )


def AssignLabel(word: str, entities: dict):
    word_set = word.replace(",", "").strip().split()
    for entity_key in list(entities.keys()):  # Iterate over a copy of keys
        if entity_key == 'ADDRESS':
            entities['LOCATION'] = entities.pop(entity_key)
            entity_key = 'LOCATION'

        entity_values = str(entities[entity_key]).replace(",", "").strip()
        entity_set = entity_values.split()

        # Calculate similarity ratio
        matches_count = sum(
            any(SequenceMatcher(a=l.lower(), b=b.lower()).ratio() > 0.8 for b in entity_set)
            for l in word_set
        )

        # Adjust matching conditions as needed
        if (entity_key.upper() == 'LOCATION' and (matches_count / len(word_set)) >= 0.5) or \
           (entity_key.upper() != 'LOCATION' and matches_count == len(word_set)) or \
           matches_count == len(entity_set):
            return entity_key.upper()

    return "O"



def AssignNERTags(words : list, entities : dict, boxes : list):
    if len(words)>1:
        ner_tags = []
        max_area = {"TOTAL": (0, -1), "DATE": (0, -1)}  # Value, index
        already_labeled = {"TOTAL": False,
                        "DATE": False,
                        "LOCATION": False,
                        "COMPANY": False,
                        "TAX" : False,
                        "CURRENCY" : False,
                        "O": False
        }

        for i , word in enumerate(words) :
            label = AssignLabel(word, entities)

            already_labeled[label] = True
            if (label == "LOCATION" and already_labeled["TOTAL"]) or \
            (label == "COMPANY" and (already_labeled["DATE"] or already_labeled["TOTAL"])):
                label = "O"

            if (label =='COMPANY' and already_labeled['LOCATION'] or already_labeled["TAX"]) :
                label = "O"

            # Assign to the largest bounding box
            if label in ["TOTAL", "DATE"]:
                bbox = boxes[i]
                area = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])

                if max_area[label][0] < area:
                    max_area[label] = (area, i)

                label = "O"

            ner_tags.append(label)
        
        

        ner_tags[max_area["DATE"][1]] = "DATE"
        ner_tags[max_area["TOTAL"][1]] = "TOTAL"

    else :
        return []

    
    return ner_tags