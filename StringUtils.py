import numpy as np



def CorrectDigits(text = str) :

    """
    Corrects mispelled digits in a given text based on a dictionary of personal knowledge.
    """
    DigitCorrection = {'@': '0', 'S': '5', 'U': '4', 'T': '7', 'I': '1', 'L': '1', 'O': '0', 'Q': '0', 'B': '8', 'o': '0', 's': '5'}
    corrected_text = ""
    
    for i in range(len(text)):
        if text[i] in DigitCorrection:
            if (i > 0 and (text[i-1].isdigit() or text[i-1]=='.' )) or (i == 0 and text[i+1].isdigit()):
                corrected_text += DigitCorrection[text[i]]
            elif (i < len(text)-1 and (text[i+1].isdigit() or text[i+1]=='.')):
                corrected_text += DigitCorrection[text[i]]
            else:
                corrected_text += text[i]
        else:
            corrected_text += text[i]

    return corrected_text
