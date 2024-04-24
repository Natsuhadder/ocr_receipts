import pytesseract
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from StringUtils import CorrectDigits
import cv2
import numpy as np 

def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


def denormalize_box(box,width,height):
    return [
        int((box[0] * width)/1000),
        int((box[1] * height)/1000),
        int((box[2] * width)/1000),
        int((box[3] * height)/1000),
    ]


def ProcessImage(image ) :
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.adaptiveThreshold(image, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 61, 12)
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image,kernel, iterations = 1)
    image = cv2.bitwise_not(image)
    image = Image.fromarray(image)
    image = image.resize((600,1000))

    return image
def apply_tesseract(image: Image.Image, config = '--psm 4'):
    """Applies Tesseract OCR on a document image, and returns recognized words + normalized bounding boxes."""

    # apply OCR
    data = pytesseract.image_to_data(image, output_type="dict", config= config)
    words, left, top, width, height = data["text"], data["left"], data["top"], data["width"], data["height"]

    # filter empty words and corresponding coordinates
    irrelevant_indices = [idx for idx, word in enumerate(words) if not word.strip()]
    words = [word for idx, word in enumerate(words) if idx not in irrelevant_indices]
    left = [coord for idx, coord in enumerate(left) if idx not in irrelevant_indices]
    top = [coord for idx, coord in enumerate(top) if idx not in irrelevant_indices]
    width = [coord for idx, coord in enumerate(width) if idx not in irrelevant_indices]
    height = [coord for idx, coord in enumerate(height) if idx not in irrelevant_indices]

    # turn coordinates into (left, top, left+width, top+height) format
    actual_boxes = []
    for x, y, w, h in zip(left, top, width, height):
        actual_box = [x, y, x + w, y + h]
        actual_boxes.append(actual_box)

    image_width, image_height = image.size

    # finally, normalize the bounding boxes
    normalized_boxes = []
    for box in actual_boxes:
        normalized_boxes.append(normalize_box(box, image_width, image_height))

    assert len(words) == len(normalized_boxes), "Not as many words as there are bounding boxes"

    corrected_words = []
    for word in words : 
        corrected_words.append(CorrectDigits(word))

    return corrected_words, normalized_boxes



def DrawBoundBoxes(image,boxes):
    width,height = image.size
    fig, ax = plt.subplots()

    ax.imshow(image)
    for box in boxes:
        box = denormalize_box(box,width,height)

        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()