import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline

object_detector = pipeline("object-detection",
                model="facebook/detr-resnet-50")


def draw_bounding_boxes(image, detections, font_path=None, font_size=20):
    """
    Draws bounding boxes on the given image based on the detections.
    :param image: PIL.Image object
    :param detections: List of detection results, where each result is a dictionary containing
                       'score', 'label', and 'box' keys. 'box' itself is a dictionary with 'xmin',
                       'ymin', 'xmax', 'ymax'.
    :param font_path: Path to the TrueType font file to use for text.
    :param font_size: Size of the font to use for text.
    :return: PIL.Image object with bounding boxes drawn.
    """
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
       
        font = ImageFont.load_default() 
    for detection in detections:
        box = detection['box']
        xmin = box['xmin']
        ymin = box['ymin']
        xmax = box['xmax']
        ymax = box['ymax']
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)
        label = detection['label']
        score = detection['score']
        text = f"{label} {score:.2f}"
        if font_path:  
            text_size = draw.textbbox((xmin, ymin), text, font=font)
        else:
            text_size = draw.textbbox((xmin, ymin), text)

        draw.rectangle([(text_size[0], text_size[1]), (text_size[2], text_size[3])], fill="red")
        draw.text((xmin, ymin), text, fill="white", font=font)

    return draw_image


def detect_object(image):
    raw_image = image
    output = object_detector(raw_image)
    processed_image = draw_bounding_boxes(raw_image, output)
    return processed_image

demo = gr.Interface(fn=detect_object,
                    inputs=[gr.Image(label="Select Image",type="pil")],
                    outputs=[gr.Image(label="Processed Image", type="pil")],
                    title="Arif's @GenAI Project: Object Detector",
                    description="This application will be used to detect objects inside the provided input image.")
demo.launch()




