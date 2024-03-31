import pytesseract
from ultralytics import YOLO
from typing import Union
from PIL import Image

class CarPlate:
    def __init__(self, model_path: str, tesseract_path: str, tesseract_white_list: str):
        self.modelPath = model_path
        self.tesseractPath = tesseract_path
        self.whiteList = tesseract_white_list

    def read_text(self, image: Union[str, Image.Image]) -> str:
        if isinstance(image, str):
            image_pil = Image.open(image)
        elif isinstance(image, Image.Image):
            image_pil = image
        else:
            raise TypeError("The input parameter must be either a string (the path to the image) or a Pillow image object")
        gray_image_pil = image_pil.convert('L').point(lambda x: 255 - x)
        pytesseract.pytesseract.tesseract_cmd = self.tesseractPath
        custom_config = '--psm 6 -c tessedit_char_whitelist=' + self.whiteList
        recognized_text = pytesseract.image_to_string(gray_image_pil, config=custom_config)

        return recognized_text


    def CarPlate_detect(self, path: str) -> Image.Image:
        image = Image.open(path)
        model = YOLO(self.modelPath)
        results = model(source=image, save=False)
        boxes = results[0].boxes.xyxy.cpu().tolist()

        if len(boxes) > 0:
            boxes = boxes[0]
            xmin = int(boxes[0])
            ymin = int(boxes[1])
            xmax = int(boxes[2])
            ymax = int(boxes[3])
            
            if results:
                cropped_image = image.crop((xmin, ymin, xmax, ymax))
                return cropped_image
            else:
                return False


