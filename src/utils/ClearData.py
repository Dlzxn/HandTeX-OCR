import os
from pathlib import Path
from PIL import Image

data_dir = Path("/home/alex/dev/python/HandTeX-OCR/src/data/archive/")

count = 0
for img_path in data_dir.rglob("*.*"):
    try:
        with Image.open(img_path) as img:
            img.verify()  # проверяем, что файл корректный
    except (IOError, SyntaxError, Image.UnidentifiedImageError) as e:
        print(f"Удаляем битый файл: {img_path}")
        count += 1
        os.remove(img_path)

print(f"Files with errors deleted: {count}")
