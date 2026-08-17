import xml.etree.ElementTree as ET
from configs import IMAGE_WIDTH, IMAGE_HEIGHT, CLASS_TO_IDX
from pathlib import Path
from PIL import Image
import csv

# parse individual xml files
def parse_xml(xml_path, writer):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    # store ratios to scale bounding box coordinates
    img_width_ratio = IMAGE_WIDTH / width
    img_height_ratio = IMAGE_HEIGHT / height

    for obj in root.findall("object"):
        class_name = obj.find("name").text
        class_idx = CLASS_TO_IDX[class_name]

        # parse bounding box coordinates and scale them to align with the
        # resized image width and height
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text) * img_width_ratio
        ymin = int(bndbox.find("ymin").text) * img_height_ratio
        xmax = int(bndbox.find("xmax").text) * img_width_ratio
        ymax = int(bndbox.find("ymax").text) * img_height_ratio

        # 1) convert (xmin, ymin, xmax, ymax) to (x, y, w, h) format
        # 2) normalize all values with respect to the resized images' width and
        # height
        x = int(round((xmin+xmax)/2)) / IMAGE_WIDTH
        y = int(round((ymin+ymax)/2)) / IMAGE_HEIGHT
        w = int(round(xmax-xmin)) / IMAGE_WIDTH
        h = int(round(ymax-ymin)) / IMAGE_HEIGHT

        # write to annotations.csv file
        writer.writerow([filename, class_idx, x, y, w, h])

# write to csv annotations files
def write_csv(output_dir, annot_dir):
    with open (output_dir / "annotations.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")

        writer.writerow(["filename", "class_idx", "x", "y", "w", "h"])

        for annot_path in annot_dir.glob("*.xml"):
            parse_xml(annot_path, writer)

# resize image dimensions to 224x224
def process_images(img_dir, output_img_dir):
    for img_path in img_dir.glob("*.jpg"):
        output_img_path = output_img_dir / img_path.name

        with Image.open(img_path) as img:
            processed_img = img.convert("RGB").resize(
                (IMAGE_WIDTH, IMAGE_HEIGHT)
            )
            processed_img.save(output_img_path)

# 1) raw annotations -> structure annotations.csv file
# 2) raw image dimensions -> resize to 224x224
def preprocess():
    # create data directory
    data_dir = Path("../data").mkdir(parents=True, exist_ok=True)

    # move trainval and test dataset directories into data directory
    trainval_dir = Path("../VOCtrainval_06-Nov-2007")
    test_dir = Path("../VOCtest_06-Nov-2007")

    trainval_dir.rename(data_dir / trainval_dir.name)
    test_dir.rename(data_dir / test_dir.name)

    # store the raw trainval and test annotations directories
    trainval_annot_dir = Path(
        "../data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations"
    )
    test_annot_dir = Path(
        "../data/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/Annotations"
    )

    # store the raw trainval and test image directories
    trainval_img_dir = Path(
        "../data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages"
    )
    test_img_dir = Path(
        "../data/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages"
    )

    # create the trainval and test output directories
    trainval_output_dir = Path("../data/preprocessed/trainval/images").mkdir(
        parents=True, exist_ok=True
    )
    test_output_dir = Path("../data/preprocessed/test/images").mkdir(
        parents=True, exist_ok=True
    )

    # write the annotations.csv file to the respective output directory
    write_csv(trainval_output_dir.parent, trainval_annot_dir)
    write_csv(test_output_dir.parent, test_annot_dir)

    # process the images and store them in their respective output image
    # directories
    process_images(trainval_img_dir, trainval_output_dir)
    process_images(test_img_dir, test_output_dir)

if __name__ == "__main__":
    preprocess()