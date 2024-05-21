# Data collection
    git clone https://github.com/EscVM/OIDv4_ToolKit.git
    cd OIDv4_ToolKit
    python main.py downloader --classes <ClassNames> --type_csv train --multiclass 1 --limit <DataSize>

To get the data from Kaggle dataset, access:
    https://www.kaggle.com/datasets/andrewmvd/car-plate-detection/download?datasetVersionNumber=1

# Data management
We will use Roboflow to manage and anotate the images

# Data training
    https://colab.research.google.com/drive/1MT5me9t18PFTEPNs7tYWa1IOelhTNWEt?usp=sharing

# API for getting data from Roboflow

## Car_Motocycle dataset
    !pip install roboflow

    from roboflow import Roboflow
    rf = Roboflow(api_key="ArRtQ5Zf4EONPlk1ETfa")
    project = rf.workspace("anpr-sy3tn").project("car-motorbike-detector")
    dataset = project.version(1).download("yolov8")

## Cars' register plate
    !pip install roboflow

    from roboflow import Roboflow
    rf = Roboflow(api_key="TsnGjvJZN79ZEbpWdg9h")
    project = rf.workspace("no-7iqhh").project("platerecognition-zqyo1")
    dataset = project.version(3).download("yolov8")

## Pre-requirement for using tesseract OCR
1. Install tesseract by following this instruction: 
    https://tesseract-ocr.github.io/tessdoc/Installation.html
2. Copy the plate.traineddata into tessdata (tessdata is in the 'tesseract-ocr/4.00/' in the installation directory)
