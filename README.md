# **🌟 Object Detection with YOLO**

<p align="justify">
  Welcome to my exploration of <strong>Object Detection with YOLO (You Only Look Once)!</strong> Object detection is a powerful computer vision technique that enables machines to identify and locate objects in an image or video. It not only detects what objects are present but also determines where they are located by drawing bounding boxes around them.
</p>

---

## **📸 What is Object Detection?**

Object detection models follow a structured pipeline to identify objects in images or videos. The process involves:  
1. **Region Proposal Generation**: Proposing regions likely to contain objects.  
2. **Feature Extraction**: Extracting meaningful features from these regions.  
3. **Classification Unit**: Classifying and localizing objects within the regions.  

<p align="center">
  <img src="https://blog.paperspace.com/content/images/2021/01/Fig01.jpg" alt="Object Detection Pipeline" width="600" height="300">
</p>

---

## **🤖 What is YOLO?**

<p align="justify">
  <strong>YOLO (You Only Look Once)</strong> is an advanced object detection algorithm that performs object classification and localization in a single forward pass of the neural network. Unlike traditional approaches that analyze individual image regions separately, YOLO processes the entire image in one go, making it extremely fast and efficient.
</p>

---

### **🛠️ How YOLO Works**

1. **📏 Divides the Image into a Grid**  
   - YOLO splits the input image into a S X S grid.  
   - Each grid cell is responsible for detecting objects whose center falls within the cell.  

2. **🔮 Predicts for Each Grid Cell**  
   Each grid cell predicts:  
   - **Bounding Boxes**: Position and size of objects it detects.  
   - **Confidence Score**: Probability that the bounding box contains an object.  
   - **Class Label**: The type of object detected (e.g., car, person, dog).  

3. **🚫 Filters Predictions**  
   - Overlapping bounding boxes are eliminated using **Non-Maximum Suppression (NMS)**, keeping only the most confident predictions.

<p align="center">
  <img src="https://i.imgur.com/dyMH05i.png" alt="YOLO Object Detection Example" width="700" height="300">
</p>

---

## **📐 Difference Between Object Classification, Detection, and Segmentation**

| **Task**                | **Description**                                                                                         | **Output**                                 |
|-------------------------|---------------------------------------------------------------------------------------------------------|-------------------------------------------|
| **Object Classification** | Assigns a single label to the entire image (e.g., "This is a dog").                                      | 🐕 Label only                              |
| **Object Detection**      | Identifies objects and their locations using bounding boxes.                                           | 🐕 ➡️ Box around the object                |
| **Object Segmentation**   | Identifies objects, their locations, and outlines their exact shapes.                                  | 🐕 ➡️ Exact outline of the object          |

---

### **Segmentation Types**

1. **Semantic Segmentation**: Assigns a label to each pixel but doesn't differentiate between multiple instances of the same object.  
2. **Instance Segmentation**: Assigns a label to each pixel and distinguishes between individual objects.

---

## **📊 Metrics Used for Object Detection**

1. **📐 Intersection over Union (IoU)**:  
   Measures how well the predicted bounding box matches the ground truth box.  

   $$
   IoU = \frac{\text{Area of Overlap}}{\text{Area of Union}}
   $$

2. **🎯 Average Precision (AP)**:  
   Combines precision and recall to evaluate performance.  

3. **🌟 Mean Average Precision (mAP)**:  
   Calculates the average of AP across all object classes.

   mAP=Average of AP values for all classes

---

## **🏗️ YOLO Architecture**

YOLO is built using a **Convolutional Neural Network (CNN)** that processes the image.  

- **Input**: Resized image  
- **Feature Extraction**: Extracts patterns like edges, textures, and shapes.  
- **Output**: Bounding boxes, confidence scores, and class labels.

Below is the architecture from the original YOLO paper, *You Only Look Once: Unified, Real-Time Object Detection*.  
The network consists of 24 convolutional layers followed by 2 fully connected layers.  

<p align="center">
  <img src="https://i.imgur.com/9jlUhRj.png" alt="YOLO Architecture" width="700">
</p>  

**Link to Paper**: [YOLO: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

---

## **🌟 Why YOLO Stands Out**

1. **⚡ Real-Time Performance**: Processes images/videos at incredible speeds.  
2. **🎯 High Accuracy**: Maintains a balance between speed and precision.  
3. **🛠️ End-to-End Framework**: Combines detection and classification in one pipeline.

---

## **🚀 Applications of Object Detection**

Object detection has a wide range of real-world applications, including:  

- 🚗 **Self-Driving Cars**: Detecting pedestrians, vehicles, and traffic signs.  
- 🛡️ **Security Systems**: Monitoring surveillance footage and recognizing intruders.  
- 🛒 **Retail**: Tracking inventory or analyzing customer behavior.  
- 🏥 **Healthcare**: Identifying abnormalities in medical images.

---

## **📂 Project Details**

- **Dataset:** Pascal VOC dataset downloaded with the help of the Kaggle API in data.py.
- **Model:** YOLO-based architecture trained for object detection tasks.

---

## **📦 Files in this Repository**

- **data.py:** Script to download and preprocess the Pascal VOC dataset using the Kaggle API.
- **outputs/models/model.keras:** Trained model file for YOLO-based object detection.

---

## **🔗 Links**

- **Data Folder:** [Download Data Folder](https://drive.google.com/drive/folders/1-2WPVeEZDie0KRrUMv-JeHJnf5ZSXaHr?usp=sharing)
- **Model File:** [Download model.keras](https://drive.google.com/file/d/1SsFqwPNs5NV4MTIC-6rrxZG7E0U17EHk/view?usp=sharing)

---

## **📜 How to Use**

1. Clone the repository:

```bash
git clone https://github.com/AnasHasan786/Object-Detection-with-YOLO.git
```

2. Download the dataset and pretrained model:

* Place the dataset in the data/ folder.
* Download and place the model.keras file in the outputs/models/ directory.

3. Run the object detection script:

```bash
python components/model/test.py
```

--- 

## **⚠️ Model Status**

Please note that the model is still under training, and its performance may not be optimal yet. I am working on refining the model for better results.

For now, the provided model (`model.keras`) serves as a work-in-progress demonstration.

--- 

Feel free to use and share! 🚀  
