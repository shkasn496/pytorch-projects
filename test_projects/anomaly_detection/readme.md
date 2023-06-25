Task
- From the MVTec AD dataset, classify images as Good or Anomaly. Also, detect/localize the presence of the anomalies in the images during inference, having no access to ground truth anomaly bounding boxes. 

Dataset
- MVTEC Anomaly Detection Dataset (https://www.mvtec.com/company/research/datasets/mvtec-ad/) includes 15 class types namely 
['Bottle', 'Cable', 'Capsule', 'Carpet', 'Grid', 'Hazelnut', 'Leather', 'Metal Nut', 'Pill', 'Screw', 'Tile', 'Toothbrush', 'Transistor', 'Wood', 'Zipper'].

System Design and Architecture
1. Data 
    a. Preprocessing
    b. Class Imbalance
    c. Data Augmentation
2. Model
    a. Transfer Learning with custom layers
    b. Loss Function
    c. Optimizer
    d. Hyperparameters
3. Training Pipeline
4. Inference Pipeline
5. Evaluation Metrics

1. Data
- Since I'm going to use a Dense Layer as the final layer to my custom model, I need to resize my images to match the requirement of pretrained model size (224,224,3).
- Convert the 1 channel image to 3 channel image as well.



Dataset Resource and License
Attribution
-----------
If you use the dataset in scientific work, please cite:

Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger,
"A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection",
IEEE Conference on Computer Vision and Pattern Recognition, 2019


License
-------
Copyright 2019 MVTec Software GmbH

This work is licensed under a Creative Commons 
Attribution-NonCommercial-ShareAlike 4.0 International License.

You should have received a copy of the license along with this work.
If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.

For using the data in a way that falls under the commercial use clause
of the license, please contact us.


Contact
-------
If you have any questions or comments about the dataset, feel free to
contact us via: paul.bergmann@mvtec.com, fauser@mvtec.com,
sattlegger@mvtec.com, steger@mvtec.com
