# Automated-Fruit-And-Veg-Identification
In many agricultural, retail, and food processing environments, the manual identification and sorting of fruits and vegetables is time-consuming, labour-intensive, and prone to human error. As the demand for efficiency, consistency, and scalability in produce handling grows, there is a critical need for an automated, accurate system capable of identifying fruits and vegetables using image data.

Deepfruitveg addresses this need by leveraging advanced deep learning techniques to automatically classify various types of fruits and vegetables based on their visual features. 

The project is divided into three key phases:
1. Training Phase (Training_fruit_vegetable.ipynb): A convolutional neural network (CNN) is trained on a labelled dataset of fruit and vegetable images. This model learns to extract and understand complex features such as shape, colour, and texture.
2. Testing and Deployment Phase (testingmodel.ipynb): The trained model is evaluated on unseen test images to validate its performance. It is then used for real-time classification, simulating its deployment in real-world scenarios like sorting lines, supermarket inspections, or crop health monitoring systems.
3.  Web Application (Flask-Based): To make the model accessible and user-friendly, a web application was developed using Flask. It features a clean, responsive interface built with Bootstrap, organized into pages for Home, Prediction, and About. The backend integrates the trained deep learning model using TensorFlow, this interface is ideal for deployment in agricultural, retail, or educational settings.

demo: https://drive.google.com/file/d/14HypDtGkuTmAnnDtk0sDyx20Jtcobm2i/view?usp=sharing
