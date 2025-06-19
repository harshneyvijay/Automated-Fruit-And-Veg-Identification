import streamlit as st
import tensorflow as tf
import numpy as np

#tensorflow model prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #converting single image to batch
    prediction = model.predict(input_arr)
    return np.argmax(prediction) #retturn index of max element


#sidebar
st.sidebar.title("DASHBOARD")
app_mode=st.sidebar.selectbox("Select Page",["Home","About Project","Prediction"])

#main page
if(app_mode=="Home"):
    st.header("FRUITS AND VEGETABLES RECOGNITION SYSTEM")
    image_path="fruitCbM.jpg"
    st.image(image_path)

#about project
elif(app_mode=="About Project"):
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset encompasses images of various fruits and vegetables, providing a diverse collection for image recognition tasks. The included food items are:")
    st.code("Fruits: Banana, Apple, Pear, Grapes, Orange, Kiwi, Watermelon, Pomegranate, Pineapple, Mango")
    st.code("Vegetables: Cucumber, Carrot, Capsicum, Onion, Potato, Lemon, Tomato, Radish, Beetroot, Cabbage, Lettuce, Spinach, Soybean, Cauliflower, Bell Pepper, Chilli Pepper, Turnip, Corn, Sweetcorn, Sweet Potato, Paprika, Jalapeño, Ginger, Garlic, Peas, Eggplant")
    st.text("")
    st.text("The dataset is organized into three main folders:")
    st.text("1. Train: Contains 100 images per category.")
    st.text("2. Test: Contains 10 images per category.")
    st.text("3. Validation: Contains 10 images per category.")
    st.text("Each of these folders is subdivided into specific folders for each type of fruit and vegetable, containing respective images.")

#prediction page
elif(app_mode=="Prediction"):
    st.header("Predictor")
    test_image = st.file_uploader("Choose an image:")
    if(st.button("Show Image")):
        st.image(test_image, width=4,use_container_width=True)
    #predict button
    if(st.button("Predict")):
        st.write("Our prediction:")
        result_index = model_prediction(test_image)
        #reading labels
        with open("labels.txt") as f:
            content = f.readlines()
        label=[]
        for i in content:
            label.append(i[:-1])
        #st.success("Model is predicting it's a ".format(label[result_index]))





