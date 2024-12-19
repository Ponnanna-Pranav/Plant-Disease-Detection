import streamlit as st
import tensorflow as tf
import numpy as np

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("Trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(256, 256))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return predictions

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-image: url('https://www.example.com/farm_background.jpg');
        background-size: cover;
        background-position: center;
    }
    .sidebar .sidebar-content {
        background-color: #f0f0f0;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: black;
        text-align: center;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("Farm Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.title("🌾 Plant Disease Recognition System 🌿")
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

elif app_mode == "About":
    st.title("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
    This dataset consists of about 87K RGB images of healthy and diseased crop leaves which are categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
    A new directory containing 33 test images is created later for prediction purposes.
    
    #### Content
    1. Train (70295 images)
    2. Test (33 images)
    3. Validation (17572 images)
    """)

elif app_mode == "Disease Recognition":
    st.title("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    
    if test_image:
        st.image(test_image, use_column_width=True)
        
        if st.button("Predict"):
            st.write("Analyzing the image...")
            predictions = model_prediction(test_image)
            result_index = np.argmax(predictions)
            confidence = np.max(predictions)
            
            # Reading Labels
            class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                           'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                           'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                           'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                           'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                           'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                           'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                           'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                           'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                           'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                           'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                           'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                           'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                           'Tomato___healthy']

            # Threshold for confidence
            confidence_threshold = 0.5
            if confidence > confidence_threshold:
                st.success(f"Model is predicting it's a {class_names[result_index]}")
            else:
                st.error("The image is not a plant or is not recognized by the model. Please try again with a different image.")

# Add a footer
st.markdown(
    """
    <div class="footer">
        <p>© 2024 Plant Disease Recognition System. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True
)
