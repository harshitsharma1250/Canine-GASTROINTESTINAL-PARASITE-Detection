# import streamlit as st
# import tensorflow as tf
# import requests
# from tensorflow.keras.preprocessing.image import img_to_array
# from PIL import Image, UnidentifiedImageError
# import numpy as np

# st.title("Parasite Images Classifier")

# st.markdown("You can upload the image from your local machine.")

# image = st.file_uploader(label="Upload parasite image here", type=['png', 'jpg', 'jpeg'])

# st.markdown("Or paste the URL link here!")

# image_url = st.text_input("Enter the URL of the parasite image:")

# model = tf.keras.models.load_model('ResNet50V2-model.h5')

# class_names = {
#     0: 'Ascariasis',
#     1: 'Babesia',
#     2: 'Capillaria philippinensis',
#     3: 'Enterobius vermicularis',
#     4: 'Epidermophyton floccosum',
#     5: 'Fasciolopsis buski',
#     6: 'Hookworm egg',
#     7: 'Hymenolepis diminuta',
#     8: 'Hymenolepis nana',
#     9: 'Leishmania',
#     10: 'Opisthorchis viverrini',
#     11: 'Paragonimus spp',
#     12: 'Trichophyton rubrum (T. rubrum)',
#     13: 'Taenia spp',
#     14: 'Trichuris trichiura',
# }

# if image is not None:
#     img = Image.open(image)
    
#     st.image(img)
    
#     img = tf.image.resize(np.array(img), (224, 224))
    
#     img_array = img_to_array(img)
#     img_array = np.copy(img_array)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = np.copy(img_array)
#     img_array = tf.keras.applications.resnet_v2.preprocess_input(img_array)

#     predictions = model.predict(img_array)

#     top_classes = np.argsort(predictions.flatten())[-5:][::-1]
#     top_probabilities = predictions.flatten()[top_classes]

#     st.write("Top 5 Predicted Classes:")
#     for class_idx, probability in zip(top_classes, top_probabilities):
#         class_name = class_names[class_idx]
#         st.write(f"{class_name}: {probability:.2%}")

# elif image_url:
#     img = Image.open(requests.get(image_url, stream = True).raw)
        
#     st.image(img)
    
#     img = img.resize((224, 224)) 
    
#     img_array = img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = tf.keras.applications.resnet_v2.preprocess_input(img_array)

#     predictions = model.predict(img_array)

#     top_classes = np.argsort(predictions.flatten())[-5:][::-1]
#     top_probabilities = predictions.flatten()[top_classes]

#     st.write("Top 5 Predicted Classes:")
#     for class_idx, probability in zip(top_classes, top_probabilities):
#         class_name = class_names[class_idx]
#         st.write(f"{class_name}: {probability:.2%}")
            
# else:
#     st.write("Upload an Image")




import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, UnidentifiedImageError



model = tf.keras.models.load_model('ResNet50V2-model.h5')


class_names = {
    0: 'Ascariasis',
    1: 'Babesia',
    2: 'Capillaria philippinensis',
    3: 'Enterobius vermicularis',
    4: 'Epidermophyton floccosum',
    5: 'Fasciolopsis buski',
    6: 'Hookworm egg',
    7: 'Hymenolepis diminuta',
    8: 'Hymenolepis nana',
    9: 'Leishmania',
    10: 'Opisthorchis viverrini',
    11: 'Paragonimus spp',
    12: 'Trichophyton rubrum (T. rubrum)',
    13: 'Taenia spp',
    14: 'Trichuris trichiura',
}


if 'history' not in st.session_state:
    st.session_state.history = {"Precision": [], "Recall": [], "F1 Score": []}
    
epochs = range(1, 11)
train_loss = [1.9140, 1.2486, 1.0463, 0.9223, 0.8246, 0.7627, 0.7083, 0.6554, 0.6380, 0.5961]
val_loss = [2.7678, 4.3666, 5.8075, 2.1025, 2.3482, 1.3904, 0.6803, 1.0908, 0.5010, 0.5381]
train_accuracy = [0.4263, 0.5846, 0.6522, 0.6904, 0.7198, 0.7437, 0.7590, 0.7769, 0.7836, 0.7967]
val_accuracy = [0.1826, 0.1830, 0.3300, 0.4416, 0.4643, 0.6171, 0.7838, 0.6667, 0.8296, 0.8207]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b-o', label='Training Loss')
plt.plot(epochs, val_loss, 'r-o', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy, 'b-o', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r-o', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
st.pyplot(plt)

st.title("Parasite Classifier with Top-5 Predictions & Metrics Visualization")
st.markdown("You can upload an image or provide an image URL to classify and visualize graphs.")


image = st.file_uploader(label="Upload parasite image here", type=['png', 'jpg', 'jpeg'])

if image:
    try:
        
        img = Image.open(image)
        st.image(img, caption="Uploaded Image")

        
        img = tf.image.resize(np.array(img).astype(np.float32), (224, 224))
        img_array = img_to_array(img).astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.resnet_v2.preprocess_input(img_array)

        
        predictions = model.predict(img_array)
        predicted_probs = predictions[0]  

        
        top_5_idx = np.argsort(predicted_probs)[-5:][::-1]
        top_5_classes = [(class_names[idx], predicted_probs[idx]) for idx in top_5_idx]

        st.subheader("Top 5 Predicted Classes")
        for class_name, prob in top_5_classes:
            st.write(f"{class_name}: {prob:.2%}")

        
        st.subheader("Select Ground Truth Class for Metrics Comparison:")
        true_class = st.selectbox(
            "Choose the actual class (ground truth) to compute metrics:",
            options=list(class_names.values()),
        )

        
        true_class_idx = [idx for idx, name in class_names.items() if name == true_class][0]

        
        predicted_class_idx = np.argmax(predicted_probs)  

        
        
        true_labels = np.zeros_like(predicted_probs)
        true_labels[true_class_idx] = 1  

        predicted_binary = np.zeros_like(predicted_probs)
        predicted_binary[predicted_class_idx] = 1  

        
        precision = precision_score(true_labels, predicted_binary, zero_division=0)
        recall = recall_score(true_labels, predicted_binary, zero_division=0)
        f1 = f1_score(true_labels, predicted_binary, zero_division=0)

        
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1 Score: {f1:.2f}")

        
        sns.set_style("whitegrid")
        fig, ax = plt.subplots()
        sns.barplot(x=["Precision", "Recall", "F1 Score"], y=[precision, recall, f1], ax=ax, palette="viridis")
        ax.set_ylabel("Scores")
        ax.set_title("Model Metrics Visualization")
        st.pyplot(fig)

        
        st.session_state.history["Precision"].append(precision)
        st.session_state.history["Recall"].append(recall)
        st.session_state.history["F1 Score"].append(f1)

        
        st.subheader("Metrics Over Time")
        if len(st.session_state.history["Precision"]) > 1:
            plt.figure(figsize=(12, 6))
            plt.plot(st.session_state.history["Precision"], label="Precision", marker='o')
            plt.plot(st.session_state.history["Recall"], label="Recall", marker='o')
            plt.plot(st.session_state.history["F1 Score"], label="F1 Score", marker='o')
            plt.xlabel('Upload Index')
            plt.ylabel('Scores')
            plt.title('Metrics Over Multiple Image Uploads')
            plt.legend()
            st.pyplot(plt)
        else:
            st.write("Upload multiple images to see metrics over time.")
            

    except UnidentifiedImageError:
        st.error("The uploaded file could not be processed. Please upload a valid image file.")
else:
    st.write("Please upload an image to proceed.")















