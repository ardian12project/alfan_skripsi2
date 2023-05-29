import pandas as pd
import streamlit as st
import numpy as np
import tensorflow as tf
import base64
from PIL import Image , ImageOps
from tensorflow.compat.v1.keras import backend as K



@st.cache(allow_output_mutation=True)
def load_model_catsdogs():

	model = tf.keras.models.load_model('alfan-pest-85.26.h5', compile=False)
	# model.compile(loss='binary_crossentropy',
    #           optimizer="Adam",
    #           metrics=['accuracy'])

	#model._make_predict_function()
	model.summary()  # included to make it visible when model is reloaded
	session = K.get_session()
	return model,session

@st.cache(allow_output_mutation=True)
def gift():

		file_ = open("catdog.gif", "rb")
		contents = file_.read()
		data_url = base64.b64encode(contents).decode("utf-8")
		file_.close()
		return data_url


def image_upload(img):
		
		images = Image.open(img)
		st.image(images, caption='Uploaded Image.',width= 250)
		st.write("")
		st.write("Classifying...")
		return images
		
def Model(img, model):
		images = img
		data = np.ndarray(shape=(1,150, 150, 3), dtype=np.float32)
		im = ImageOps.fit(images, (150,150))
		image_array = np.asarray(im)
		data[0] = image_array

		result = model.predict(data)
		output = np.argmax(result)

		if output == 0:
			prediction = 'Wanita Melanesia'
			var = '👩'
		elif output == 1:
			prediction = 'Laki-Laki Melanesia'
			var = '👨'
		elif output == 2:
			prediction = 'Wanita Non Melanesia'
			var = '👩'	
		elif output == 3:
			prediction = 'Laki-Laki Non Melanesia'
			var = '👨'
		st.title(var)
		st.write("hasil Prediksi yaitu :", prediction)


  
def main():
	st.title("Image Classification with Convolution Neural Network")
	analysis = st.sidebar.selectbox("Menu", ["Problem", "Female v/s Male"])
	st.sidebar.info("""
	Developer : Alfandris Tatinting         
	NIM		  : 201855202039""")
#=======================================================================================================================================
# EXPLAINING THE CNN	
#=======================================================================================================================================	
	if analysis == "Problem":

		st.header("Female vs Male - A Binary classification problem")
		st.text(" Ini adalah web untuk mendeteksi sebuah gambar apakah gambar seorang wanita atau pria ")
		st.image('catdog12.jpg', width= 700, use_column_width=True)
		
		# st.subheader("CNN Model")

		data_url = gift()
		# st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',unsafe_allow_html=True)

#=======================================================================================================================================	
# CATS & DOGS MODEL
#=======================================================================================================================================			
	elif analysis == "Female v/s Male":	
		st.header("Classification Model")
		st.subheader("Problem Type: Female V/S Male Classifier")
		model, session= load_model_catsdogs()		
		K.set_session(session)
		uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
		if uploaded_file is not None:
			test_image= image_upload(uploaded_file)
			Model(test_image,model)

main()