# plot feature map of first conv layer for given image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims

#from keras.preprocessing.image import load_img
#from keras.preprocessing.image import img_to_array

from tensorflow.keras.utils import load_img 
from tensorflow.keras.utils import img_to_array
from matplotlib.backends.backend_pdf import PdfPages

# load the model
model = VGG16()

# redefine model to output right after the first hidden layer
model = Model(inputs=model.inputs, outputs=model.layers[1].output)
model.summary()

paths = ['bird.jpg','pilares.jpg']

with PdfPages('filtros.pdf') as pdf:
	for path in paths:
  	# load the image with the required shape
		img = load_img(path, target_size=(224, 224))
		# convert the image to an array
		img = img_to_array(img)
		# expand dimensions so that it represents a single 'sample'
		img = expand_dims(img, axis=0)
		## prepare the image (e.g. scale pixel values for the vgg)
		img = preprocess_input(img)
		# get feature map for first hidden layer
		feature_maps = model.predict(img)

		square = 8
		ix = 1
		for i in range(square):
			for j in range(square):
				fig, ax = pyplot.subplots(figsize=(8, 6), dpi=100)
				ax.set_xticks([])
				ax.set_yticks([])
				# plot filter channel in grayscale
				pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
				pyplot.title("("+str(i+1)+','+str(j+1)+') '+path, fontsize=18)
				pdf.savefig(fig)
				pyplot.close()
				ix += 1
