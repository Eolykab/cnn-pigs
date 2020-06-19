# Load model
from keras.models import load_model

model = load_model('model.h5')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

import numpy

from keras.preprocessing import image

img_width = 64
img_height = 64
test_image = image.load_img('pig.jpeg', target_size=(img_width, img_height))

# convert input image in input format that model accepts
test_image = image.img_to_array(test_image)
test_image = numpy.expand_dims(test_image, axis=0)

# test_image = test_image.reshape(img_width, img_height)

# 0=Pig
# 1=Not a pig
result = model.predict(test_image)
if result[0][0] == 0:
    print('Pig')
else:
    print('Not a pig')
