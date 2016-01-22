# florescence-recognition
Recognize the florescence of corn in digital images.  
There're three stages of florescence for corn: non-bloomed, partially-bloomed and fully-bloomed. The features of flowers in different florescences, such as shape and color, can be useful in automatic recognization.  
The code is based on OpenCV3.10.

### Pre-process for images
Considering green leaves as a large part of the images is useless background for recognization, a pre-process for removing green part of the images may enhance performance. Besides, red-and-white sticks and soil are removed too.

### Others
SIFT is used to detect features.  
In `vocabulary.h`, there is a variable named `vocab_size` for modifying performance, and I choose 33 after experiments.  
There are 50 images in each florescence. By default, 20 of them will be randomly picked to train the classifier, and others are to test the classifier. A variable named `train_pic_num_each_class` in `image.h` is for this.