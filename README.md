# florescence-recognition
Recognize the florescence of corn in digital images.  
There're three stages of florescence for corn: non-bloomed, partially-bloomed and fully-bloomed. The features of flowers in different florescences, such as shape and color, can be useful in automatic recognition.  
The code is based on OpenCV3.10.

### Process
1. Compute the descriptors from train images with SIFT.
2. Cluster the descriptors with k-means and get the vocabularies.
3. Compute the bag of words(BoW) vector in the training set for each class.
4. Train classifiers with SVM for each class.
5. Compute the descriptors for each image in the test set.
6. Implement the classifiers on the test dataset.
7. Output the results to files and screen.

### Results
The results are in `data/results.txt`. The accuracy is average above 0.7. Besides, process data of vocabularies, SVMs, BoWs, etc is also in folder `data`.

### Others
SIFT is used to detect features.  
In `vocabulary.h`, there is a variable named `vocab_size` for modifying performance, and I choose 33 after experiments.  
There are 50 images in each florescence. By default, 20 of them will be randomly picked to train the classifier, and others are to test the classifier. A variable named `train_pic_num_each_class` in `image.h` is for this.