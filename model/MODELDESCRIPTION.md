# AIHEALTHLUNGCANCER MODEL DESCRIPTION

## About the model
CNNs are models used in image processing that mimic how the human brain recognizes images. It “scans” the images using a process called convolution and then classifies them. This makes it useful for classifying CT scans with or without lung cancer because CT scans are images. 

Our model is a convolutional neural network that has 2 convolutional layers and 1 output (fully connected) layer. Layers consist of neurons, and our model has 192 neurons in total. 

## Results
In our model, we got to a testing accuracy of over 99% and a validation accuracy of 82.4%. These are very promising results that clearly demonstrate the potential AI has in the field of healthcare.
However, there is a caveat to the results. These results were achieved with a relatively small dataset of **1587 images from 61 patients**. There is a high chance the model is hypertuned to the data set that we repeatedly trained it on, so accuracy may not be as high in actual application. To improve on this issue, we are looking to training the model with a larger 124GB dataset.

## Model
The final hypertuned and fully developed model is in the MODEL folder for your viewing.
