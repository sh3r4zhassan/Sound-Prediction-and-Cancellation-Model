# Sound Classification and Cancellation System
In the first step for our system, we used a CNN model which can classify different sound noises. The second important part in our system is the Simulink simulation which takes the noise as an input and gives an output with noise minimized. In the end, a python script was used as a bridge to join both to obtain one complete system and the details can be seen below
## 1-CNN Model:
We implemented the Deep CNN model described in Salamon and Bello's paper for Environmental Sound Classification on Urbansound8k dataset and achieved state of the art accuracy in the results. The model can classify any Noise to almost an accuracy of 86 percent. Complete details of this model along with its python files can be found here:
(https://github.com/Sheraz-hassan/Deep-CNN-for-Environmental-Sound-Classification)

## 2-Simulink ANC model:
This Simulink ANC model takes a noise as input and adjusts its weight gradually so to reduce the output noise to its minimum. The ANC model used can be found here:
(https://github.com/Sheraz-hassan/Active_Noise_Cancellation_Simulink)

### Training the ANC model:
This model was pretrained for our system on all the type of classes that were available in the Urbansound8K dataset and these pretrained weights for each class were saved. These weights are now ready to use to cancel any sound that enters the system without requiring to train and gradual cancellation of the sound. Complete details on how we trained the model are documented in “Readme” file inside the “ANC Model Training” folder.

## 3-Python Script:
The python script works as a connecting link between the CNN model and the Simulink ANC model. It takes the audio clip as an input and passes it through the pretrained CNN model to get a classification output. Make sure the sound is in the format of “Noise.wav” in the parent folder. The CNN model is the first step of our system. After we can distinguish the input noise among the between different types of noises, we pass it through our Simulink ANC model which has pretrained weights saved for every class. This makes the simulation ready to use at that very instance without the need of updating its weight. So, once we get the classification, the pretrained corresponding weights gets uploaded to the Simulink model automatically and the system output a sound along with the graphs of input and the output. From the example below, we can see that the output noise (blue) is minimized when compared to the input (Yellow) from the very start which was possible only due to pretrained weights for every separate class.

 The script also allows the user to save new trained weights in a new file for future use without effecting the pre-trained weights. In the end, Sound level pressure (SLP) of both the input and the output sounds appear on a plot for comparison.

## Requirements:
Other than the python requirnments, following should be installed on the system for the code to work
1-MATLAB and Simulink
2-Certain Simulink toolboxes (Refer to the slx file to check all toolboxes requirements)
