#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import os
import librosa
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from playsound import playsound
from librosa.feature import melspectrogram


# In[3]:


# filename = sys.argv[1]
filename='Noise.wav'


# In[4]:


f, sr = librosa.load(filename,sr=44100)
actual_audio_length=len(f)/sr
three_sec_samples=3*sr
if(len(f)>=three_sec_samples):
    log_mel_spec = librosa.power_to_db(melspectrogram(f[:three_sec_samples], sr=sr, n_fft=1034, hop_length=1034))
    # Log Mel spectograms of first 3 secs 
else:                               
    #If audio is not 3 sec repeat pad until its 3 secs long
    while(len(f)<three_sec_samples):          
        f=np.concatenate((f, f))
    log_mel_spec = librosa.power_to_db(melspectrogram(f[:three_sec_samples], sr=sr, n_fft=1034, hop_length=1034))
CNN_model = tf.keras.models.load_model('CNN_model.h5')
predicted_class=np.argmax(CNN_model.predict(np.array([log_mel_spec])))


# In[5]:


predicted_class


# In[6]:


print("Playing and identifying sound....")
playsound(filename)


# In[7]:


classes = ["Air Conditioner","Horn","Children","Dog","Drill","Engine","Gun","Hammer","Siren","Music"]
print("Sound Identified as '{}' sound".format(classes[predicted_class]))


# In[8]:


actual_audio_length





# #  Simulink start here

# In[10]:


ANC_Model='ANC_Model.slx'
pretrainedweights='ANC Model Training/Pre-Trained Weights/Pretrained_weights_Class_{}.txt'.format(predicted_class)
newtrainedweights='New-Trained Weights/New-Trained_weights_Class_{}.txt'.format(predicted_class)


# In[11]:


yourtrained = 0
pretrained  = 0
print('Do you want to use your previously saved weights for this class?\nEnter 1 for "YES" and 0 for "NO"')
yourtrained = int(input())
if yourtrained!=1 and yourtrained!=0:
    while(yourtrained!=1 and yourtrained!=0):                 
        print('Invalid Input. Enter 1 or 0')
        yourtrained = int(input())
        
if (yourtrained == 0):
    print('Using Pre-Trained weights')
    pretrained = 1

print('Do you want to update your previously saved weights for this class based on this simulation?')
print('Note: Pretrained weights will not be effected by this\nEnter 1 for "YES" and 0 for "NO"')
update = int(input())
if update!=1 and update!=0:
    while(update!=1 and update!=0):                 
        print('Invalid Input. Enter 1 or 0')
        update = int(input())


# In[12]:


simulation_time=actual_audio_length


# In[13]:


update


# In[14]:


pretrained


# In[15]:


import matlab.engine
eng = matlab.engine.start_matlab()


# In[16]:


if (yourtrained == 1):
    fileID = eng.fopen(newtrainedweights,'r')
    inputweights = eng.fscanf(fileID,'%f')
else:
    fileID = eng.fopen(pretrainedweights,'r')
    inputweights = eng.fscanf(fileID,'%f')


# In[17]:


eng.workspace['inputweights']=inputweights
eng.workspace['gain'] = 1.0
eng.workspace['volume'] = 100
eng.workspace['timeout'] = simulation_time


# In[18]:


inputweights


# In[19]:


eng.load_system(ANC_Model)


# In[20]:


print("Output of the system can be heard along with the visualizations as plots which shows both input and output")
print("Yellow plot shows the input sound whereas blue is the system output")


# In[21]:


eng.sim(ANC_Model, nargout =0)
if (update == 1):
    print("Updating Weights....")
    latest_updatedweights = eng.workspace['simout'][-1]
    fid=eng.fopen(newtrainedweights,'w', nargout =1);
    b=eng.fprintf(fid,'%.15f\n',latest_updatedweights)
    eng.fclose(fid, nargout =0)
    print("Weights Updated")


# In[22]:


outputsound=np.array(eng.workspace['outsound'])


# In[23]:


eng.quit()


# In[24]:


realsound, _ = librosa.load(filename,sr=44100)
ref_value=0.00002 #ref_value here is the lowest possible noise humans can hear  
abs_realsound=abs(realsound)
abs_realsound[abs_realsound<ref_value]=ref_value  #no value should be less than the refrence value i.e refrence value = 0dB
SPL_real=np.sum(20*np.log10(abs_realsound/0.00002))/len(abs_realsound)
SPL_real


# In[25]:


abs_output=abs(outputsound)
abs_output[abs_output<ref_value]=ref_value
SPL_output=np.sum(20*np.log10(abs_output/0.00002))/len(abs_output)
SPL_output


# In[26]:


df = pd.DataFrame({'group':list(('SPL BEFORE ANC','SPL AFTER ANC')), 'values':list((SPL_real,SPL_output)) })
 
# Reorder it based on the values
ordered_df = df.sort_values(by='values')
my_range=range(1,len(df.index)+1)
 
# The horizontal plot is made using the hline function
plt.hlines(y=my_range, xmin=0, xmax=ordered_df['values'], color='red')
plt.plot(ordered_df['values'], my_range, "o")
 
# Add titles and axis names
plt.yticks(my_range, ordered_df['group'])
plt.title("SPL OF NOISE")
plt.xlabel('SPL (dB)')


# Show the plot
plt.show()


# In[ ]:




