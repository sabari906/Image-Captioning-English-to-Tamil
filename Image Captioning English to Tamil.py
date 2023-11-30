#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pickle
import numpy as np
from tqdm.notebook import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add


# In[3]:


BASE_DIR = r'C:\Users\Sabari\Downloads\archive1'


# In[4]:


with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
    next(f)
    captions_doc = f.read()


# In[5]:


# create mapping of image to captions
mapping = {}
# process lines
for line in tqdm(captions_doc.split('\n')):
    # split the line by comma(,)
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    # remove extension from image ID
    image_id = image_id.split('.')[0]
    # convert caption list to string
    caption = " ".join(caption)
    # create list if needed
    if image_id not in mapping:
        mapping[image_id] = []
    # store the caption
    mapping[image_id].append(caption)


# In[6]:


len(mapping)


# In[7]:


def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
        # take one caption at a time
            caption = captions[i]
            # preprocessing steps
            # convert to lowercase
            caption = caption.lower()
            # delete digits, special chars, etc., 
            caption = caption.replace('[^A-Za-z]', '')
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            # add start and end tags to the caption
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption


# In[8]:


clean(mapping)


# In[9]:


all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)


# In[10]:


all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)


# In[11]:


len(all_captions)


# In[12]:


# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1


# In[13]:


vocab_size


# In[14]:


# get maximum length of the caption available
max_length = max(len(caption.split()) for caption in all_captions)
max_length


# In[15]:



# create data generator to get data in batch (avoids session crash)
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    # loop over images
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            # process each caption
            for caption in captions:
                # encode the sequence
                seq = tokenizer.texts_to_sequences([caption])[0]
                # split the sequence into X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pairs
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store the sequences
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
                if n == batch_size:
                    X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                    yield [X1, X2], y
                    X1, X2, y = list(), list(), list()
                    n = 0


# In[16]:


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# In[17]:


model_path = r"C:\Users\Sabari\OneDrive\Desktop\image\report\vgg,gru,adam\main h5 file\main.h5"


# In[18]:


from tensorflow.keras.models import load_model

# Define the path to the model file
model_path = r"C:\Users\Sabari\OneDrive\Desktop\image\report\vgg,gru,adam\main h5 file\main.h5"

# Load the model from the .h5 file without specifying the optimizer
model2 = load_model(model_path, compile=False)


# In[19]:


vgg_model = VGG16() 
# restructure the model
vgg_model = Model(inputs=vgg_model.inputs,             
                  outputs=vgg_model.layers[-2].output)


# In[ ]:





# In[ ]:





# In[20]:


import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


# In[21]:


max_length=35


# In[22]:


def predict_caption(image_path):
    # Load the image
    image = load_img(image_path, target_size=(224, 224))
    # Convert image pixels to numpy array
    image = img_to_array(image)
    # Reshape data for model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # Preprocess image from VGG
    image = preprocess_input(image)
    # Extract features
    feature = vgg_model.predict(image, verbose=0)
    # Predict from the trained model
    caption = generate_caption(model2, feature, tokenizer, max_length)
    return caption


# In[23]:


def generate_caption(model, photo, tokenizer, max_length):
    # Start the caption with the start sequence token
    in_text = 'startseq'
    # Iterate until the maximum caption length is reached or the end sequence token is generated
    for _ in range(max_length):
        # Encode the input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Pad the input sequence
        sequence = pad_sequences([sequence], maxlen=max_length)
        # Predict the next word
        yhat = model2.predict([photo, sequence], verbose=0)
        # Convert the predicted word index to a word
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        # Break if end sequence token is generated
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
    return in_text


# In[24]:


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# In[25]:


def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        caption = predict_caption(file_path)
        caption_label.config(text=caption)


# In[29]:



from googletrans import Translator

def translate_to_tamil(text):
    translator = Translator(service_urls=['translate.google.com'])
    translation = translator.translate(text, dest='ta')
    return translation.text


# In[31]:


import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import os
import pandas as pd

# Initialize a list to store all the captions
all_captions = []

def open_image():
    # Open a file dialog to select an image
    image_path = filedialog.askopenfilename()

    # Extract the image name from the path
    image_name = os.path.basename(image_path)

    # Load and display the selected image
    with open(image_path, "rb") as f:
        image = Image.open(f)
        image = image.resize((300, 300))  # Adjust the size as needed
        image = ImageTk.PhotoImage(image)
        image_label.configure(image=image)
        image_label.image = image

    # Call the predict_caption function
    caption1 = predict_caption(image_path)
    caption = caption1[8:-7]
    
    # Translate English caption to Tamil (using your translation method/tool)
    tamil_caption = translate_to_tamil(caption)
    print("Tamil Caption:", tamil_caption)
    
    caption_label.configure(text=caption)
    caption_label1.configure(text=tamil_caption)

    # Append the captions to the list
    all_captions.append({
        "Image Name": image_name,
        "English Caption": caption,
        "Tamil Caption": tamil_caption
    })

def save_captions():
    # Check if at least one caption has been generated
    if all_captions:
        # Get the directory path for saving the captions
        save_dir = filedialog.askdirectory()

        # Save the captions in an Excel file
        excel_file_path = os.path.join(save_dir, "captions.xlsx")
        df = pd.DataFrame(all_captions)
        df.to_excel(excel_file_path, index=False)

        # Show a message box indicating successful save
        tk.messagebox.showinfo("Save Successful", "Captions saved successfully in an Excel file.")
    else:
        # Show a message box indicating that no captions have been generated
        tk.messagebox.showwarning("Save Error", "No captions available to save.")

window = tk.Tk()

# Set the window title
window.title("Image Caption Generator")

# Set the window size and position it at the center of the screen
window_width = 1000
window_height = 800
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)
window.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Set a background image
background_image_path = r"C:\Users\Sabari\OneDrive\Desktop\image/07.jpg"  # Replace with the path to your background image
background_image = Image.open(background_image_path)
background_image = background_image.resize((window_width, window_height))
background_image_tk = ImageTk.PhotoImage(background_image)

# Create a label for the background image
background_label = tk.Label(window, image=background_image_tk)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Create a label to display the selected image
image_label = tk.Label(window)
image_label.pack()

# Create an "Open Image" button
open_button = tk.Button(window, text="Open Image", command=open_image)
open_button.pack()

# Create a "Save Captions" button
save_button = tk.Button(window, text="Save Captions", command=save_captions)
save_button.pack()

# Create a label to display the generated caption
caption_label = tk.Label(window, text="Caption will appear here", font=("Helvetica", 14), fg="white", bg="black")
caption_label.pack(pady=20)

caption_label1 = tk.Label(window, text="Caption will appear here", font=("Helvetica", 14), fg="white", bg="black")
caption_label1.pack(pady=20)

# Configure the caption label's design
caption_label.config(relief=tk.RAISED, bd=5, padx=10, pady=10)
caption_label1.config(relief=tk.RAISED, bd=5, padx=10, pady=10)

# Run the Tkinter event loop
window.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:




