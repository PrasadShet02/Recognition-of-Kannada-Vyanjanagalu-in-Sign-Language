import tkinter as tk
from PIL import Image, ImageTk
import threading

dic = {"Ba": "ಬ" , "Bha" : "ಭ", "Cha" :"ಚ" , "Chha" : "ಛ" , "Da":"ದ" , "Daa" : "ಡ" ,
"Dhaa" : "ಢ" , "Dhha" : "ಧ" ,  "Ga": "ಗ" , "Gha":"ಘ" , "Ha" : "ಹ" , "Ja" :"ಜ" ,
"Jha": "ಝ" , "Ka": "ಕ" , "Kha" : "ಖ", "la": "ಲ" , "lla" : "ಳ" , "Ma": "ಮ", 
"Na" : "ಞ" , "Nah": "ನ" , "Nna" : "ಣ" , "Nya" : "ಙ" , "Pa": "ಪ", "Pha": "ಫ",
"Ra": "ರ" , "sa": "ಸ", "sha":"ಷ" ,  "she":"ಶ" , "Ta":"ತ", "Taa": "ಟ" , "Tha": "ಥ",
"Thaa": "ಠ", "va": "ವ","Ya" : "ಯ", "None":"None"}

# Initialize the tkinter window
root = tk.Tk()
root.title("Sign Language Recognition")

# Create a label to display the image
image_label = tk.Label(root)
image_label.pack()

# Create a label to display text with borders and bigger font size
text_label = tk.Label(root, font=("Helvetica", 20), borderwidth=2, relief="solid")
text_label.pack()

# Function to update the window with a new frame and text
def update_window(frame, eng_text):

    global dic
    # Create a PhotoImage object directly from the frame
    frame_img = ImageTk.PhotoImage(Image.fromarray(frame))

    # Update the image label with the new frame
    image_label.config(image=frame_img)
    image_label.image = frame_img

    # Translate English text to Kannada
    kannada_text = dic[eng_text]
        
    # Update the text label with the translated text
    text_label.config(text=kannada_text)

    root.update()  # Update the tkinter window

# Function to start the tkinter main loop in a separate thread
def start_tkinter_mainloop():
    root.mainloop()

# Create a thread to run the tkinter main loop
tkinter_thread = threading.Thread(target=start_tkinter_mainloop)

# Start the tkinter thread
tkinter_thread.start()

# Now, you can call external_update_window() from your external code
