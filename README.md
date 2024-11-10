# Kannada Vyanjanagalu Recognition Project

This project aims to efficiently recognize Kannada Vyanjanagalu with higher accuracy and reduced inference time using machine learning techniques.

## Abstract

Sign language serves as a means of communication for speech and hearing-impaired individuals. This project focuses on the real-time recognition of Static Kannada Sign Language, Vyanjanagalu, utilizing machine learning. The dataset comprises 3,400 images representing 34 static signs of Kannada Vyanjanagalu. Various machine learning models were explored, with the Gated Recurrent Units(GRU) and Support Vector Machine(SVM) achieving an accuracy rate above 99%. This technology aims to enhance the quality of life for the hearing-impaired population by improving communication through sign language recognition.

## Dataset

The curated dataset consists of 3,400 static images depicting 34 distinct signs of Kannada Vyanjanagalu, each with 100 images. These signs involve single-hand gestures captured at 640 x 480 pixels resolution. The dataset's validation was confirmed by the Association of People with Disabilities(APD) in Bengaluru.

## System Architecture
![System Arch](https://github.com/user-attachments/assets/40a3be41-465e-493d-bde1-d38ad433ddeb)


### Feature Extractor

The project utilizes Mediapipe Hands, a tool developed by Google, for extracting hand landmarks. This tool precisely identifies 21 landmarks per hand gesture under various lighting conditions, providing normalized x, y, z coordinates. The output is organized into numpy arrays, each adopting a (100, 21, 3)-dimensional matrix configuration, optimizing computational efficiency and facilitating data management.

### Models and Results

![confusion matrix](https://github.com/user-attachments/assets/e85057e2-8cf5-4478-9098-74de8a857f3f)

![precision](https://github.com/user-attachments/assets/5b412df6-a8bc-4312-870b-afeb8fe4968d)


Several machine learning models were evaluated, with GRU and SVM achieving an accuracy rate above 99%.


### Real-time Output

![output1](https://github.com/user-attachments/assets/af436afd-7090-4d98-b93d-8ae142837a55)

![output2](https://github.com/user-attachments/assets/867387dd-4774-4349-8fd8-40b53e0592de)


https://github.com/mohan-0709/Kannada-Sign-Language-Recognition/assets/79490917/dc76b050-b7f7-4d6b-97a6-89028dbf7170 


### How to Run

1. Install Python, Mediapipe, OpenCV, tkinter and other required libraries on your system.
2. Check every file for Path in it and modify it according to your path.
3. Create a folder named `Dataset` and store all the images as shown in the `Dataset` folder in this repository.
4. `hfe.py` is a feature extractor file. Run this file to extract all the features and store them in the `FEATURES` folder.
5. Inside the `Models` folder, you will find various models implemented on this dataset. Run the Python file named with Train.py for the respective model.
6. Run the `real_time.py` file to see the model working and identifying signs in real-time. Make sure to change the model path for your respective model in this file.


## Conclusion

The project contributed a comprehensive dataset and explored multiple machine learning models, achieving exceptional accuracy in Kannada Vyanjanagalu recognition. Notable achievements include reduced inference time, real-time capability, adaptability to lighting conditions, and robustness against occlusions. The technology demonstrates proficiency in recognizing complex signs, accommodating multiple signers, and enabling translation to the Kannada language. The implementation of real-time recognition exemplifies practical applications, laying the groundwork for robust sign language recognition technologies.

This research aims to significantly advance technologies benefiting individuals with hearing impairments by facilitating accurate and efficient recognition of sign languages.
