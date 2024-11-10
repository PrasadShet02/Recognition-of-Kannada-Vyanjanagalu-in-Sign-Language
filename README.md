# Kannada Vyanjanagalu Recognition Project

This project aims to efficiently recognize Kannada Vyanjanagalu with higher accuracy and reduced inference time using machine learning techniques.

## Abstract

Sign language serves as a means of communication for speech and hearing-impaired individuals. This project focuses on the real-time recognition of Static Kannada Sign Language, Vyanjanagalu, utilizing machine learning. The dataset comprises 3,400 images representing 34 static signs of Kannada Vyanjanagalu. Various machine learning models were explored, with the Gated Recurrent Units(GRU) and Support Vector Machine(SVM) achieving an accuracy rate above 99%. This technology aims to enhance the quality of life for the hearing-impaired population by improving communication through sign language recognition.

## Dataset

The curated dataset consists of 3,400 static images depicting 34 distinct signs of Kannada Vyanjanagalu, each with 100 images. These signs involve single-hand gestures captured at 640 x 480 pixels resolution. The dataset's validation was confirmed by the Association of People with Disabilities(APD) in Bengaluru.

## System Architecture
![workFlow](https://github.com/mohan-0709/Kannada-Sign-Language-Recognition/assets/79490917/3c9943d8-1f8e-416b-9b57-89541419410c)

### Feature Extractor

The project utilizes Mediapipe Hands, a tool developed by Google, for extracting hand landmarks. This tool precisely identifies 21 landmarks per hand gesture under various lighting conditions, providing normalized x, y, z coordinates. The output is organized into numpy arrays, each adopting a (100, 21, 3)-dimensional matrix configuration, optimizing computational efficiency and facilitating data management.

### Models and Results

![confusion_matrix](https://github.com/mohan-0709/Kannada-Sign-Language-Recognition/assets/79490917/b4e9337e-6f43-4c1c-8048-695a0d45db5f)
![average_precision](https://github.com/mohan-0709/Kannada-Sign-Language-Recognition/assets/79490917/67267df5-ab5f-4a25-8d18-0d153535e09c)


Several machine learning models were evaluated, with GRU and SVM achieving an accuracy rate above 99%.

### Real-time Output

![ಕ](https://github.com/mohan-0709/Kannada-Sign-Language-Recognition/assets/79490917/ab730ba1-3b67-48e0-88ce-d2a339f40657)
![ಫ](https://github.com/mohan-0709/Kannada-Sign-Language-Recognition/assets/79490917/2f7c9a2d-594b-4a7c-8e44-5e48d5153c9b)

https://github.com/mohan-0709/Kannada-Sign-Language-Recognition/assets/79490917/dc76b050-b7f7-4d6b-97a6-89028dbf7170


### How to Run

1. Install Python, Mediapipe, OpenCV, tkinter and other required libraries on your system.
2. Create a folder named `Dataset` and store all the images as shown in the `Dataset` folder in this repository.
3. `hfe.py` is a feature extractor file. Run this file to extract all the features and store them in the `FEATURES` folder.
4. Inside the `Models` folder, you will find various models implemented on this dataset. Run the Python file of the respective model.
5. Run the `real_time.py` file to see the model working and identifying signs in real-time. Make sure to change the model path for your respective model in this file.


## Conclusion

The project contributed a comprehensive dataset and explored multiple machine learning models, achieving exceptional accuracy in Kannada Vyanjanagalu recognition. Notable achievements include reduced inference time, real-time capability, adaptability to lighting conditions, and robustness against occlusions. The technology demonstrates proficiency in recognizing complex signs, accommodating multiple signers, and enabling translation to the Kannada language. The implementation of real-time recognition exemplifies practical applications, laying the groundwork for robust sign language recognition technologies.

This research aims to significantly advance technologies benefiting individuals with hearing impairments by facilitating accurate and efficient recognition of sign languages.
