# Hand_Gesture_project_repo

A real-time hand gesture recognition system using **MediaPipe** for landmark detection and **Scikit-learn** for machine learning classification.

## Features
- **Real-time Detection:** Recognizes gestures using an input video and by using my webcam via OpenCV and MediaPipe.
- **Smart Preprocessing:** Hand landmarks are centered and scaled relative to the wrist to ensure the model works regardless of hand position or distance from the camera.
- **Ensemble Learning:** Uses a majority vote between multiple models (KNN, SVM, RandomForest, AdaBoost) for higher robustness and accuracy.

##  Tech Stack
- **Python**: Core programming language(v. 3.10.11 as mediapipe requests this version).
- **MediaPipe**: For hand landmark detection(v. 0.10.5).
- **OpenCV**: For webcam video processing.
- **Scikit-learn**: For training machine learning models.
- **Joblib**: For saving and loading trained models.

##  Installation
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/karimabdelnasser39/Hand-Gesture-Classification.git](https://github.com/karimabdelnasser39/Hand-Gesture-Classification.git)
    cd Hand-Gesture-Classification
    ```
2.  **Install dependencies:**
    ```bash
    pip install mediapipe opencv-python joblib pandas numpy scikit-learn matplotlib
    ```

##  Usage
1.  **Data Collection (Optional):** Run a script to generate your `data.csv` if you wish to train on your own gestures.
2.  **Training:** Open `my_project.ipynb` and run the cells to train the models and generate the `.pkl` files.
3.  **Input Video** Create a video that contains all the hand gestures that passed to the script.
4.  **Live Detection:** Run the live prediction script:
    *Press 'q' to exit the webcam feed.*

##  Project Workflow
The notebook follows these steps:
1.  **Data Loading**: Loads landmark coordinates and labels from `data.csv`.
2.  **Preprocessing**: Centers coordinates around the wrist and scales them for invariance.
3.  **Model Training**: Trains KNN, SVM, Random Forest, and AdaBoost classifiers.
4.  **Live Inference**: Processes webcam frames in real-time, applies preprocessing, and uses a voting mechanism for final classification.
