# Real-Time Facial Emotion Recognition using CNN

A deep learning project that recognizes human facial emotions in real time using a custom Convolutional Neural Network (CNN) built with **PyTorch** and **OpenCV**.

## Features

* Facial emotion classification using CNN
* Recognizes 7 emotions:

  * Angry
  * Disgust
  * Fear
  * Happy
  * Neutral
  * Sad
  * Surprise
* Real-time face detection using OpenCV
* Real-time emotion prediction through webcam
* Model evaluation using standard classification metrics
* Saves and loads the trained model

## Dataset

This project uses the **FER-2013 Facial Expression Dataset**.

The dataset contains grayscale facial images belonging to seven emotion classes.

The dataset is not included in this repository.

[FER-2013 Dataset on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

## Evaluation

The project includes an evaluation script that calculates the major classification metrics:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Run the evaluation with:

```bash
python evaluate.py
```

The evaluation requires the trained model and test dataset.

## Real-Time Detection

The trained model can be used to recognize emotions through a webcam.

Run:

```bash
python webcam_test.py
```

The application detects faces from the webcam feed and displays the predicted emotion.

## Technologies

* Python
* PyTorch
* OpenCV
* NumPy
* Pillow
* Scikit-learn
* Matplotlib

## Training

To train the model:

```bash
python face.py
```

The trained model is saved as `emotion_cnn.pth`.

## Results

Evaluation metrics can be generated using `evaluate.py`.

The original trained model checkpoint is not currently included in the repository, so numerical evaluation results are not reported here.

## Future Improvements

* Data augmentation
* Improved CNN architecture
* Transfer learning
* Better face detection
* Real-time performance optimization
* More detailed error analysis

## License

This project is intended for educational and research purposes.
