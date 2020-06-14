#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
    An app that detects faces, reads emotions, and displays an avatar with the
    same emotions as the detected face.


    How to run the program:
    1.  run the image collection program with: python3 avatar_emo.py collect
        - take images of your face to use with the training for each emotion
    2.  train the MLP on your images for each emotion or mouth shape
        - do this by running the command: python3 train_classifier.py --data
          <pathToYourData> --save <pathToWhereYouWantToSave>
    3. Run the emotions program with: python3 avatar_emo.py demp --classifier
        <PathToWhereYouSavedTheClassifier>

"""

import argparse     # Library for parsing arguments from user
import cv2          # OpenCV library needed for working with openCV
import numpy as np  # Num py library to work with arrays
import time         # Library to work with time
import boto3        # This is the library for talking to amazon lex

import wx           # Library for working with the wxPython GUI
from pathlib import Path

from data.store import save_datum, pickle_load  # Used to save and read from files
from data.process import _pca_featurize         # Needed to read face features
from detectors import FaceDetector              # Detects a face in the frame
from wx_gui import BaseLayout                   # Get the base layout template from wx

### GLOBAL VARIABLES ###
lastlabel = 'neutral' # Initialize variable for keeping track of emotion state


### This class runs when DEMO mode is selected ###
class FacialExpressionRecognizerLayout(BaseLayout):
    def __init__(self, *args,
                 clf_path=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.clf = cv2.ml.ANN_MLP_load(str(clf_path / 'mlp.xml'))

        self.index_to_label = pickle_load(clf_path / 'index_to_label')
        self.pca_args = pickle_load(clf_path / 'pca_args')

        self.face_detector = FaceDetector(
            face_cascade='params/haarcascade_frontalface_default.xml',

            # Use this one if you don't have glasses
            eye_cascade='params/haarcascade_lefteye_2splits.xml')

            # Use this one if you have glasses
            #eye_cascade='params/haarcascade_eye_tree_eyeglasses.xml')

    def featurize_head(self, head):
        return _pca_featurize(head[None], *self.pca_args)

    def augment_layout(self):
        """Initializes GUI"""
        # initialize data structure
        self.samples = []
        self.labels = []

        pnl4 = wx.Panel(self, -1)
        hbox4 = wx.BoxSizer(wx.HORIZONTAL)
        img = cv2.imread('neutral.jpg',1)
        cv2.imshow('AVATAR', img)

        print(type(img))
        # <class 'numpy.ndarray'>

        print(img.shape)
        # (225, 400, 3)



### Send Responses to Amazon LEX ###
        #txt = input("You can talk to Ella by typing something to her.")

        #response = client.post_text(
        #botName='PizzaOrderingBot',
        #botAlias='MeanPizzaBot',
        #userId='user1',
        #sessionAttributes={
    #        'string': 'string'
    #    },
    #    requestAttributes={
    #        'string': 'string'
    #    },
    #    inputText=txt
    #    )


        # arrange all horizontal layouts vertically
        self.panels_vertical.Add(pnl4, flag=wx.EXPAND | wx.BOTTOM, border=1)

    #    pass
    def update(label):
        global lastlabel
        lastlabel = label
        return lastlabel

    def process_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        success, frame, self.head, (x, y) = self.face_detector.detect_face(
            frame_rgb)
        if not success:
            return frame

        success, head = self.face_detector.align_head(self.head)
        if not success:
            return frame

        # We have to pass [1 x n] array predict.
        _, output = self.clf.predict(self.featurize_head(head))
        label = self.index_to_label[np.argmax(output)]

        # Draw predicted label above the bounding box.
        cv2.putText(frame, label, (x, y - 20),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        global lastlabel

        # Display the appropriate image for the emotion label
        if lastlabel != label:
            cv2.destroyWindow('img')
            if label == 'happy':  #if this was selected, go to collection mode.
                cv2.waitKey(1)
                img = cv2.imread('happy.jpg',1)
                cv2.imshow('AVATAR', img)
                lastlabel = label

            elif label == 'sad':  #if this was selected, go to collection mode.
                cv2.waitKey(1)
                img2 = cv2.imread('sad.jpg',1)
                cv2.imshow('AVATAR', img2)

                cv2.waitKey(100)
                img2 = cv2.imread('sad1.jpg',1)
                cv2.imshow('AVATAR', img2)

                cv2.waitKey(100)
                img2 = cv2.imread('sad2.jpg',1)
                cv2.imshow('AVATAR', img2)

                cv2.waitKey(100)
                img2 = cv2.imread('sad3.jpg',1)
                cv2.imshow('AVATAR', img2)
                lastlabel = label

            elif label == 'surprised':  #if this was selected, go to collection mode.
                cv2.waitKey(1)
                img2 = cv2.imread('surprised.jpg',1)
                cv2.imshow('AVATAR', img2)
                lastlabel = label

            elif label == 'angry':  #if this was selected, go to collection mode.
                cv2.waitKey(1)
                img2 = cv2.imread('angry.jpg',1)
                cv2.imshow('AVATAR', img2)
                lastlabel = label

            elif label == 'disgusted':  #if this was selected, go to collection mode.
                cv2.waitKey(1)
                img2 = cv2.imread('disgusted.jpg',1)
                cv2.imshow('AVATAR', img2)
                lastlabel = label

            elif label == 'neutral':  #if this was selected, go to collection mode.
                cv2.waitKey(1)
                img2 = cv2.imread('neutral.jpg',1)
                cv2.imshow('AVATAR', img2)
                lastlabel = label

        ### If the user is running mouth shapes setting, then display those ###
            elif label == 'AH':  #if this was selected, go to collection mode.
                cv2.waitKey(1)
                img2 = cv2.imread('AH.jpg',1)
                cv2.imshow('AVATAR', img2)
                lastlabel = label

            elif label == 'D':  #if this was selected, go to collection mode.
                cv2.waitKey(1)
                img2 = cv2.imread('D.jpg',1)
                cv2.imshow('AVATAR', img2)
                lastlabel = label

            elif label == 'Ee':  #if this was selected, go to collection mode.
                cv2.waitKey(1)
                img2 = cv2.imread('Ee.jpg',1)
                cv2.imshow('AVATAR', img2)
                lastlabel = label

            elif label == 'F':  #if this was selected, go to collection mode.
                cv2.waitKey(1)
                img2 = cv2.imread('F.jpg',1)
                cv2.imshow('AVATAR', img2)
                lastlabel = label

            elif label == 'L':  #if this was selected, go to collection mode.
                cv2.waitKey(1)
                img2 = cv2.imread('L.jpg',1)
                cv2.imshow('AVATAR', img2)
                lastlabel = label

            elif label == 'M':  #if this was selected, go to collection mode.
                cv2.waitKey(1)
                img2 = cv2.imread('M.jpg',1)
                cv2.imshow('AVATAR', img2)
                lastlabel = label

            elif label == 'oh':  #if this was selected, go to collection mode.
                cv2.waitKey(1)
                img2 = cv2.imread('oh.jpg',1)
                cv2.imshow('AVATAR', img2)
                lastlabel = label

            elif label == 'R':  #if this was selected, go to collection mode.
                cv2.waitKey(1)
                img2 = cv2.imread('R.jpg',1)
                cv2.imshow('AVATAR', img2)
                lastlabel = label

            elif label == 'S':  #if this was selected, go to collection mode.
                cv2.waitKey(1)
                img2 = cv2.imread('S.jpg',1)
                cv2.imshow('AVATAR', img2)
                lastlabel = label

            elif label == 'Uh':  #if this was selected, go to collection mode.
                cv2.waitKey(1)
                img2 = cv2.imread('Uh.jpg',1)
                cv2.imshow('AVATAR', img2)
                lastlabel = label

            elif label == 'Woo':  #if this was selected, go to collection mode.
                cv2.waitKey(1)
                img2 = cv2.imread('Woo.jpg',1)
                cv2.imshow('AVATAR', img2)
                lastlabel = label


        return frame

### This class runs when COLLECTION mode is selected ###
class DataCollectorLayout(BaseLayout):

    def __init__(self, *args,
                 training_data='data/cropped_faces.csv',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.face_detector = FaceDetector(
            face_cascade='params/haarcascade_frontalface_default.xml',
            eye_cascade='params/haarcascade_lefteye_2splits.xml')

        self.training_data = training_data

    def augment_layout(self):
        """Initializes GUI"""
        # initialize data structure
        self.samples = []
        self.labels = []

        ### create a horizontal layout with all buttons for Emotions ###
        pnl2 = wx.Panel(self, -1)
        self.neutral = wx.RadioButton(pnl2, -1, 'neutral', (10, 10),
                                      style=wx.RB_GROUP)
        self.happy = wx.RadioButton(pnl2, -1, 'happy')
        self.sad = wx.RadioButton(pnl2, -1, 'sad')
        self.surprised = wx.RadioButton(pnl2, -1, 'surprised')
        self.angry = wx.RadioButton(pnl2, -1, 'angry')
        self.disgusted = wx.RadioButton(pnl2, -1, 'disgusted')

        self.Ah = wx.RadioButton(pnl2, -1, 'Ah')

        self.D = wx.RadioButton(pnl2, -1, 'D')
        self.Ee = wx.RadioButton(pnl2, -1, 'Ee')
        self.F = wx.RadioButton(pnl2, -1, 'F')
        self.L = wx.RadioButton(pnl2, -1, 'L')
        self.M = wx.RadioButton(pnl2, -1, 'M')
        self.oh = wx.RadioButton(pnl2, -1, 'oh')
        self.R = wx.RadioButton(pnl2, -1, 'R')
        self.S = wx.RadioButton(pnl2, -1, 'S')
        self.Uh = wx.RadioButton(pnl2, -1, 'Uh')
        self.Woo = wx.RadioButton(pnl2, -1, 'Woo')

        hbox2 = wx.BoxSizer(wx.VERTICAL)
        hbox2.Add(self.neutral, 1)
        hbox2.Add(self.happy, 1)
        hbox2.Add(self.sad, 1)
        hbox2.Add(self.surprised, 1)
        hbox2.Add(self.angry, 1)
        hbox2.Add(self.disgusted, 1)
        hbox2.Add(self.Ah, 1)
        hbox2.Add(self.D, 1)
        hbox2.Add(self.Ee, 1)
        hbox2.Add(self.F, 1)
        hbox2.Add(self.L, 1)
        hbox2.Add(self.M, 1)
        hbox2.Add(self.oh, 1)
        hbox2.Add(self.R, 1)
        hbox2.Add(self.S, 1)
        hbox2.Add(self.Uh, 1)
        hbox2.Add(self.Woo, 1)
        pnl2.SetSizer(hbox2)

        # create horizontal layout with single snapshot button
        pnl3 = wx.Panel(self, -1)
        self.snapshot = wx.Button(pnl3, -1, 'Click Here to Take Photo')
        self.Bind(wx.EVT_BUTTON, self._on_snapshot, self.snapshot)
        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        hbox3.Add(self.snapshot, 1)
        pnl3.SetSizer(hbox3)

        """
        ### This section will add a new panel to the layout, but it overides
        ### the emotion data. Looks better in the GUI though.
        ### create a horizontal layout with buttons for mouth shapes ###
        pnl4 = wx.Panel(self, -1)
        self.Ah = wx.RadioButton(pnl4, -1, 'Ah', (10, 10),
                                      style=wx.RB_GROUP)
        self.D = wx.RadioButton(pnl4, -1, 'D')
        self.Ee = wx.RadioButton(pnl4, -1, 'Ee')
        self.F = wx.RadioButton(pnl4, -1, 'F')
        self.L = wx.RadioButton(pnl4, -1, 'L')
        self.M = wx.RadioButton(pnl4, -1, 'M')
        self.oh = wx.RadioButton(pnl4, -1, 'oh')
        self.R = wx.RadioButton(pnl4, -1, 'R')
        self.S = wx.RadioButton(pnl4, -1, 'S')
        self.Uh = wx.RadioButton(pnl4, -1, 'Uh')
        self.Woo = wx.RadioButton(pnl4, -1, 'Woo')

        hbox4 = wx.BoxSizer(wx.HORIZONTAL)
        hbox4.Add(self.Ah, 1)
        hbox4.Add(self.D, 1)
        hbox4.Add(self.Ee, 1)
        hbox4.Add(self.F, 1)
        hbox4.Add(self.L, 1)
        hbox4.Add(self.M, 1)
        hbox4.Add(self.oh, 1)
        hbox4.Add(self.R, 1)
        hbox4.Add(self.S, 1)
        hbox4.Add(self.Uh, 1)
        hbox4.Add(self.Woo, 1)
        pnl4.SetSizer(hbox4)
        """

        # arrange all horizontal layouts panels vertically
        self.panels_vertical.Add(pnl2, flag=wx.EXPAND | wx.BOTTOM, border=1)
    #   self.panels_vertical.Add(pnl4, flag=wx.EXPAND | wx.BOTTOM, border=1)
        self.panels_vertical.Add(pnl3, flag=wx.EXPAND | wx.BOTTOM, border=1)



    def process_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Add a bounding box around the face if a face is detected.
        """
        _, frame, self.head, _ = self.face_detector.detect_face(frame_rgb)
        return frame

    def _on_snapshot(self, evt):
        """Takes a snapshot of the current frame

            This method takes a snapshot of the current frame, preprocesses
            it to extract the head region, and upon success adds the data
            sample to the training set.
        """
        ## Give labels to images for emotions
        if self.neutral.GetValue():
            label = 'neutral'
        elif self.happy.GetValue():
            label = 'happy'
        elif self.sad.GetValue():
            label = 'sad'
        elif self.surprised.GetValue():
            label = 'surprised'
        elif self.angry.GetValue():
            label = 'angry'
        elif self.disgusted.GetValue():
            label = 'disgusted'

        ## Give labels to images for mouth shapes
        if self.Ah.GetValue():
            label = 'Ah'
        elif self.D.GetValue():
            label = 'D'
        elif self.Ee.GetValue():
            label = 'Ee'
        elif self.F.GetValue():
            label = 'F'
        elif self.L.GetValue():
            label = 'L'
        elif self.M.GetValue():
            label = 'M'
        elif self.oh.GetValue():
            label = 'oh'
        elif self.R.GetValue():
            label = 'R'
        elif self.S.GetValue():
            label = 'S'
        elif self.Uh.GetValue():
            label = 'Uh'
        elif self.Woo.GetValue():
            label = 'Woo'

        if self.head is None:
            print("No face detected")
        else:
            success, aligned_head = self.face_detector.align_head(self.head)
            if success:
                save_datum(self.training_data, label, aligned_head)
                print(f"Saved {label} training datum.")
            else:
                print("Could not align head (eye detection failed?)")

### GUI Layout for Collection MOde ###
def run_layout(layout_cls, **kwargs):
    # open webcam
    capture = cv2.VideoCapture(0)
    # opening the channel ourselves, if it failed to open.
    if not(capture.isOpened()):
        capture.open()

    # Set the size of the openCV window that opens
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # start graphical user interface
    app = wx.App()
    layout = layout_cls(capture, **kwargs)
    layout.Center()
    layout.Show()
    app.MainLoop()

# GUI Layout for Demo MOde
def run_layout2(layout_cls, **kwargs):
    # open webcam
    capture = cv2.VideoCapture(0)
    # opening the channel ourselves, if it failed to open.
    if not(capture.isOpened()):
        capture.open()

    # Set the size of the openCV window that opens
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # start graphical user interface
    app = wx.App()
    layout = layout_cls(capture, **kwargs)
    layout.Center()
    layout.Show()
    app.MainLoop()


# Here we check to see if the user wants to collect new data or if they
# want to run the program with the trained dataset.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['collect', 'demo'])
    parser.add_argument('--classifier', type=Path)
    args = parser.parse_args()

    if args.mode == 'collect':  #if this was selected, go to collection mode.
        run_layout(DataCollectorLayout, title='Select Emotion and Take Photo to Collect Data')
    elif args.mode == 'demo':   #Otherwise, we want to run the program.
        assert args.classifier is not None, 'you have to provide --classifier'
        run_layout2(FacialExpressionRecognizerLayout,
                   title='Facial Expression Recognizer',
                   clf_path=args.classifier)


    #    run_layout3(FacialExpressionRecognizerLayout,
    #               title='Facial Expression Recognizer',
    #               clf_path=args.classifier)
