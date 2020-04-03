# -*- coding : utf-8 -*-

# for GUI
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic

# for CV
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
# from imageai.Detection.Custom import CustomObjectDetection
import os

UI = 'DeepROV_UI.ui'

class DeepROV(QDialog):
	def __init__(self):
		QDialog.__init__(self, None)
		uic.loadUi(UI, self)

		# Load Input Image
		self.LoadImg_pushButton.clicked.connect(self.LoadImgClicked)

		# Load Pre-trained Weight 
		self.LoadWeight_pushButton.clicked.connect(self.LoadWeightClicked)

		# Load json
		self.Loadjson_pushButton.clicked.connect(self.LoadjsonClicked)

		# Detection
		self.Detection_pushButton.clicked.connect(self.Detection)



	def LoadImgClicked(self):
		# print('Clicked!')
		fname = QFileDialog.getOpenFileName(self)
		# 화면 상에 파일 경로를 출력
		self.ImgName.setText(fname[0])

		# 'InputImgName'에 파일 이름 할당 
		self.InputImgName = fname[0]
		# print(self.InputImgName)

	def LoadWeightClicked(self):
		fname = QFileDialog.getOpenFileName(self)
		self.WeightName.setText(fname[0])
		self.WeightFileName = fname[0]

	def LoadjsonClicked(self):
		fname = QFileDialog.getOpenFileName(self)
		self.jsonName.setText(fname[0])
		self.jsonFileName = fname[0]

	def Detection(self):
		from imageai.Detection.Custom import CustomObjectDetection
		import matplotlib.pyplot as plt
		from matplotlib.image import imread
		import os

		self.StatusBox.setText('Library Loading Finished. Run Detection.')

		PretrainedModel = self.WeightFileName
		self.outputImg = self.InputImgName[:-4] + '_output.jpg'

		detector = CustomObjectDetection()
		detector.setModelTypeAsYOLOv3()
		detector.setModelPath(PretrainedModel)
		detector.setJsonPath(self.jsonFileName)
		detector.loadModel()
		detectionImg = detector.detectObjectsFromImage(input_image=self.InputImgName,
		                                             output_image_path=self.outputImg,
		                                             minimum_percentage_probability=30)

		self.StatusBox.setText('Finished')

		for eachObject in detectionImg:
		    print(eachObject["name"] , " : " , eachObject["percentage_probability"], " : ", eachObject["box_points"])





app = QApplication(sys.argv)
main_dialog = DeepROV()
main_dialog.show()
app.exec_()
