# -*- coding: utf-8 -*-
#!/usr/bin/python

# Standard imports
import Tkinter
from Tkinter import *
import cv2
import numpy as np;
import os
import math
from matplotlib import pyplot as plt
import random
from scipy.stats import norm
import matplotlib.mlab as mlab
import re
import datetime
import numpy.linalg as la
import scipy.misc
from scipy.ndimage import label




class simpleapp_tk(Tkinter.Tk):
	
	def __init__(self,parent):
		Tkinter.Tk.__init__(self,parent)
		self.parent=parent
		self.initialize()

	def initialize(self):

		self.grid()

		self.entryVariable = Tkinter.StringVar()
		### Enter Folder 
		self.entry= Tkinter.Entry(self,textvariable=self.entryVariable)
		self.entry.grid(column=0, row=0, sticky='EW')
		self.entryVariable.set(u"//LANDISK-HDD1/disk/ben/test")

		label4 = Tkinter.Label(self, text="Image Folder",anchor="w")
		label4.grid(column=1,row=0,columnspan=2,sticky="EW")

		preview = Tkinter.Button(self,text=u"Load Variables From Folder",command=self.LoadVariablesFromFolder)
		preview.grid(column=2, row=0)

		### Enter variables to start with, looks for variables in absolute.txt
		self.entryVariable2 = Tkinter.StringVar()
		self.entry2= Tkinter.Entry(self,textvariable=self.entryVariable2)
		self.entry2.grid(column=0, row=1, sticky='EW')
		try:
			autoPointFile = open('absolute.txt', 'r')
			variables = autoPointFile[0]
			
		except Exception as e:
			variables="undefined"
		else:
			variables="undefined"
		finally:
			pass
		self.entryVariable2.set(u""+variables)

		label5 = Tkinter.Label(self, text="Default Variables/Variables from previous training",anchor="w")
		label5.grid(column=1,row=1,columnspan=2,sticky="EW")

		# ### Enter # of Channels 3/4
		# self.entryVariable3 = Tkinter.StringVar()
		# self.entry3= Tkinter.Entry(self,textvariable=self.entryVariable3)
		# self.entry3.grid(column=0, row=2, sticky='EW')
		# self.entryVariable3.set(u"3")
		# label6 = Tkinter.Label(self, text="Number of channels to analyze 3/4 (not yet implemented)",anchor="w")
		# label6.grid(column=1,row=2,columnspan=2,sticky="EW")

		### Enter labels
		self.entryVariable4 = Tkinter.StringVar()
		self.entry4= Tkinter.Entry(self,textvariable=self.entryVariable4)
		self.entry4.grid(column=0, row=4, sticky='EW')
		self.entryVariable4.set(u"starting delta")

		### Enter labels
		self.entryVariable5 = Tkinter.StringVar()
		self.entry5= Tkinter.Entry(self,textvariable=self.entryVariable5)
		self.entry5.grid(column=0, row=5, sticky='EW')
		self.entryVariable5.set(u"rounds after stuck")

		### Enter labels
		self.entryVariable6 = Tkinter.StringVar()
		self.entry6= Tkinter.Entry(self,textvariable=self.entryVariable6)
		self.entry6.grid(column=0, row=6, sticky='EW')
		self.entryVariable6.set(u"randomize value")

		### Enter labels
		self.entryVariable7 = Tkinter.StringVar()
		self.entry7= Tkinter.Entry(self,textvariable=self.entryVariable7)
		self.entry7.grid(column=0, row=7, sticky='EW')
		self.entryVariable7.set(u"Cost Equation (FP,TP) FP-TP, higher FP value means more penalty for false positives, higher TP greater benefit from true positives, so with 1,1 finding 1 new true positive at the cost of 1 more false positive is neutral")

		### Enter labels
		self.entryVariable8 = Tkinter.StringVar()
		self.entry8= Tkinter.Entry(self,textvariable=self.entryVariable8)
		self.entry8.grid(column=1, row=4, sticky='EW')
		self.entryVariable8.set(u"1")

		### Enter labels
		self.entryVariable9 = Tkinter.StringVar()
		self.entry9= Tkinter.Entry(self,textvariable=self.entryVariable9)
		self.entry9.grid(column=1, row=5, sticky='EW')
		self.entryVariable9.set(u"6")

		### Enter labels
		self.entryVariable10 = Tkinter.StringVar()
		self.entry10= Tkinter.Entry(self,textvariable=self.entryVariable10)
		self.entry10.grid(column=1, row=6, sticky='EW')
		self.entryVariable10.set(u"3")

		### Enter labels
		self.entryVariable11 = Tkinter.StringVar()
		self.entry11= Tkinter.Entry(self,textvariable=self.entryVariable11)
		self.entry11.grid(column=1, row=7, sticky='EW')
		self.entryVariable11.set(u"1")

		# ### Enter labels
		# self.entryVariable12 = Tkinter.StringVar()
		# self.entry12 = Tkinter.Entry(self,textvariable=self.entryVariable12)
		# self.entry12.grid(column=2, row=4, sticky='EW')
		# self.entryVariable12.set(u"UNUSED")

		# ### Enter labels
		# self.entryVariable13 = Tkinter.StringVar()
		# self.entry13= Tkinter.Entry(self,textvariable=self.entryVariable13)
		# self.entry13.grid(column=2, row=5, sticky='EW')
		# self.entryVariable13.set(u"UNUSED")

		# ### Enter labels
		# self.entryVariable14 = Tkinter.StringVar()
		# self.entry14= Tkinter.Entry(self,textvariable=self.entryVariable14)
		# self.entry14.grid(column=2, row=6, sticky='EW')
		# self.entryVariable14.set(u"UNUSED")

		### Enter labels
		self.entryVariable15 = Tkinter.StringVar()
		self.entry15= Tkinter.Entry(self,textvariable=self.entryVariable15)
		self.entry15.grid(column=2, row=7, sticky='EW')
		self.entryVariable15.set(u"1")


		self.labelVariable = Tkinter.StringVar()
		label = Tkinter.Label(self, textvariable=self.labelVariable,
							anchor="w", fg="white", bg="blue")
		label.grid(column=0,row=8,columnspan=2,sticky="EW")


		label2 = Tkinter.Label(self, text="Values",
							anchor="w")
		label2.grid(column=1,row=3,columnspan=2,sticky="EW")

		# label3 = Tkinter.Label(self, text="max",
		# 					anchor="w")
		# label3.grid(column=2,row=3,columnspan=2,sticky="EW")

		button = Tkinter.Button(self,text=u"Start Program",
								command=self.OnButtonClick)
		button.grid(column=1, row=8)

		self.test = Tkinter.IntVar()
		c = Checkbutton(self, text="Test. If un-checked, will do training", variable=self.test)
		c.grid(column=2, row=8)

		self.analyze = Tkinter.IntVar()
		c = Checkbutton(self, text="Analyze Cells That haven't been handlabeled", variable=self.analyze)
		c.grid(column=2, row=9)

		self.conservative = Tkinter.IntVar()
		c = Checkbutton(self, text="Analyze only area of cells that fall in the minimum radius", variable=self.conservative)
		c.grid(column=2, row=10)

		self.troubleshoot = Tkinter.IntVar()
		c = Checkbutton(self, text="Troubleshoot (output intermediate images)", variable=self.troubleshoot)
		c.grid(column=0, row=10)

		self.grid_columnconfigure(0,weight=1)
		#self.grid_rowconfigure(0,weight=1)
		self.resizable(False,False)

		#self.entry.focus_set()
        #self.entry.selection_range(0, Tkinter.END)


	def OnButtonClick(self):
		print "You clicked the button !"
		runMain(self.entryVariable.get(),self.entryVariable2.get(),self.entryVariable4.get(),self.entryVariable5.get(),self.entryVariable6.get(),self.entryVariable7.get(),self.entryVariable8.get(),self.entryVariable9.get(),self.entryVariable10.get(),self.entryVariable11.get(),self.entryVariable15.get(),self.test.get(),self.analyze.get(),self.conservative.get(),self.troubleshoot.get())
		## removed: self.entryVariable3.get(),

	def LoadVariablesFromFolder(self):
		print "You clicked the LoadVariablesFromFolder !"
		directory=self.entryVariable.get()
		os.chdir(directory)
		print "Now in directory: "+os.getcwd()

		try:
			print "trying to read"
			autoPointFile = open('absolute.txt', 'r')
			#print "failed after open"
			
			for line in autoPointFile:
				print line
				variables = line
			#self.entryVariable2.set(u""+variables)
		except Exception as e:
			print "failed"
			print e
			variables="undefined"
		else:
			pass
		finally:
			pass
		self.entryVariable2.set(u""+variables)

		#self.self.entryVariable2.set(variables)

def segment_on_dt(a, img,its):
    border = cv2.dilate(img, None, iterations=its)
    border = border - cv2.erode(border, None)

    img = cv2.dilate(img, None, iterations=1)

    dt = cv2.distanceTransform(img, 1, 5)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
    _, dt = cv2.threshold(dt, 180, 255, cv2.THRESH_BINARY)
    lbl, ncc = label(dt)
    lbl = lbl * (255/(max(1,ncc)))
    # Completing the markers now. 
    lbl[border == 255] = 255

    lbl = lbl.astype(np.int32)
    cv2.watershed(a, lbl)

    lbl[lbl == -1] = 0
    lbl = lbl.astype(np.uint8)
    return 255 - lbl

def invert(imagem):
    imagem = (255-imagem)
    return imagem


def getAngle(a, b, c):

    # Create vectors from points


    ba = [ aa-bb for aa,bb in zip(a,b) ]
    bc = [ cc-bb for cc,bb in zip(c,b) ]

    cosang = np.dot(ba, bc)
    sinang = la.norm(np.cross(ba, bc))
    return np.arctan2(sinang, cosang)

def dist(x1,y1, x2,y2, x3,y3): # x3,y3 is the point
    px = x2-x1
    py = y2-y1

    something = px*px + py*py

    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(something)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    # Note: If the actual distance does not matter,
    # if you only want to compare what this function
    # returns to other results of this function, you
    # can just return the squared distance instead
    # (i.e. remove the sqrt) to gain a little performance

    dist = math.sqrt(dx*dx + dy*dy)

    return dist

def findValues(image,posX,posY,diameter):
	square=image[posY-diameter/2:posY+diameter/2+1,posX-diameter/2:posX+diameter/2+1]
	#cv2.imwrite('ExampleSquare.png',square)
	###should return the average of pixel values within the circle inside this square
	count=0
	total=0

	try:
		for x in range(0,len(square)):
			for y in range(0,len(square)):
				if ((diameter/2-y)**2+(diameter/2-x)**2)**.5<diameter/2: ###look at points in circle around
					#square[y,x][2]=255
					count+=1.0
					total+=square[y,x]

		return total/count
	except:
		return -99999



def cost(tp,fp,fn,fpWeight,tpWeight):
	return (fp*fpWeight-tp*tpWeight)
	#return (fp/(10-gDRound/10.0)-tp)*1.0

def checkPoints(pointsX, pointsY, autoPointsX, autoPointsY, draw,w,h,minArea,bg_with_keypoints,noGradientDescent,imageFile):

	print w
	print h
	print minArea
	TP=0
	FN=0
	FP=0

	for point in range(len(pointsX)):
		cv2.rectangle(bg_with_keypoints, (pointsX[point],pointsY[point]), (pointsX[point]+1, pointsY[point]+1),(0,255,0),1)
		
	
	for autoPoint in range(len(autoPointsX)):
		match=0
		for point in range(len(pointsX)):
			if autoPointsX[autoPoint]>pointsX[point]-w*3/8 and autoPointsX[autoPoint]<pointsX[point]+w*3/8 and autoPointsY[autoPoint]>pointsY[point]-h*3/8 and autoPointsY[autoPoint]<pointsY[point]+h*3/8:
				if noGradientDescent==1 or draw==1:
	 				cv2.rectangle(bg_with_keypoints, (pointsX[point]-w/4,pointsY[point]-h/4), (pointsX[point]+w/4, pointsY[point]+h/4),(255,0,0),1)
	 				cv2.putText(bg_with_keypoints,str(autoPoint), (autoPointsX[autoPoint]+w/4, autoPointsY[autoPoint]+h/4), cv2.FONT_HERSHEY_SIMPLEX, .5,(255,255,255),1)	
	 				cv2.circle(bg_with_keypoints,(autoPointsX[autoPoint],autoPointsY[autoPoint]), int((minArea/3.14)*1/2), (255,155,255), 1)
	 				
	 				#cv2.rectangle(bg_with_keypoints, (autoPointsX[autoPoint],autoPointsY[autoPoint]), (autoPointsX[autoPoint]+1, autoPointsY[autoPoint]+1),(255,255,255),1)
				del pointsX[point]
	 			del pointsY[point]
	 			match=1
	 			TP+=1
	 			if point%200==0:
	 				print "Now breaking on",point,"with TP=",TP ## this was included because sometimes this step is very slow
				break
		if match==0:
			if noGradientDescent==1 or draw==1:
				### Draw yellow around the false positives
				cv2.rectangle(bg_with_keypoints, (autoPointsX[autoPoint]-w/4,autoPointsY[autoPoint]-h/4), (autoPointsX[autoPoint]+w/4, autoPointsY[autoPoint]+h/4),(0,255,255),1)
				cv2.putText(bg_with_keypoints,str(autoPoint), (autoPointsX[autoPoint]+w/4, autoPointsY[autoPoint]+h/4), cv2.FONT_HERSHEY_SIMPLEX, .5,(255,255,255),1)
				cv2.circle(bg_with_keypoints,(autoPointsX[autoPoint],autoPointsY[autoPoint]), int((minArea/3.14)*1/2), (255,155,255), 1)
				#print "something happened",autoPointsX[autoPoint]
				#cv2.imshow("asdasd",bg_with_keypoints)
				#cv2.waitKey(0)
			FP+=1
			#print FP
			#print 1.0*TP/(FP+TP)/1.0
	for point in range(len(pointsX)):
	 	if noGradientDescent==1 or draw==1:
	 		### Draw yellow around the false positives
	 		cv2.rectangle(bg_with_keypoints, (pointsX[point]-w/4-1,pointsY[point]-h/4+1), (pointsX[point]+w/4-1, pointsY[point]+h/4+1),(0,255,0),1)
	 	if point%200==0:
	 		print "Drawing false negative on",point,"with FN=",FN ## this was included because sometimes this step is very slow
		FN+=1

	if noGradientDescent==1:


		cv2.imwrite('LabeledImages'+str(imageFile)+'.png',bg_with_keypoints)


	return TP, FP, FN, bg_with_keypoints

def analyzePoints(autoPointsX, autoPointsY, bg_with_keypoints,imageFile,minArea):
	for autoPoint in range(len(autoPointsX)):
		cv2.putText(bg_with_keypoints,str(autoPoint), (autoPointsX[autoPoint]+20/4, autoPointsY[autoPoint]+20/4), cv2.FONT_HERSHEY_SIMPLEX, .5,(255,255,255),1)
		cv2.circle(bg_with_keypoints,(autoPointsX[autoPoint],autoPointsY[autoPoint]), int((minArea/3.14)*1/2), (255,155,255), 1)
	cv2.imwrite('LabeledImages'+str(imageFile)+'.png',bg_with_keypoints)
	return




def runMain(directory,variables,label1,label2,label3,label4, deltaDelta, roundsRounds, randomRandom, fpWeight, tpWeight, test, noCheck,conservative, troubleshoot):


	try:
		deltaDelta=int(deltaDelta)
		roundsRounds=int(roundsRounds)
		randomRandom=int(randomRandom)
		fpWeight=int(fpWeight) #min4
		tpWeight=int(tpWeight) #max4
	except Exception as e:
		raise e
	

	start= datetime.datetime.now()
	folder = directory #'//LANDISK-HDD1/disk/harada/161012 data for Ben/in one folder'
	os.chdir(folder)
	print "Now in directory: "+os.getcwd()
	#dfgdfg=raw_input()

	###
	### w and h are the expected width and height for cells, these can be tuned by the machine learning algorithm, but perhaps this should be a value the user can select?
	###
	w=15
	h=15

	if variables=="zero" or variables=="undefined":
		absVars = [0] * 85
	
	else:
		my_list = variables.split(",")
		print "split variables worked"
		absVars=[]
		for asdf in my_list:
			absVars.append(int(asdf))

	print len(absVars)

	absVarsCopy=[]
	for item in absVars:
		absVarsCopy.append(item)




	#### use this to check unlabeled data
	#noCheck=0

	### Need to go back and remove this stuff
	afterStuckBest=-141
	trueBestCost=afterStuckBest


	continuous=1

	fileList=os.listdir(os.getcwd())
		
	dirList=[]
	imageJFiles=[]

	for filezz in fileList:
		if os.path.isdir(filezz):
			if filezz!='.AppleDouble':
					dirList.append(filezz)
		elif filezz.endswith('.xml') and filezz!='.DS_Store':
			imageJFiles.append(filezz)
			imageJFiles.sort(key=str.lower)
			print imageJFiles

	dirList.sort(key=str.lower)
	print dirList


	ims=[]
	imOrigs=[]
	im2s=[]
	im3s=[]
	imGs=[]
	backImgs=[]
	

	currDir=os.getcwd()
	for folderzz in dirList:
		os.chdir(currDir+'/'+folderzz)
		fileList=os.listdir(os.getcwd())
		for filezzz in fileList:
			if os.path.isdir(filezzz)!=1 and filezzz!='.DS_Store':
				print filezzz
				
				if 'ch00' in filezzz:
					ims.append(cv2.imread(filezzz, cv2.IMREAD_GRAYSCALE))
					imOrigs.append(cv2.imread(filezzz, cv2.IMREAD_GRAYSCALE))
									
				elif 'ch01' in filezzz:
					im2s.append(cv2.imread(filezzz, cv2.IMREAD_GRAYSCALE))				
				elif 'ch02' in filezzz:
					im3s.append(cv2.imread(filezzz, cv2.IMREAD_GRAYSCALE))
				elif 'ch03' in filezzz:
					imGs.append(cv2.imread(filezzz, cv2.IMREAD_GRAYSCALE))
				elif filezzz.endswith('.tif'):
					backImgs.append(cv2.imread(filezzz, cv2.IMREAD_GRAYSCALE))
				else:
					print "ERROR", filezzz
				#elif filezzz.endswith('.xml'):
				#	imageJFiles.append(filezzz)

		if len(ims)>len(backImgs):
			print "background image was not found, using ch00 as background image"
			for filezzz in fileList:
				if os.path.isdir(filezzz)!=1 and filezzz!='.DS_Store':
					print filezzz
					if 'ch00' in filezzz:
						backImgs.append(cv2.imread(filezzz, cv2.IMREAD_GRAYSCALE))

		print "Now in directory: "+os.getcwd()

	os.chdir(currDir)


	if noCheck==1:
		test=1

	#raw_input()



	while continuous==1:
		badResult=0
		variables=absVars
		if test==0:
			continuous=1
			randomize=randomRandom
		else:
			continuous=0
			randomize=0

		if randomize>0:
			for item in range(0, len(variables)):
				if item%2==1:
					print "before",variables[item]
					variables[item]+=random.randint(-randomize, randomize)
					print "after",variables[item]


		nextVars = [0] * len(variables)
		print "number of variables: ",len(variables)
		#variables = [random.randrange(0,2,1) for _ in range (41)]
		#nextVars = [0] * 41
		###
		###
		###


		


		if test==0:
			record=0
			noGradientDescent=0
		else:
			record=1
			noGradientDescent=1

		

		###

		bestCost=0
		currentCost=0
		delta = deltaDelta
		startingdelta=deltaDelta


		first = 1
		improving = 0

		addOtherColors=0
		addEdges=0

		gDRound=0

		roundsAfterStuck=roundsRounds

		###
		###
		###
		while improving < roundsAfterStuck+1:
			if badResult==1:
				break
			gDRound+=1
			print "#################################"
			print "#################################"
			print "round of gradient descent:",gDRound
			print "current delta", delta
			print "last round ended with",bestCost,"new round's variables starting at:",variables


			improving +=1
			for var in range(len(nextVars)):
				TPs=[]
				FPs=[]
				FNs=[]
				variables[var]+=delta
				w=15+variables[1]-variables[2]
				h=15+variables[3]-variables[4]



				


				for imageFile in range(0,len(ims)):
					#print 'what is going on?'
					#print ims
					#print len(ims)
					#print len(imOrigs)
					#print len(dirList)

					#print dirList
					#print imageFile
					# print ims[imageFile]
					# print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ "
					im = ims[imageFile]
					#imMedian=int(np.median(filter(lambda a: a != 0, im.ravel()))*(10.0+variables[35]-variables[36])/10)
					imOrig = imOrigs[imageFile]
					im2 = ims[imageFile]
					#imMedian2 =int(np.median(filter(lambda a: a != 0, im2.ravel()))*(10.0+variables[37]-variables[38])/10)
					im3 = ims[imageFile]
					#imMedian3 = int(np.median(filter(lambda a: a != 0, im3.ravel()))*(10.0+variables[39]-variables[40])/10)
					
					imChan2 = im2s[imageFile]

					imChan3 = im3s[imageFile]


					tileGridSize=max(8,8+variables[77]*4-variables[84]*4)

					clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tileGridSize,tileGridSize))
					cl1 = clahe.apply(im)
					if troubleshoot==1:
						cv2.imwrite('01-clahe_2.jpg',cl1)
					im=cl1
					if troubleshoot==1:
						cv2.imwrite('02-after local thresholding.png',im)

					firstBlur=3+2*(variables[65]-variables[66])
					if firstBlur>0:
						#im = cv2.GaussianBlur(im,(firstBlur,firstBlur),0)
						im = cv2.medianBlur(im,firstBlur)
					ret, otsu = cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

					#cv2.imwrite('after otsu maybe.png',ret)
					


					min1=ret-1+variables[5]-variables[6]
					max1=255-variables[37]

					###
					###
					subtract=-min1
					multiplyFactor=255/max(1,(max1-min1))
					array_sub = np.array([subtract])
					array_mF = np.array([multiplyFactor])

					im= cv2.add(im,subtract)
					im= cv2.multiply(im,multiplyFactor)

					dst = 255-im

					im=dst
					if troubleshoot==1:
						cv2.imwrite('03-AAA.png',im)

					imCopy=im.copy()

					im = cv2.GaussianBlur(im,(1,1),0)

					#th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV);
					# Copy the thresholded image.
					im_floodfill = im.copy()
					# Mask used to flood filling.
					# Notice the size needs to be 2 pixels than the image.
					hh, ww = im.shape[:2]
					mask = np.zeros((hh+2, ww+2), np.uint8)
					# Floodfill from point (0, 0)
					cv2.floodFill(im_floodfill, mask, (0,0), 255);
					im=im_floodfill

					tileGridSize=32

					clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tileGridSize,tileGridSize))
					cl1 = clahe.apply(im)
					if troubleshoot==1:
						cv2.imwrite('04-clahe_2.jpg',cl1)
					im=cl1

					im = cv2.addWeighted(im,.25,imCopy,.75,0)
					if troubleshoot==1:
						cv2.imwrite('05-remixed.png',im)
					
					ret, otsu = cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

					#plt.hist(im.ravel(),256,[0,256])
					#plt.show()
					min1=ret-10+variables[7]-variables[8]
					max1=ret+variables[35]-variables[36]

					subtract=-min1
					if max1-min1<=0:
						multiplyFactor=255

					else:
						multiplyFactor=255/(max1-min1)

					array_sub = np.array([subtract])
					array_mF = np.array([multiplyFactor])

					im= cv2.add(im,subtract)
					im= cv2.multiply(im,multiplyFactor)
					if troubleshoot==1:
						cv2.imwrite('06-BBB.png',im)

					im = cv2.GaussianBlur(im,(1,1),0)

						#th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV);
					# Copy the thresholded image.
					im_floodfill = im.copy()
					# Mask used to flood filling.
					# Notice the size needs to be 2 pixels than the image.
					hh, ww = im.shape[:2]
					mask = np.zeros((hh+2, ww+2), np.uint8)



					# Floodfill from point (0, 0)
					cv2.floodFill(im_floodfill, mask, (0,0), 255);
					im=im_floodfill
				

					#print ret,min1,max1

					if math.isnan(np.median(filter(lambda a: a != 255 and a != 0, im.ravel()))):
						medianish=0
					else:
						medianish = int(np.median(filter(lambda a: a != 255 and a != 0, im.ravel())))*(10.0+variables[41]-variables[42])/10.0

					multiplyFactor=(255-(medianish))/255.0
					add = 255-255*(255-(medianish))/255.0
					#print "medianish=",medianish
					#print "add=",add

					array_mF = np.array([multiplyFactor])
					im= cv2.multiply(im,multiplyFactor)

					array_add = np.array([add])

					#cv2.imwrite('before add.png',im)
					im= cv2.add(im,add)
					if troubleshoot==1:
						cv2.imwrite('07-CCC.png',im)



					#brightest = int(np.median(filter(lambda a: a != 0, im.ravel())))
					#im[np.where((im == [brightest]).all(axis = 0))] = [255]
					ret, im2 = cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

					im = cv2.addWeighted(im,.5+variables[39]/10.0-variables[40]/10.0,im2,.5+variables[40]/10.0-variables[39]/10.0,0)
					
					###maybe this is a good place to do the contours
					if troubleshoot==1:
						cv2.imwrite('08-before contours.png',im)



					imBefore= im.copy()
					#invert for contours
					dst = 255-im
					im=dst

					#im = cv2.imread('afterprocessingthresh.png')
					imCopy = im.copy()
					imgray = im.copy() #cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

					iterations=5+variables[43]-variables[44]

					#cv2.imshow("show",imCopy)
					#cv2.waitKey
					if iterations < 1:
						iterations==1

					for i in range(0,iterations):
						

						im = imCopy.copy()
						imgray = im.copy()#cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
						#ret,thresh = cv2.threshold(imgray,127,255,0)
						ret,thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
						image, contours, hierarchy = cv2.findContours(imgray,2,1)

						if i==0:
							if troubleshoot==1:
								cv2.imwrite('09-insidefirstloop.png',imCopy)

						#cv2.imshow("teeth4.jpg",im2)
						#cv2.waitKey(0)

						for item in contours:
							#print "worked"

							cnt = item

							epsilon = 0.006+(variables[51]-variables[52])/500.0*cv2.arcLength(cnt,True)
							approx = cv2.approxPolyDP(cnt,epsilon,True)

							cnt=approx
							



							#im2 = cv2.drawContours(im2, [cnt], 0, (0,255,0), 3)

							hull =cv2.convexHull(cnt,returnPoints = False)

							if cv2.contourArea(cnt) > 250 + 20*(variables[82]-variables[83]):


								if cv2.isContourConvex(cnt)==False:
									#print cv2.isContourConvex(cnt)
									defects = cv2.convexityDefects(cnt,hull)
									if defects != None:


										for i in range(defects.shape[0]):
											if i!=0:

												s,e,f,d = defects[i,0]
												start = tuple(cnt[s][0])
												end = tuple(cnt[e][0])
												far = tuple(cnt[f][0])

												avDist=dist(start[0],start[1], end[0],end[1], far[0],far[1])

												angle=getAngle(start,far,end)


												if variables[45]-variables[46]>-2:
													mainAx=max(1,min(avDist/max(.5,(2.0+variables[55]/5.0-variables[56]/5.0))+variables[53]-variables[54],(12+variables[57]-variables[58])/max(1,(angle+1)))+variables[45]-variables[46])
													if mainAx>3:
														cv2.ellipse(imCopy,far,(int(mainAx),max(1,int(mainAx*4.0/max(.1,(10.0+variables[59]-variables[60]))))),angle/2/math.pi*360,0,360,[0,0,0],-1)
														#cv2.imshow("show",imCopy)
														#cv2.waitKey(0)
														#cv2.circle(imCopy,far,max(1,min(int(avDist/(2.0+variables[55]/5.0-variables[56]/5.0))+variables[53]-variables[54],int((12+variables[57]-variables[58])/(angle+1))+variables[45]-variables[46])),[0,0,0],-1)
												if variables[49]-variables[50]>0:
														kernel = np.ones((2,2),np.uint8)
														edge = cv2.dilate(imCopy,kernel,iterations = variables[49]-variables[50])

												if variables[49]-variables[50]<0:
														kernel = np.ones((2,2),np.uint8)
														edge = cv2.erode(imCopy,kernel,iterations = variables[50]-variables[49])







					#uninvert after contours
					dst = 255-imCopy
					imCopy=dst

					im = cv2.addWeighted(imBefore,.5+variables[47]/10.0-variables[48]/10.0,imCopy,.5+variables[48]/10.0-variables[47]/10.0,0)
					

					#watershed = 255-imBefore
					watershed = imBefore.copy()
					ret, watershed = cv2.threshold(watershed,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
					watershed = cv2.distanceTransform(watershed,cv2.DIST_L2,0)

					watershed = watershed#/255.0
					watershed = 255-watershed



					watershed = watershed.astype(int)

					# for item in watershed:
					# 	for item2 in item:
					# 		print type(item2)

					hist,bins = np.histogram(watershed.flatten(),256,[0,256])
					cdf = hist.cumsum()
					#cumsumdf_normalized = cdf *hist.max()/ cdf.max() # this line not necessary.

					cdf_m = np.ma.masked_equal(cdf,0)
					cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
					cdf = np.ma.filled(cdf_m,0).astype('uint8')

					try:

							watershed = cdf[watershed]
							watershed1 = watershed/50.0

							scipy.misc.imsave('outfile.png', watershed1)

							watershed1 = cv2.imread('outfile.png', cv2.IMREAD_GRAYSCALE)

							multiplyFac=(2+variables[78]-variables[79])
							if multiplyFac > 0:
								watershed1=watershed1*multiplyFac
					except:
						pass

					#watershed1 = watershed1*(2+variables[78]-variables[79])

					try:
						im=cv2.addWeighted(im,.5+variables[61]/10.0-variables[62]/10.0,watershed1,.5+variables[62]/10.0-variables[61]/10.0,0,0,-1)
					except:
						pass

					#watershed = 255-imBefore
					#watershed = imCopy.copy()
					watershed = im.copy()

					ret, watershed = cv2.threshold(watershed,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
					watershed = cv2.distanceTransform(watershed,cv2.DIST_L2,0)

					watershed=watershed#/255.0
					watershed = 255-watershed

					# for index in range(0,len(watershed)):
					# 	for index2 in range(0,len(watershed[index])):
					# 		watershed[index][index2]=int(watershed[index][index2])
					try:
						watershed = watershed.astype(int)

						# for item in watershed:
						# 	for item2 in item:
						# 		print type(item2)

						hist,bins = np.histogram(watershed.flatten(),256,[0,256])
						cdf = hist.cumsum()
						#cumsumdf_normalized = cdf *hist.max()/ cdf.max() # this line not necessary.

						cdf_m = np.ma.masked_equal(cdf,0)
						cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
						cdf = np.ma.filled(cdf_m,0).astype('uint8')

						watershed = cdf[watershed]


						watershed2 = watershed/255.0

						scipy.misc.imsave('outfile.png', watershed2)

						watershed2 = cv2.imread('outfile.png', cv2.IMREAD_GRAYSCALE)


						multiplyFac=(2+variables[80]-variables[81])
						if multiplyFac > 0:
								watershed2=watershed2*multiplyFac
						#watershed = watershed.astype(float)
						if troubleshoot==1:
							cv2.imwrite('10-outfiletimes10.png',watershed2)
					
					except:
						pass

					try:
						im=cv2.addWeighted(im,.5+variables[63]/10.0-variables[64]/10.0,watershed2,.5+variables[64]/10.0-variables[63]/10.0,0,)
					except:
						pass
					if troubleshoot==1:
						cv2.imwrite('11-after contours.png',im)

					img = im
					img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
					img = invert(img)

					# Pre-processing.
					img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
					_, img_bin = cv2.threshold(img_gray, 0, 255,
					        cv2.THRESH_OTSU)

					# Copy the thresholded image.
					im_floodfill = img_bin.copy()

					# Mask used to flood filling.
					# Notice the size needs to be 2 pixels than the image.
					h2, w2 = img_bin.shape[:2]
					mask = np.zeros((h2+2, w2+2), np.uint8)

					# Floodfill from point (0, 0)
					cv2.floodFill(im_floodfill, mask, (0,0), 255);

					#cv2.imwrite("im_floodfill.png", im_floodfill)

					# Invert floodfilled image
					im_floodfill_inv = cv2.bitwise_not(im_floodfill)

					# Combine the two images to get the foreground.
					im_out = img_bin | im_floodfill_inv

					img_bin = im_out



					img = im_out

					img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
					if troubleshoot==1:
						cv2.imwrite("12-img_binbefore.png", im_out)

					img_bin_3 = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,
					        np.ones((11, 11), dtype=int))

					img_bin_5 = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,
					        np.ones((5, 5), dtype=int))

					img_bin_7 = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,
					        np.ones((7, 7), dtype=int))

					edges3 = cv2.Canny(img_bin_3,100,200)
					edges5 = cv2.Canny(img_bin_5,100,200)
					edges7 = cv2.Canny(img_bin_7,100,200)

					seg_amount=max(1,9+variables[67]*2-variables[68]*2)

					seg_img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,
					        np.ones((seg_amount, seg_amount), dtype=int))
					if troubleshoot==1:
						cv2.imwrite("13-img_binafter.png", seg_img_bin)

					result = segment_on_dt(img, img_bin, max(1,5+variables[75]-variables[76]))
					if troubleshoot==1:
						cv2.imwrite("14-output1.png", result)

					edges = cv2.Canny(seg_img_bin,100,200)

					result[result != 255] = 0
					result = cv2.dilate(result, None)


					img = invert(img)

					img2 = img.copy()
					img2[result == 255] = (255, 255, 255)

					#img2=cv2.addWeighted(img, 1, img2, 1, 1)

					img3 = img.copy()
					img3[edges == 255] = (255, 255, 255)
					img3=cv2.addWeighted(img, 1, img3, 1, 1)
					edges = cv2.dilate(edges, np.ones((3, 3)))
					img3[edges == 255] = (255, 255, 255)
					img3=cv2.addWeighted(img, 1, img3, max(0,.5+variables[69]/20.0-variables[70]/20.0), 1)

					#edges3 = cv2.dilate(edges3, np.ones((3, 3)))


					img4 = img.copy()
					img4[edges5 == 255] = (255, 255, 255)
					img4=cv2.addWeighted(img, 1, img4, 1, 1)
					edges5 = cv2.dilate(edges5, np.ones((3, 3)))
					img4[edges5 == 255] = (255, 255, 255)
					img4=cv2.addWeighted(img, 1, img4, max(0,.5+variables[71]/20.0-variables[72]/20.0), 1)

					img5 = img.copy()
					img5[edges7 == 255] = (255, 255, 255)
					img5=cv2.addWeighted(img, 1, img5, 1, 1)
					edges7 = cv2.dilate(edges7, np.ones((3, 3)))
					img5[edges7 == 255] = (255, 255, 255)
					img5=cv2.addWeighted(img, 1, img5, .5+variables[73]/20.0-variables[74]/20.0, 1)

					img6=cv2.addWeighted(img2, 1, img3, .75, 1)
					img7=cv2.addWeighted(img4, .75, img5, .75, 1)

					#img8=img.copy()
					img=cv2.addWeighted(img6, 1, img7, .5, 1)

					if troubleshoot==1:
						cv2.imwrite("15-output2.png", img)


					im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


					blur = 1+2*variables[9]-2*variables[10]
					if blur>0:
					 	im = cv2.GaussianBlur(im,(blur,blur),0)

					
					if troubleshoot==1:
						cv2.imwrite('16-beforefinalblur.png',im)

					min1=5+variables[11]*3-variables[12]*3
					max1=225+variables[13]-variables[14]

					subtract=-min1#
					multiplyFactor=255/(max1-min1)
					array_sub = np.array([subtract])
					array_mF = np.array([multiplyFactor])

					im2= cv2.add(im,subtract)
					im2= cv2.multiply(im,multiplyFactor)

					blur2 = 1+2*variables[15]-2*variables[16]
					if blur2>0:
					 	#im2 = cv2.GaussianBlur(im2,(blur2,blur2),0)
					 	im2 = cv2.medianBlur(im2,blur2)
					#else:
					# 	im2=im

					im=im2

					if troubleshoot==1:
						cv2.imwrite('17-afterfinalblur.png',im)


					min1=max(1,5+variables[17]*4-variables[18]*4)
					#min1=0
					max1=max(min(255,245+variables[19]*4-variables[20]*4),1)
					#max1=1
					subtract=-min1
					multiplyFactor=255/max(1,(max1-min1))
					array_sub = np.array([subtract])
					array_mF = np.array([multiplyFactor])

					im= cv2.add(im,subtract)
					im= cv2.multiply(im,multiplyFactor)
					if troubleshoot==1:
						cv2.imwrite('18-afterprocessing.png',im)


					paramsNew = cv2.SimpleBlobDetector_Params()


					paramsNew.minThreshold = max(1,50+variables[21]-variables[22])
					paramsNew.maxThreshold = 255 #min(255,255+
					# Filter by Area.
					paramsNew.filterByArea = True

					minArea=max((h/4*w/4),(h/4*w/4)+variables[23]-variables[24])
					paramsNew.minArea = minArea ### the higher this value, the greater minimum size of blobs that can be detected
					paramsNew.maxArea = max((h*w)/4,(h*w)/4+variables[25]-variables[26])

					# Filter by Circularity
					paramsNew.filterByCircularity = True
					paramsNew.minCircularity = .01+variables[27]/20.0-variables[28]/20.0 ### higher the value the more circular images have to be. 1 for true circles

					# params2.filterByCircularity = True
					

					# # Filter by Inertia
					paramsNew.filterByInertia = True
					paramsNew.minInertiaRatio = 0.01+variables[29]/25.0-variables[30]/25.0 ##higher value the more equal width and length

					# params2.filterByInertia = True
					# params2.minInertiaRatio = 0.01/40.0 ##higher value the more equal width and length

					# # Filter by Convexity
					paramsNew.filterByConvexity = True
					paramsNew.minConvexity = 0.925+variables[31]/40.0-variables[32]/40.0

					# params2.filterByConvexity = True
					# params2.minConvexity = 0.95


					

					

					# Create a detector with the parameters
					ver = (cv2.__version__).split('.')
					if int(ver[0]) < 3 :
						detector = cv2.SimpleBlobDetector(paramsNew)
						#detector2 = cv2.SimpleBlobDetector(params2)
					else : 
						detector = cv2.SimpleBlobDetector_create(paramsNew)
						#detector2 = cv2.SimpleBlobDetector_create(params2)

					keypoints = detector.detect(im)

					#print keypoints[0]
					#cv2.waitKey(0)



					pointsindex=0

					for points in keypoints:
						pointsindex+=1
					 	for points2 in range(pointsindex,len(keypoints)):	
					 		if len(keypoints)>0 and points2 < len(keypoints):
					 			if points != keypoints[points2]:
					 				if (keypoints[points2].pt[0]-points.pt[0])**2+(keypoints[points2].pt[1]-points.pt[1])**2 < 100+((w/2)**2 + (h/2)**2+variables[33]*5-variables[34]*5):#:
					 					keypoints.remove(keypoints[points2])
										points2-=1
					

					#cv2.imwrite('TESETESTEST.png',imOrigs[imageFile])
					
					#print "error here?"
					if troubleshoot==1:
						cv2.imwrite('testesttest.png',backImgs[imageFile])
					bg_with_keypoints = cv2.drawKeypoints(backImgs[imageFile], keypoints, np.array([]), (255,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
					im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (255,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

					### TODO: modify this to make multiple points.txt files
					file = open('points.txt', "w")

					for points in keypoints:
						file.write("<MarkerX>"+str(int(points.pt[0]))+"</MarkerX>"+"\n")
						file.write("<MarkerY>"+str(int(points.pt[1]))+"</MarkerY>"+"\n")
						cv2.rectangle(bg_with_keypoints, (int(points.pt[0]),int(points.pt[1])), (int(points.pt[0]+1), int(points.pt[1]+1)),(0,255,255),1)

					#for points in keypoints2:
					#	file.write("<MarkerX>"+str(int(points.pt[0]))+"</MarkerX>"+"\n")
					#	file.write("<MarkerY>"+str(int(points.pt[1]))+"</MarkerY>"+"\n")
					#	cv2.rectangle(bg_with_keypoints, (int(points.pt[0]),int(points.pt[1])), (int(points.pt[0]+1), int(points.pt[1]+1)),(0,255,255),1)

					file.close()

					# Show blobs
					if troubleshoot==1:
						cv2.imwrite('Keypoints.png',bg_with_keypoints)
						cv2.imwrite('IM_Keypoints.png',im_with_keypoints)

					autoPointFile = open('points.txt', 'r')

					autoPointsX =[]
					autoPointsY =[]

					for line in autoPointFile:
						x = re.search("<MarkerX>(\d+)</MarkerX>",
						line)
						y = re.search("<MarkerY>(\d+)</MarkerY>",
						line)
						if x:
							autoPointsX.append(int(x.groups()[0]))
						if y:
							autoPointsY.append(int(y.groups()[0]))


					autoPointFile.close()


					
					#imageJFile = open("CellCounter_1-2-L-V_z0-2-cut-composite.xml", 'r')
					if noCheck != 1:
						imageJFile = open(imageJFiles[imageFile], 'r')
					
						
						pointsX =[]
						pointsY =[]
						# autoPointsX =[]
						# autoPointsY =[]


						for line in imageJFile:
							x = re.search("<MarkerX>(\d+)</MarkerX>",
						      line)
							y = re.search("<MarkerY>(\d+)</MarkerY>",
						      line)
							if x:
								pointsX.append(int(x.groups()[0]))
							if y:
								pointsY.append(int(y.groups()[0]))

						imageJFile.close()

						

					if record==1:
						file = open('cellvalues'+str(imageFile)+'.txt', "w")
						file.write("Cell#,X,Y,size,chan0,chan1,chan2,chan3\n")
						for points in range(0,len(keypoints)):

							if conservative==1:
								file.write(str(points)+","+str(int(keypoints[points].pt[0]))+","+str(int(keypoints[points].pt[1]))+","+str(int(2*int((minArea/3.14)**.5)))+","+str(findValues(ims[imageFile],keypoints[points].pt[0],keypoints[points].pt[1],2*int((minArea/3.14)**.5)))+","+str(findValues(imGs[imageFile],keypoints[points].pt[0],keypoints[points].pt[1],2*int((minArea/3.14)**.5)))+","+str(findValues(im2s[imageFile],keypoints[points].pt[0],keypoints[points].pt[1],2*int((minArea/3.14)**.5)))+","+str(findValues(im3s[imageFile],keypoints[points].pt[0],keypoints[points].pt[1],2*int((minArea/3.14)**.5)))+"\n")
								#print "conservative triggered"
							else:
								file.write(str(points)+","+str(int(keypoints[points].pt[0]))+","+str(int(keypoints[points].pt[1]))+","+str(int(keypoints[points].size))+","+str(findValues(ims[imageFile],keypoints[points].pt[0],keypoints[points].pt[1],keypoints[points].size))+","+str(findValues(imGs[imageFile],keypoints[points].pt[0],keypoints[points].pt[1],keypoints[points].size))+","+str(findValues(im2s[imageFile],keypoints[points].pt[0],keypoints[points].pt[1],keypoints[points].size))+","+str(findValues(im3s[imageFile],keypoints[points].pt[0],keypoints[points].pt[1],keypoints[points].size))+"\n")
								#print "else triggered"

							cv2.rectangle(bg_with_keypoints, (int(keypoints[points].pt[0]),int(keypoints[points].pt[1])), (int(keypoints[points].pt[0]+1), int(keypoints[points].pt[1]+1)),(0,255,255),1)
							

						file.close()

					if noCheck != 1:
						#print "triggering checkPoints from line 409"
					
						TP, FP, FN, a = checkPoints(pointsX, pointsY, autoPointsX, autoPointsY, 1,w,h,minArea,bg_with_keypoints,noGradientDescent,imageFile)
						#cv2.imwrite('temp.png',a)

						TPs.append(TP)
						FPs.append(FP)
						FNs.append(FN)

						print "$$$$",sum(TPs)
						print "$$$$",sum(FPs)
						print "$$$$",sum(FNs)
						print "for this section:"
						print "Sensitivity",TP*1.0/(TP+FN)
						print "Accuracy of Positives",TP*1.0/max(1,(TP+FP))

					else:
						print w, h, minArea
						analyzePoints(autoPointsX, autoPointsY, bg_with_keypoints,imageFile,minArea)

				if noCheck != 1:
					print "noCheck !=1 triggered"
					

					TP=sum(TPs)
					FP=sum(FPs)
					FN=sum(FNs)

					if first==1:

						print "first triggered"
						first=0
						firstCost=cost(TP,FP,FN,fpWeight,tpWeight)
						if noGradientDescent!=1:
							if TP<5:
								if TP>0:
									variables=absVars
									print "some positives, but less than threshold retrying random search from this point"
								else:
									absVars=absVarsCopy
									variables=absVars
									print "No positives, retrying random search"
								badResult=1
								break
								
							else:
								print "RANDOMIZATION SUCCESS"

						currentCost=firstCost
						bestCost=firstCost
						trueBestCost=firstCost

						bestVars=list(variables) ### added this in because of error if there is no improvement found on first round
						
						### added this in, maybe messed up something from before
						tempVars=list(variables)

						#print firstCost
						var=0

						### TODO Not sure why this is here so removing it
						### maybe have to add back in later?
						if noGradientDescent!=1:
							#print "triggering checkPoints from line 422"
							a, b, c, bg_with_keypoints = checkPoints(pointsX, pointsY, autoPointsX, autoPointsY, 1,w,h,minArea,bg_with_keypoints,noGradientDescent,imageFile)
							#print "triggering draw blue square!!!"
						#cv2.imwrite('Keypoints-Matches.png',bg_with_keypoints)
						#cv2.imwrite('Keypoints-Matches-on-original.png',bg_with_keypoints)
					else:
						#print "first didnt trigger triggered"
						if var==0:
							if cost(TP,FP,FN,fpWeight,tpWeight)<trueBestCost:
								bestVars=list(variables)
								bestCost=cost(TP,FP,FN,fpWeight,tpWeight)
								#trueBestCost=cost(TP,FP,FN) ##this occurs below and variables are written
								currentCost=cost(TP,FP,FN,fpWeight,tpWeight)
							else:
								variables=list(bestVars)
								bestCost=trueBestCost
								currentCost=trueBestCost
								trueBestCost=trueBestCost

						if cost(TP,FP,FN,fpWeight,tpWeight)<trueBestCost:
							bestVars=list(variables)
							file = open('bestvars.txt', "w")
							file.write(",".join(str(x) for x in bestVars))
							file.close()
							trueBestCost=cost(TP,FP,FN,fpWeight,tpWeight)

						if cost(TP,FP,FN,fpWeight,tpWeight)<currentCost:
							print "new best improvement found"
							improving = 0
							### If this is enabled only one variable will change each round
							#for zero in range(len(nextVars)):
							#	nextVars[zero]=0
							nextVars[var]=delta
							print "current next",nextVars
							bestCost=cost(TP,FP,FN,fpWeight,tpWeight)
							tempVars=list(variables)
							#tempVars[var]+=delta
							print "new best vars found at",bestVars,"with cost",bestCost
							#print "triggering checkPoints from line 449"
							a, b, c, bg_with_keypoints = checkPoints(pointsX, pointsY, autoPointsX, autoPointsY, 1,w,h,minArea,bg_with_keypoints,noGradientDescent,imageFile)
							#print "triggering draw blue square"

							

							#cv2.imwrite('Keypoints-Matches.png',bg_with_keypoints)
					print "OVERALL:"
					print "modified variable",var
					print "True Positive",TP
					print "False Negative",FN
					print "Sensitivity",TP*1.0/(TP+FN)
					print "Accuracy of Positives",TP*1.0/max(1,(TP+FP))
					print "False Positive",FP
					print "Cost (function being minimized)",cost(TP,FP,FN,fpWeight,tpWeight)
					print "Last rounds best cost", currentCost
					print "Current best cost",bestCost
					print "True best cost",trueBestCost
					print "firstCost", firstCost
					print "SECTION BY SECTION ANALYSIS"
					for item in range(0,len(TPs)):
						print "SECTION ",item
						print "Sensitivity",TPs[item]*1.0/(TPs[item]+FNs[item])
						print "Accuracy of Positives",TPs[item]*1.0/max(1,(TPs[item]+FPs[item]))


					print "variables before",variables
					variables[var]-=delta
					print "variables after",variables



				#print "first didnt trigger triggered"
				else:
					print "no check triggered"

				if noGradientDescent==1:
						break
			if noGradientDescent==1:
				break

			for item in range(len(nextVars)):
				variables[item]+=nextVars[item]
			currentCost=bestCost

			for zero in range(len(nextVars)):
				nextVars[zero]=0

			if improving!=0:
				delta+=startingdelta
			else:
				delta=startingdelta

		if trueBestCost<afterStuckBest:
			afterStuckBest=trueBestCost
			with open("bestvars.txt") as f:
				line=f.readline()
				print line
				absVars=map(int, line.split(','))
			with open("absolute.txt", "w") as f1:
				f1.writelines(line)

	#cv2.imwrite('Keypoints-Matches.png',bg_with_keypoints)	

	if noGradientDescent!=1:
		print "Lowest cost found with",bestVars



	finish = datetime.datetime.now()

	print "Start time:",start
	print "End time:",finish

if __name__ == "__main__":
	app = simpleapp_tk(None)
	app.title('my application')
	app.mainloop()
#TPs.append(TP)
#FPs.append(FP)
#FNs.append(FN)

