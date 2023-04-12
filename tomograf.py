from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QDate
from PyQt5 import uic
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.transform import radon
import numpy as np
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
import pydicom._storage_sopclass_uids

from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity



def BresenhamLine(x1,y1,x2,y2):

    d, dx, dy, ai, bi, xi, yi = 0, 0, 0, 0, 0, 0, 0
    x=x1
    y=y1
    
    vertexes_x=[]
    vertexes_y=[]
    
    if(x1 < x2):
        xi = 1
        dx = x2 - x1
    else:
        xi = -1
        dx = x1 - x2

    if (y1 < y2):
        yi = 1
        dy = y2 - y1
    else:
        yi = -1
        dy = y1 - y2

    vertexes_x.append(x)
    vertexes_y.append(y)

    if (dx > dy):
        ai = (dy - dx) * 2
        bi = dy * 2
        d = bi - dx

        while (x != x2):
            if (d >= 0):
                x += xi
                y += yi
                d += ai
            else:
                d += bi
                x += xi
                
            vertexes_x.append(x)
            vertexes_y.append(y)
    else:
        ai = ( dx - dy ) * 2
        bi = dx * 2
        d = bi - dy
        while (y != y2):
            if (d >= 0):
                x += xi
                y += yi
                d += ai
            else:
                d += bi
                y += yi
             
            vertexes_x.append(x)
            vertexes_y.append(y)
    return vertexes_x,vertexes_y

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.img = cv.imread('C:\\main\\Medycyna\\tomograf\\zdjecia\\CT_ScoutView-large.jpg', cv.IMREAD_GRAYSCALE)
        
        uic.loadUi('tomograf.ui', self)

        self.setWindowTitle('Tomograf')

        self.loadButton.clicked.connect(self.read_image)

        self.startButton.clicked.connect(self.start)

        self.slider = self.findChild(QSlider, 'Slider')
        self.slider.valueChanged.connect(self.slide_it)
        self.slider.setVisible(False)

        self.delta_alphaEdit.setText('2')
        self.n_Edit.setText('100')
        self.l_edit.setText('100')

        self.loadButton_2.clicked.connect(self.loadfile)

        self.saveButton.clicked.connect(self.savefile)

        self.show()
        
    def read_image(self):

        empty_pixmap = QPixmap()
        self.slider.setVisible(False)
        self.sin_label.setPixmap(empty_pixmap)
        self.output_label.setPixmap(empty_pixmap)

        filename, _ = QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.jpg)")
        self.img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        pixmap = QPixmap(filename)
        pixmap = pixmap.scaled(300, 300)

        self.image_label.setPixmap(pixmap)

        self.img = addPadding(self.img)

    def start(self):
        
        empty_pixmap = QPixmap()
        self.slider.setVisible(False)
        self.sin_label.setPixmap(empty_pixmap)
        self.output_label.setPixmap(empty_pixmap)

        delta_alpha = self.delta_alphaEdit.text()
        num = self.n_Edit.text()
        l = self.l_edit.text()

        self.slider.setMaximum(360//int(delta_alpha))

        sinogram = radon(self.img, int(delta_alpha), int(num), int(l))

        plt.imsave('temp\\sinogram.jpg', sinogram,cmap='gray')
        file = 'temp\\sinogram.jpg'
        # Create a QPixmap object from the QImage
        pixmap = QPixmap(file)
        pixmap = pixmap.scaled(300, 300)

        self.sin_label.setPixmap(pixmap)

        print(delta_alpha, num, l)

        if self.filtrBox.isChecked():
            self.output = i_radon(self.img, int(delta_alpha), int(num), int(l), True)
        else:
            self.output = i_radon(self.img, int(delta_alpha), int(num), int(l), False)
        plt.imsave('temp\\output.jpg', self.output[(360//int(delta_alpha)-1)],cmap='gray')
        file = 'temp\\output.jpg'

        pixmap = QPixmap(file)
        pixmap = pixmap.scaled(300, 300)
        self.output_label.setPixmap(pixmap)
        self.slider.setVisible(True)
        return self.output
    
    def slide_it(self, value):
        print(value)
        plt.imsave('temp\\output.jpg', self.output[value-1],cmap='gray')
        file = 'temp\\output.jpg'

        pixmap = QPixmap(file)
        pixmap = pixmap.scaled(300, 300)
        self.output_label.setPixmap(pixmap)

    def loadfile(self):
        filename, _ = QFileDialog.getOpenFileName(None, "Select DICOM", "", "DICOM file (*.dcm)")
        dicom_file = pydicom.dcmread(filename)

        id = dicom_file.PatientID
        name = dicom_file.PatientName
        comm = dicom_file.ImageComments
        try:
            date = dicom_file.Date
            self.calendar.setSelectedDate(QDate.fromString(date, 'yyyyMMdd'))
        except:
            print("nodate")

        self.PatientIDEdit.setText(str(id))
        self.PatientNameEdit.setText(str(name))
        self.ImageCommentsEdit.setText(str(comm))

        image = dicom_file.pixel_array

        plt.imsave('temp\\DICOM_img.jpg',image ,cmap='gray')
        pixmap = QPixmap('temp\\DICOM_img.jpg')
        pixmap = pixmap.scaled(300, 300)
        self.loadedImage.setPixmap(pixmap)

    def savefile(self):
        filename, _ = QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.jpg)")
        img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        img_converted = convert_image_to_ubyte(img)
        id = self.PatientIDEdit.text()
        name = self.PatientNameEdit.text()
        comm = self.ImageCommentsEdit.toPlainText()
        date = self.calendar.selectedDate().toString("yyyyMMdd")
        meta = Dataset()
        meta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
        meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian  
        ds = FileDataset(str(id)+'.dcm', {}, preamble=b"\0" * 128)
        ds.file_meta = meta
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        ds.SOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        
        ds.PatientName = str(id)
        ds.PatientID = str(name)
        ds.ImageComments = str(comm)
        ds.Date = str(date)

        ds.Modality = "CT"
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
        ds.FrameOfReferenceUID = pydicom.uid.generate_uid()

        ds.BitsStored = 8
        ds.BitsAllocated = 8
        ds.SamplesPerPixel = 1
        ds.HighBit = 7

        ds.ImagesInAcquisition = 1
        ds.InstanceNumber = 1

        ds.Rows, ds.Columns = img_converted.shape

        ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"

        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0

        pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)

        ds.PixelData = img_converted.tobytes()

        ds.save_as('DICOM\\'+str(id)+'.dcm', write_like_original=False)

def kernel_gen(size):
    kernel = []
    for i in range(size):
        if i == size // 2:
            kernel.append(1)
        else:
            if abs(i - size / 2) % 2 == 0:
                kernel.append(0)
            else:
                kernel.append(-4 / np.pi ** 2 * (1 / (abs(i - size // 2) ** 2)))
    return kernel


def addPadding(img):
    result = np.zeros([max(img.shape), max(img.shape)]) if max(img.shape) % 2 == 1 else np.zeros(
        [max(img.shape) + 1, max(img.shape) + 1])
    result[:img.shape[0], :img.shape[1]] = img
    return result

def convert_image_to_ubyte(img):
    return img_as_ubyte(rescale_intensity(img, out_range=(0.0, 1.0)))

def radon(img, delta_alpha, num, l):
    size = min(img.shape)
    sinogram = []
    radius = size // 2

    theta = np.arange(0, 360, delta_alpha)
    offsets=np.linspace(-l/2,l/2,num)

    for i in theta:

        xe = radius * np.cos(np.deg2rad(i)) + radius
        ye = radius * np.sin(np.deg2rad(i)) + radius

        lines=[]

        for n in range(num):
            
            xD = radius * np.cos(np.deg2rad(i + offsets[n] + 180) ) + radius
            yD = radius * np.sin(np.deg2rad(i+ offsets[n] + 180) ) + radius

            lines.append(BresenhamLine(int(xe), int(ye), int(xD), int(yD)))
        result=[]

        for line in lines:
            res=0
            for i in range(len(line[0])):
                if(line[0][i]<size and line[1][i]<size): 
                    res+=img[line[0][i],line[1][i]]
            result.append(res/len(line[0])) 
    
        sinogram.append(result)
    return sinogram

def i_radon(img, delta_alpha, num, l, filtr):
    size = min(img.shape)
    partial_output=[]
    radius = size // 2

    theta = np.arange(0, 360, delta_alpha)
    offsets=np.linspace(-l/2,l/2,num)
    kernel = kernel_gen(num//2)
    for i in theta:

        xe = radius * np.cos(np.deg2rad(i)) + radius
        ye = radius * np.sin(np.deg2rad(i)) + radius
        lines = []

        for n in range(num):
            
            xD = radius * np.cos(np.deg2rad(i + offsets[n] + 180) ) + radius
            yD = radius * np.sin(np.deg2rad(i+ offsets[n] + 180) ) + radius
        
            lines.append(BresenhamLine(int(xe), int(ye), int(xD), int(yD)))

        result=[]

        for line in lines:
            res=0
            for i in range(len(line[0])):
                if(line[0][i]<size and line[1][i]<size): 
                    res+=img[line[0][i],line[1][i]]
            result.append(res/len(line[0])) 
        if filtr:
            result = np.convolve(result, kernel, "same")

        output=np.zeros((size, size))

        for i in range(len(lines)):
            for j in range(len(lines[i][0])):
                x=lines[i][0][j]
                y=lines[i][1][j]

                if(x<size and y<size): 
                    output[x][y]+=result[i]
        
        output/=len(lines)
    
        partial_output.append(output)
    
    partial_avg_output=[np.zeros((size, size)) for i in range(len(theta))]
    
    for i in range(size):
        for j in range(size):
            suma=0
            for iteration in range(0,len(theta)):
                suma+=partial_output[iteration][i][j]
                partial_avg_output[iteration][i][j]=suma/(iteration+1)     
    
    for i in range(len(partial_avg_output)):
        cv.normalize(partial_avg_output[i], partial_avg_output[i], alpha=0, beta=1, norm_type=cv.NORM_MINMAX)   
    
    return partial_avg_output



app = QApplication([])

window = MainWindow()
window.show()

app.exec_()


