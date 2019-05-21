import numpy as np
import cv2
import glob, pickle, os


class CameraCalibrator:
    '''
    Realiza a calibração da câmera através
    do método do tabuleiro de xadrez.
    '''

    def __init__(self, images_path, images_names, pickle_name):
        self.__images_path = images_path
        self.__images_names = images_names
        self.__pickle_name = pickle_name
        self.__calibrated = False

    def __calibrate(self):
        ''' 
        Calibra a câmera com inserção manual
        do número de vértices. Retorna os 
        valores de calibração. 
        '''
        print('Initializing the cameras calibration ...')

        # Definindo o número de vértices de cada imagem (ordem das imagens)
        board_dims = [  (9, 5), (9, 6), (9, 6), (9, 6), (9, 6), (9, 6), (9, 6),
                        (9, 6), (9, 6), (9, 6), (9, 6), (9, 6), (9, 6), (9, 6), 
                        (5, 6), (7, 6), (9, 6), (9, 6), (9, 6), (9, 6) ]

        # Arrays auxiliares pros objct points de cada imagem
        objpoints = [] 
        imgpoints = []

        # Imagens do tabuleiro
        images = glob.glob(self.__images_path + '/' + self.__images_names)

        for idx, fname in enumerate(sorted(images)):
            # Atualiza numero de vértices
            (NX, NY) = board_dims[idx]

            # Cria o grid de objct points do tabuleiro (z = 0)
            objp = np.zeros((NX * NY, 3), np.float32)
            objp[:, :2] = np.mgrid[0:NX, 0:NY].T.reshape(-1, 2)

            # Escala de Cinza
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detecta os vértices
            ret, corners = cv2.findChessboardCorners(gray, (NX, NY), None)

            if ret == True:
                imgpoints.append(corners)
                objpoints.append(objp)
                img = cv2.drawChessboardCorners(img, (NX, NY), corners, ret)
            else:
                print('Error detecting checkerboard {}({})'.format(fname, idx))

        img_size = (img.shape[1], img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

        # Salva os parâmetros de calibração
        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump(dist_pickle, open(self.__images_path + '/' + self.__pickle_name, "wb"))

        self.mtx = mtx
        self.dist = dist
        self.__calibrated = True

    def __load_calibration(self):
        ''' 
        Carrega os parâmetros de calibração
        já calculados.
        '''

        print('Loading the calibration parameters ...')

        with open(self.__images_path + '/' + self.__pickle_name, mode='rb') as f:
            cal_data = pickle.load(f)

        self.mtx = cal_data['mtx']
        self.dist = cal_data['dist']
        self.__calibrated = True

    def getMatrixAndCoefficients(self):
        ''' 
        Retorna os valores da matriz da
        câmera e os coeficientes de distorção
        dos arquivos de calibração.
        '''

        if os.path.isfile(self.__images_path + '/' + self.__pickle_name):
            self.__load_calibration()
        else:
            self.__calibrate()

        return self.mtx, self.dist

    def undistort(self, image):
        '''
        Retorna a imagem corrigida pelos
        coeficientes de distorção.
        '''
        
        if  self.__calibrated == False:
            self.getMatrixAndCoefficients()

        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)


CAMERA_CAL_IMAGES_PATH = "./camera_cal"
CAMERA_CAL_IMAGE_NAMES = "calibration*.jpg"
CAMERA_CAL_PICKLE_NAME = "calibration_data.p"

if __name__ == "__main__":
    calibrator = CameraCalibrator(CAMERA_CAL_IMAGES_PATH, CAMERA_CAL_IMAGE_NAMES, CAMERA_CAL_PICKLE_NAME)
    calibrator.getMatrixAndCoefficients()
