import cv2
import numpy as np

class Thresholder():
    '''
    Aplica um threshold binário para obter
    as faixas da via.
    '''

    def __init__(self, image):
        self.image = image

    def thresholdImage(self):
      
        # Converte pro espaço de cores YUV
        yuv = cv2.cvtColor(self.image, cv2.COLOR_RGB2YUV)

        # Converte pro espaço de cores HLS
        hls = cv2.cvtColor(self.image, cv2.COLOR_RGB2HLS)

        # Seleciona apenas U, V, e S
        chs = np.stack((yuv[:, :, 1], yuv[:, :, 2], hls[:, :, 2]), axis=2)

        # Converte para grayscale
        gray = np.mean(chs, 2)

        # Operador de Sobal em x com kernel de 3
        s_x = absSobel(gray, orient='x', sobel_kernel=3)

        # Operador de Sobal em y com kernel de 3
        s_y = absSobel(gray, orient='y', sobel_kernel=3)

        # Direcao do gradiente
        grad_dir = dirGradient(s_x, s_y)

        # Magnitude do gradiente
        grad_mag = magGradient(s_x, s_y)

        # Extrai o amarelo (faixa da direita)
        ylw = extractYellow(self.image)

        # Extrai pixels sombreados
        highlights = extractHighlights(self.image[:, :, 0])

        # Monta uma mascara com tudo
        mask = np.zeros(self.image.shape[:-1], dtype=np.uint8)
        mask[((s_x >= 25) & (s_x <= 255) &
                            (s_y >= 25) & (s_y <= 255)) |
                           ((grad_mag >= 30) & (grad_mag <= 512) &
                            (grad_dir >= 0.2) & (grad_dir <= 1.)) |
                           (ylw == 255) |
                           (highlights == 255)] = 1

        # Aplica o filtro e obtem a ROI trapezoidal
        return regionOfInterest(mask)

def absSobel(img_ch, orient='x', sobel_kernel=3):
    '''
    Aplica o gradiente direcional
    absoluto em x ou y.
    '''

    if orient == 'x':
        axis = (1, 0)
    elif orient == 'y':
        axis = (0, 1)

    return np.absolute(cv2.Sobel(img_ch, -1, *axis, ksize=sobel_kernel))

def magGradient(sobel_x, sobel_y):
    '''
    Magnitude do gradiente
    '''
    return np.sqrt(sobel_x ** 2 + sobel_y ** 2).astype(np.uint16)

def dirGradient(sobel_x, sobel_y):
    '''
    Direcao do gradientes
    '''
    with np.errstate(divide='ignore', invalid='ignore'):
        abs_grad_dir = np.absolute(np.arctan(sobel_y / sobel_x))
        abs_grad_dir[np.isnan(abs_grad_dir)] = np.pi / 2

    return abs_grad_dir.astype(np.float32)

def extractYellow(image):
    '''
    Extrai o amarelo da imagem
    no espaco de cor HSV.
    '''
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return cv2.inRange(hsv, (20, 50, 150), (40, 255, 255))


def extractHighlights(image, p=99.9):
    '''
    Seleciona os elementos highlights
    '''
    p = int(np.percentile(image, p) - 30)
    return cv2.inRange(image, p, 255)

def regionOfInterest(image):
    '''
    Aplica uma mascara poligonal na imagem.
    '''
    # Dimensoes da mascara
    MASK_X_PAD = 100
    MASK_Y_PAD = 85

    # Vertices do poligono
    vertices = np.array([[(0, image.shape[0]),                                              
                          (image.shape[1] / 2 - MASK_X_PAD, image.shape[0] / 2 + MASK_Y_PAD),   
                          (image.shape[1] / 2 + MASK_X_PAD, image.shape[0] / 2 + MASK_Y_PAD),   
                          (image.shape[1], image.shape[0])]],                                   
                        dtype=np.int32)

    mask = np.zeros_like(image)

    # Preenche com 1 ou 3 canais
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # Preenche o poligono
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(image, mask)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img = cv2.imread('test_images/test1.jpg')
    imgth = Thresholder(img).thresholdImage()

    plt.imshow(imgth)
    plt.show()    