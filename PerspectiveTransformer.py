import numpy as np
import cv2


class PerspectiveTransformer():
    '''
    Calcula a transformação de perspectiva
    que leva a uma visão 'aérea'. Considerando
    que as vias são suficientemente planas.
    '''
    def __init__(self):
        OFFSET = 260
        Y_HORIZON = 470
        Y_BOTTOM = 720

        # Pontos iniciais
        src = np.float32([[ 300, Y_BOTTOM],      
                          [ 580, Y_HORIZON],     
                          [ 730, Y_HORIZON],     
                          [1100, Y_BOTTOM]])     

        # Pontos finais
        dst = np.float32([
            (src[0][0]  + OFFSET, Y_BOTTOM),
            (src[0][0]  + OFFSET, 0),
            (src[-1][0] - OFFSET    , 0),
            (src[-1][0] - OFFSET    , Y_BOTTOM)])

        # Calcula a transformada de perspectiva
        self.M = cv2.getPerspectiveTransform(src, dst)
       
        # Calcula a transformada inversa de perspectiva
        self.Minv = cv2.getPerspectiveTransform(dst, src)


    def warp(self, image):
        return cv2.warpPerspective(image, self.M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)


    def inverseWarp(self, image):
        return cv2.warpPerspective(image, self.Minv, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img = cv2.imread('test_images/test2.jpg')
    imgpt =  PerspectiveTransformer().warp(img)

    plt.imshow(imgpt)
    plt.show()    