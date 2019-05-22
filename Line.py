import numpy as np

XM_PER_PIXEL = 3.7 / 700                # (m/px) em x
YM_PER_PIXEL = 30.0 / 720               # (m/px) em y

IMAGE_MAX_Y = 719

class Line():
    '''
    Mantem o track de uma faixa (ou esq ou direita) 
    '''
    def __init__(self, n_frames=1, detected_x=None, detected_y=None):
        # Numero do quadro anterior para o smooth
        self.n_frames = n_frames     
        # Coef. polinomial da média dos ultimos quadros
        self.best_fit = None      
        # Coef. polinomial do ultimo quadro
        self.current_fit = None      
        # Polinomio com os coef. do ultimo quadro
        self.current_fit_poly = None
        # Polinomio com os coef. da media dos ultimos quadros
        self.best_fit_poly = None    
        # X da linha detectada
        self.allx = None      
        # Y da linha detectada
        self.ally = None      

        # Update do objeto
        self.update(detected_x, detected_y)

    def update(self, x, y):
        '''
        Atualiza as propriedades conforme
        as coordenadas de x e y
        '''
        self.allx = x
        self.ally = y

        # Faz o fit quadratico e smooth com o ultimo frame
        self.current_fit = np.polyfit(self.allx, self.ally, 2)
        
        # Atualiza os coef polinomial da média dos ultimos
        if self.best_fit is None:
            self.best_fit = self.current_fit
        else:
            self.best_fit = (self.best_fit * (self.n_frames - 1) + self.current_fit) / self.n_frames

        self.current_fit_poly = np.poly1d(self.current_fit)
        self.best_fit_poly = np.poly1d(self.best_fit)


    def areLinesParallel(self, other_line, threshold=(0, 0)):
        '''
        Determina se as duas faixas são
        paralelas através do fit
        '''
        diff_coeff_first = np.abs(self.current_fit[0] - other_line.current_fit[0])
        diff_coeff_second = np.abs(self.current_fit[1] - other_line.current_fit[1])

        return diff_coeff_first < threshold[0] and diff_coeff_second < threshold[1]

    def distanceBetweenLines(self, other_line):
        '''
        Calcula a distancia (diferenca)
        entre as os fits das duas faixas
        '''
        return np.abs(self.current_fit_poly(IMAGE_MAX_Y) - other_line.current_fit_poly(IMAGE_MAX_Y))


    def distanceBetweenBestFit(self, other_line):
        '''
        Calcula a distrancia (diferenca)
        entre os melhores fits das duas faixas
        '''
        return np.abs(self.best_fit_poly(IMAGE_MAX_Y) - other_line.best_fit_poly(IMAGE_MAX_Y))

def calcCurvature(self, fit_cr):
    '''
    Calcula a curvatura da faixa
    em metros
    '''
    y = np.array(np.linspace(0, IMAGE_MAX_Y, num=10))
    x = np.array([fit_cr(x) for x in y])
    y_eval = np.max(y)

    fit_cr = np.polyfit(y * YM_PER_PIXEL, x * XM_PER_PIXEL, 2)
    curverad = ((1 + (2 * fit_cr[0] * y_eval / 2. + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])

    return curverad