import numpy as np
import matplotlib.pyplot as plt
import cv2
class CrossCorrelation:
    def __init__(self,image) -> None:
        self.image = image

    def fit(self, template,steps = 5):
        image_height, image_width = self.image.shape
        template_height, template_width = template.shape
        self.spaces = []
        height_jump = np.floor((image_height-template_height)/steps)
        width_jump = np.floor((image_width-template_width)/steps)
        offset = None
        image = self.image
        epsilon = 1
        for i in range(steps+1):
            new_width = int(template_width+width_jump*i)
            new_height = int(template_height+height_jump*i)
            image = cv2.resize(image,(new_width,new_height))
            if(offset is not None):
                offset[0] = int(offset[0] * (1+ 1 / steps));
                offset[1] = int(offset[1] * (1+ 1 / steps));
            offset = self.get_offset(image,template,offset,epsilon)
            epsilon = epsilon * 0.90
        self.offset = offset

    def get_offset(self,image,template,offset = None,epsilon = 1):
        image_height, image_width = image.shape
        template_height, template_width = template.shape
        # Calcular el tamaño del resultado (imagen convolucionada)
        result_height = image_height - template_height + 1
        result_width = image_width - template_width + 1
        # Inicializar la matriz resultante
        result = np.zeros((result_height, result_width))
        #Calcular la varianza de la imagen template
        template_variance = np.var(template)
        max_pearson = -np.inf
        if(offset is not None):
            for i in range(offset[0]-int(result_height*epsilon),offset[0]+int(result_height*epsilon)):
                for j in range(offset[1]-int(result_width*epsilon),offset[1]+int(result_width*epsilon)):
                    if 0 <= i <= result_height-1 and 0 <= j <= result_width -1 : 
                        image_pixels = image[i:i+template_height, j:j+template_width]
                        image_variance = np.var(image_pixels)
                        covariance = np.cov(template.flatten(),image_pixels.flatten())[0][1]
                        pearson = covariance / (template_variance * image_variance)
                        result[i,j] = pearson
                        if(pearson > max_pearson):
                            max_pearson = pearson
                            offset = [i,j]
            self.spaces.append(result)
            return offset
        else:
            # Realizar la convolución
            for i in range(result_height):
                for j in range(result_width):
                    image_pixels = image[i:i+template_height, j:j+template_width]
                    image_variance = np.var(image_pixels)
                    covariance = np.cov(template.flatten(),image_pixels.flatten())[0][1]
                    pearson = covariance / (template_variance * image_variance)
                    result[i,j] = pearson
                    if(pearson > max_pearson):
                        max_pearson = pearson
                        offset = [i,j]
            self.spaces.append(result)
            return offset
    
    def plot(self):
        if(self.spaces is None):
            return "no"
        for idx,space in enumerate(self.spaces):
            plt.figure(idx)
            m, n = space.shape
            # Crear matrices X y Y para las coordenadas en el plano
            X, Y = np.meshgrid(np.arange(1, n+1), np.arange(1, m+1))
            # Crear la figura en 3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # Graficar en 3D con la función plot_surface
            surf = ax.plot_surface(X, Y, space, cmap='viridis')
            # Configurar etiquetas y título
            ax.set_xlabel('Columnas (X)')
            ax.set_ylabel('Filas (Y)')
            ax.set_zlabel('p')
            ax.set_title('Valores p')
            fig.colorbar(surf)
        plt.show()


if __name__ == "__main__":
    model = CrossCorrelation()