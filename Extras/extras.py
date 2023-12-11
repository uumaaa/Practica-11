import numpy as np

def convolution2D(image, kernel):
    # Obtener las dimensiones de la imagen y el kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calcular el tamaño del resultado (imagen convolucionada)
    result_height = image_height - kernel_height + 1
    result_width = image_width - kernel_width + 1
    
    # Inicializar la matriz resultante
    result = np.zeros((result_height, result_width))
    
    # Realizar la convolución
    for i in range(result_height):
        for j in range(result_width):
            result[i, j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * kernel)
    
    return result
