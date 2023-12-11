import numpy as np
import matplotlib.pyplot as plt
def difference_images(A:np.ndarray,B:np.ndarray) -> np.ndarray:
    width, heigth = A.shape
    diff = np.zeros((width,heigth))
    for w in range(width):
        for h in range(heigth):
            diff[w][h] = 255 if abs(A[w][h] - (0.95*B[w][h] + .05*A[w][h])) > 30 else 0
    return diff