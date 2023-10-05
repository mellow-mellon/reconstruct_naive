import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from helper import integrateFrankot

def loadData(path = "../data/"):

    # for each image, turn into grayscale
    def oneImage(idx):
        img = Image.open(path + f'PhotometricStereo/female_0{idx}.tif')
        img_array = np.array(img)

        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return grayscale


    I = np.array([oneImage(i) for i in range(1, 8)]) 
    s = np.array([len(I[0]), len(I[0][0])]) # dimention of each image
    I = I.reshape(7, -1) # images: (number of image * dimention[0] * dimention[1])
    L = np.load('../data/sources.npy') # lights: (3 * number of images)
    return I, L, s


def estimateAlbedosAndNormals(I, L, s):
    # pseudo normal
    B = np.linalg.lstsq(L, I, rcond=None)[0]

    # for each pixel, b = an (pseudo normal)
    def eachPixel(x, y, z):
        v = np.array([x, y, z])
        norm = np.linalg.norm(v)
        if np.isinf(norm) or np.isnan(norm):
          return np.zeros_like(v)  # Return a zero vector if the norm is close to zero
        if np.isclose(norm, 0):
          return np.ones_like(v)
        normalized_v = v / norm
        return normalized_v

    def eachPixelNorm(x, y, z):
        v = np.array([x, y, z])
        return np.linalg.norm(v)

    normals = np.apply_along_axis(lambda col: eachPixel(*col), axis=0, arr=B)
    albedos = np.apply_along_axis(lambda col: eachPixelNorm(*col), axis=0, arr=B)

    return albedos, normals #(159039,), (3, 159039) 


def displayAlbedosNormals(albedos, normals, s):
    albedos = albedos.reshape(s)
    plt.imshow(albedos, cmap='gray') 
    plt.colorbar() 
    plt.show()
    normals = (normals + 1) / 2
    normals = np.mean(normals, axis=0)
    normals = normals.reshape(s)
    plt.imshow(normals) 
    plt.colorbar() 
    plt.show()



def estimateDepth(normals, s):
    normals = normals.reshape(3, s[0], s[1])
    depthimg = integrateFrankot(normals[0, ...], normals[1, ...])
    plt.imshow(depthimg, cmap='gray') 
    plt.colorbar()  
    plt.show()
    return depthimg


def displayDepth(depth, s):
    depth = - depth
    x = range(s[1])
    y = range(s[0])
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')   
    X, Y = np.meshgrid(x, y)  
    ha.plot_surface(X, Y, depth, cmap='viridis')
    plt.show()




if __name__ == '__main__':

    I, L, s = loadData()
    albedos, normals = estimateAlbedosAndNormals(I, L, s)
    displayAlbedosNormals(albedos, normals, s)
    depthimg = estimateDepth(normals, s)
    displayDepth(depthimg, s)
     
