from math import ceil, sqrt, log
from cv2 import getGaussianKernel

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

def convolve(img, schema):
    if schema.name == 'gaussian':
        sigma = schema.sigma
        epsilon = 0.0001
        apron = ceil(sigma * sqrt(2*log(1/epsilon)))
        
        convolution = getGaussianKernel(k, sigma)
        
        blocks = (32, 32)
        grids = (ceil(img.shape[0] / blocks[0]), ceil(img.shape[1] / blocks[1]))
        
        kernel = f"""
        __global__ void convolution(float *input, float *output, float *convR, int inpX, int inpY, int k) {{
            // Perform row convolution
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int idy = blockIdx.y * blockDim.y + threadIdx.y;
            
            int blockNum = blockIdx.x + blockIdx.y * gridDim.x;
            int threadNum = threadIdx.x + threadIdx.y * blockDim.x;
            
            
            
            
        }}
        
        """
        