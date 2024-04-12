import numpy as np
import cv2
# import pycuda.driver as cuda
# import pycuda.autoinit
# from pycuda.compiler import SourceModule

# # CUDA kernel for Gaussian blur
gaussian_blur_kernel = """
__global__ void gaussian_blur(float *input, float *output, int width, int height) {
    // Calculate pixel index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    __shared__ float gaussian_kernel[3][3];

    if (idx < width && idy < height) {
        // Apply Gaussian blur (simplified for illustration)
        
        
        output[idx + idy * width] = input[idx + idy * width];
    }
}
"""


# get gaussian kernel
def get_gaussian_kernel(ksize, sigma):
    kernel = cv2.getGaussianKernel(ksize, sigma)
    kernel = kernel * kernel.T
    sum = np.sum(kernel)
    
    if abs(sum - 1) > 1e-7:
        kernel = kernel / sum
        
    return kernel

# # Compile CUDA kernel
# mod = SourceModule(gaussian_blur_kernel)

# # Get kernel function
# gaussian_blur_func = mod.get_function("gaussian_blur")

# def gaussian_blur_cuda(input_array):
#     # Get image dimensions
#     height, width = input_array.shape

#     # Allocate memory on GPU
#     input_gpu = cuda.mem_alloc(input_array.nbytes)
#     output_gpu = cuda.mem_alloc(input_array.nbytes)

#     # Copy input data to GPU memory
#     cuda.memcpy_htod(input_gpu, input_array)

#     # Define block and grid dimensions
#     block = (16, 16, 1)
#     grid = ((width + block[0] - 1) // block[0], (height + block[1] - 1) // block[1], 1)

#     # Call CUDA kernel
#     gaussian_blur_func(input_gpu, output_gpu, np.int32(width), np.int32(height), block=block, grid=grid)

#     # Allocate memory on CPU for output
#     output_array = np.empty_like(input_array)

#     # Copy output data from GPU to CPU
#     cuda.memcpy_dtoh(output_array, output_gpu)

#     return output_array

# Example usage
# image = np.random.rand(512, 512).astype(np.float32)
# blurred_image = gaussian_blur_cuda(image)

print(get_gaussian_kernel(3, 1))



# t1 = 0
# t2 = 0
# t3 = 0
# t4 = 0

# kernel1 = []
# kernel2 = []
# kernel3 = []
# kernel4 = []

# n = 10000
# for i in range(0, n):
#     kernel = cv2.getGaussianKernel(3, 1)
    
#     start = cv2.getTickCount()
#     kernel1 = cv2.getGaussianKernel(3, 1) * cv2.getGaussianKernel(3, 1).T
#     end = cv2.getTickCount()
#     t1 += (end - start) / cv2.getTickFrequency()
    
#     start = cv2.getTickCount()
#     kernel2 = np.outer(cv2.getGaussianKernel(3, 1), cv2.getGaussianKernel(3, 1))
#     end = cv2.getTickCount()
#     t2 += (end - start) / cv2.getTickFrequency()
    
#     start = cv2.getTickCount()
#     kernel3 = np.outer(kernel, kernel)
#     end = cv2.getTickCount()
#     t3 += (end - start) / cv2.getTickFrequency()
    
#     start = cv2.getTickCount()
#     kernel4 = kernel * kernel.T
    
#     end = cv2.getTickCount()
#     t += (end - start) / cv2.getTickFrequency()

# # print with 8 decimal places, no scientific notation
# print(f"Time taken cv2: {t1/n:.8f}")
# print(f"Time taken npo: {t2/n:.8f}")
# print(f"Time taken def: {t3/n:.8f}")
# print(f"Time taken def: {t4/n:.8f}")