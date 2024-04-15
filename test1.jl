using CUDA
using BenchmarkTools

arr = rand(32, 32, 10000)

function mykernel1(inp)
    x = threadIdx().x
    y = threadIdx().y
    z = blockIdx().x

    if x <= 32 && y <= 32
        @inbounds inp[x, y, z] += 1
    end

    return
end

arr_GPU = CuArray(arr)

@cuda threads = (32, 32) blocks = 10000 mykernel1(arr_GPU)

time_taken = 0
for i in 1:100
    start_t = time()
    global arr_GPU = CuArray(arr)
    @cuda threads = (32, 32) blocks = 10000 mykernel1(arr_GPU)
    end_t = time()
    global time_taken += (end_t - start_t)
end

println("Time: ", time_taken, " s")


function mykernel2(inp)
    x = threadIdx().x
    y = threadIdx().y
    z = blockIdx().x

    if x <= 32 && y <= 32
        @inbounds inp[y, x, z] += 1
    end

    return
end

arr_GPU = CuArray(arr)

@cuda threads = (32, 32) blocks = 10000 mykernel2(arr_GPU)

time_taken = 0
for i in 1:100
    start_t = time()
    global arr_GPU = CuArray(arr)
    @cuda threads = (32, 32) blocks = 10000 mykernel2(arr_GPU)
    end_t = time()
    global time_taken += (end_t - start_t)
end

println("Time: ", time_taken, " s")