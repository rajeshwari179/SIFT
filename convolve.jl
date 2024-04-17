using OpenCV
using CUDA

function row_kernel_strips(inp, conv, out, width, height, apron, print)
    blockNum::UInt32 = (blockIdx().x - 1) + (blockIdx().y - 1) * gridDim().x # column major block numbering, zero-based
    threadNum::UInt16 = (threadIdx().x - 1) + (threadIdx().y - 1) * blockDim().x # column major thread numbering in a block, zero-based

    # not sure why unsigned uint16 doesn't work
    threads::Int16 = blockDim().x * blockDim().y # total number of threads in a block

    # "this" refers to the current pixel in the input image
    # "that" refers to the current pixel in the output image
    thisX::Int16 = 0 # one-based
    thisY::Int32 = 0 # one-based
    thisPX::Int32 = 0 # zero-based

    # Let's do the row first
    # if thread count is greater than one row
    # we'll do (width) * (threads ÷ width) pixels in a block
    data = CuDynamicSharedArray(Float32, (width, threads ÷ width))

    if threadNum < width * (threads ÷ width)
        thisPX = blockNum * width * (threads ÷ width) + threadNum # zero-based
        if true
            thisX = thisPX % width + 1 # one-based
            thisY = (thisPX - thisX + 1) ÷ width + 1 # one-based
            # coalesced memory access
            if (0 <= thisPX < width * height)
                X::Int16 = threadNum % width + 1
                Y::Int16 = (threadNum - (X - 1)) ÷ width + 1
                data[X, Y] = inp[thisPX+1]
            end
            sync_threads()
        end
        # thatX = thisX - apron # one-based
        # thatY = thisY # one-based
        if apron < threadNum % width + 1 <= width - apron && 0 <= thisPX < width * height
            outData::Float32 = 0
            for i in -apron:apron
                outData += data[threadNum%width+1+i, (threadNum-(X-1))÷width+1] * conv[i+apron+1]
            end
            out[1, threadNum%width+1-apron, blockNum*(threads÷width)+(threadNum-(X-1))÷width+1] = outData
        end

    end
    return
end

function row_kernel_strip(inp, conv, out, width, height, apron, print)
    blockNum::UInt32 = (blockIdx().x - 1) + (blockIdx().y - 1) * gridDim().x # column major block numbering, zero-based
    threadNum::UInt16 = (threadIdx().x - 1) + (threadIdx().y - 1) * blockDim().x # column major thread numbering in a block, zero-based
    # not sure why unsigned uint16 doesn't work
    threads::Int16 = blockDim().x * blockDim().y # total number of threads in a block

    # "this" refers to the current pixel in the input image
    # "that" refers to the current pixel in the output image
    thisX::Int16 = 0 # one-based
    thisY::Int32 = 0 # one-based
    thisPX::Int32 = 0 # zero-based

    # Let's do the row first
    # we'll do threads pixels in a block
    data = CuDynamicSharedArray(Float32, threads)

    # total blocks in a row = width ÷ threads + 1
    # "this" refers to the current pixel in the input image
    thisY = blockNum ÷ (cld(width - 2 * apron, threads - 2 * apron)) + 1 # one-based
    thisX = (blockNum % (cld(width - 2 * apron, threads - 2 * apron))) * (threads - 2 * apron) + threadNum + 1 # one-based

    if 0 < thisX <= width && 0 < thisY <= height
        thisPX = (thisY - 1) * width + (thisX - 1) # zero-based
        data[threadNum+1] = inp[thisPX+1]
    end
    sync_threads()

    if thisX <= width - apron && 0 < thisY <= height && apron <= threadNum < threads - apron
        outData = 0.0
        for i in -apron:apron
            outData += data[threadNum+1+i] * conv[i+apron+1]
        end
        out[1, thisX-apron, thisY] = outData
    end
    return
end

function col_kernel(inp, conv, out, width, fullHeight, height, apron)
    blockNum::UInt32 = (blockIdx().x - 1) * gridDim().y + (blockIdx().y - 1) # row first block numbering, zero-based
    threadNum::UInt16 = (threadIdx().x - 1) + (threadIdx().y - 1) * blockDim().x # row first thread numbering in a block, zero-based
    threads::Int16 = blockDim().x * blockDim().y # total number of threads in a block
    threadsX::Int8 = blockDim().x

    # "this" refers to the current pixel in the input image
    # "that" refers to the current pixel in the output image
    if threads <= fullHeight
        data = CuDynamicSharedArray(Float32, (threadsX, threads ÷ threadsX))

        blocksInARow::Int16 = cld(width - 2 * apron, threadsX) # this is the number of blocks in a row of the image (width)
        blocksInAColumn::Int32 = cld(height - 2 * apron, threads ÷ threadsX - 2 * apron) # this is the number of blocks in a column of the image (height)
        blocksInAnImage::Int32 = blocksInARow * blocksInAColumn

        thisImage::Int8 = blockNum ÷ blocksInAnImage # zero-based
        thisBlockNum::Int32 = blockNum % blocksInAnImage # zero-based

        thisX::Int16 = (thisBlockNum ÷ blocksInAColumn) * threadsX + threadNum % threadsX + 1 # one-based
        thisY::Int32 = thisImage * height + (thisBlockNum % blocksInAColumn) * (threads / threadsX - 2 * apron) + threadNum ÷ threadsX + 1 # one-based

        thisPX::Int32 = (thisY - 1) * (width - 2 * apron) + (thisX - 1) # zero-based

        if 0 <= thisPX < (width - 2 * apron) * fullHeight
            data[threadNum+1] = inp[thisPX+1]
        end
        sync_threads()

        if 0 < thisX <= (width - 2 * apron) && apron < thisY <= fullHeight - apron && apron <= (threadNum ÷ threadsX) < threads / threadsX - apron
            outData = 0.0
            for i in -apron:apron
                outData += data[threadNum+i*threadsX+1] * conv[i+apron+1]
            end
            out[1, thisX, thisY-(thisImage)*2*apron-apron] = outData
        end
    end
    return
end

function convolve(img, schema, imgHeight, print=1)
    if schema[:name] == "gaussian1D"
        sigma = convert(Float64, schema[:sigma])
        epsilon = haskey(schema, :epsilon) ? schema[:epsilon] : 0.0001
        apron = ceil(Int, sigma * sqrt(-2 * log(epsilon)))
        conv = reshape(OpenCV.getGaussianKernel(2 * apron + 1, sigma), 2 * apron + 1)

        if print == 1
            println("Convolve with Gaussian1D")
            println("Sigma: ", sigma)
            println("Epsilon: ", epsilon)
            println("Apron: ", apron)
        end
        width::Int32, height::Int32 = (0, 0)
        if length(size(img)) == 2
            width, height = size(img)
        else
            _, width, height = size(img)
        end

        if print == 1
            println("Width: ", width, ", Height: ", height)
        end

        blocks_row = (32, 32)
        blocks_col = (32, 32)
        # blocks_col = (32, 28)
        while blocks_col[2] - 2 * apron < 0 && blocks_col[1] > 4
            blocks_col = (blocks_col[1] ÷ 2, blocks_col[2] * 2)
        end
        grids_row = cld((width - 2 * apron), blocks_row[1] * blocks_row[2] - 2 * apron) * height
        grids_col = (cld(width - 2 * apron, blocks_col[1]), cld((height - 2 * apron), blocks_col[2] - 2 * apron) * height ÷ imgHeight)
        if print == 1
            println("Blocks: ", blocks_row)
            println("Blocks: ", blocks_col)
            println("Grids, col: $grids_col, row: $grids_row")
        end
        inp_GPU = CuArray(img)
        conv_GPU = CuArray(conv)
        out_GPU = CuArray(zeros(Float32, 1, width - 2 * apron, height))
        sharedMemSize = blocks_row[1] * blocks_row[2] * (sizeof(Float32)) # shared memory size in bytes
        if blocks_row[1] * blocks_row[2] >= width
            if print == 1
                println("more than one row in a block")
                println("Shared memory size: ", sharedMemSize / 1024, " KB")
                println("Convolution kernel: ", conv)
            end
            @cuda blocks = grids_row threads = blocks_row shmem = sharedMemSize row_kernel_strips(inp_GPU, conv_GPU, out_GPU, width, height, apron, print)
            CUDA.unsafe_free!(inp_GPU)
            @cuda blocks = grids_col threads = blocks_col shmem = sharedMemSize col_kernel(out_GPU, conv_GPU, inp_GPU, width, height, imgHeight, apron)
        else
            if print == 1
                println("many blocks in a row")
                println("Shared memory size: ", sharedMemSize / 1024, " KB")
                println("Convolution kernel: ", conv)
            end
            @cuda blocks = grids_row threads = blocks_row shmem = sharedMemSize row_kernel_strip(inp_GPU, conv_GPU, out_GPU, width, height, apron, print)
            @cuda blocks = grids_col threads = blocks_col shmem = sharedMemSize maxregs = 32 col_kernel(out_GPU, conv_GPU, inp_GPU, width, height, imgHeight, apron)
        end
        if print == 1
            println("Done")
        end
        return Array(inp_GPU), Array(out_GPU)
        return 1, 2
    end
end

function convolve1(inp_GPU, schema, imgHeight, width, height, print=1)
    if schema[:name] == "gaussian1D"
        sigma = convert(Float64, schema[:sigma])
        epsilon = haskey(schema, :epsilon) ? schema[:epsilon] : 0.0001
        apron = ceil(Int, sigma * sqrt(-2 * log(epsilon)))
        conv = reshape(OpenCV.getGaussianKernel(2 * apron + 1, sigma), 2 * apron + 1)

        if print == 1
            println("Convolve with Gaussian1D")
            println("Sigma: ", sigma)
            println("Epsilon: ", epsilon)
            println("Apron: ", apron)
        end

        if print == 1
            println("Width: ", width, ", Height: ", height)
        end

        blocks_row = (32, 32)
        blocks_col = (32, 28)
        while blocks_col[2] - 2 * apron < 0 && blocks_col[1] > 4
            blocks_col = (blocks_col[1] ÷ 2, blocks_col[2] * 2)
        end
        grids_row = cld((width - 2 * apron), blocks_row[1] * blocks_row[2] - 2 * apron) * height
        grids_col = (cld(width - 2 * apron, blocks_col[1]), cld((height - 2 * apron), blocks_col[2] - 2 * apron) * height ÷ imgHeight)
        if print == 1
            println("Blocks: ", blocks_row)
            println("Blocks: ", blocks_col)
            println("Grids, col: $grids_col, row: $grids_row")
        end
        conv_GPU = CuArray(conv)
        out1_GPU = CuArray(zeros(Float32, 1, width - 2 * apron, height))
        out2_GPU = CuArray(zeros(Float32, 1, width - 2 * apron, height - 2 * apron * height ÷ imgHeight))
        sharedMemSize = blocks_row[1] * blocks_row[2] * (sizeof(Float32)) # shared memory size bytes
        if blocks_row[1] * blocks_row[2] >= width
            if print == 1
                println("more than one row in a block")
                println("Shared memory size: ", sharedMemSize / 1024, " KB")
                println("Convolution kernel: ", conv)
            end
            @cuda blocks = grids_row threads = blocks_row shmem = sharedMemSize row_kernel_strips(inp_GPU, conv_GPU, out1_GPU, width, height, apron, print)
            @cuda blocks = grids_col threads = blocks_col shmem = sharedMemSize col_kernel(out1_GPU, conv_GPU, out2_GPU, width, height, imgHeight, apron)
        else
            if print == 1
                println("many blocks in a row")
                println("Shared memory size: ", sharedMemSize / 1024, " KB")
                println("Convolution kernel: ", conv)
            end
            @cuda blocks = grids_row threads = blocks_row shmem = sharedMemSize row_kernel_strip(inp_GPU, conv_GPU, out1_GPU, width, height, apron, print)
            @cuda blocks = grids_col threads = blocks_col shmem = sharedMemSize col_kernel(out1_GPU, conv_GPU, out2_GPU, width, height, imgHeight, apron)
        end
        if print == 1
            println("Done")
        end
        return Array(out2_GPU), Array(out1_GPU)
        return 1, 2
    end
end

# imgNum = 16
nimages = 5
println("Here we go!")
# for nimages in 1:imgNum
img = []
imgHeight = 0
time_taken = 0
begin
    for i in 1:nimages
        img_temp = OpenCV.imread("assets/DJI_20240328_234918_14_null_beauty.mp4_frame_$i.png", OpenCV.IMREAD_GRAYSCALE)
        img_temp = convert(Array{Float32}, img_temp)
        if i == 1
            global img = img_temp
            global imgHeight = size(img_temp, 3)
        else
            global img = cat(img, img_temp, dims=3)
        end
    end

    start = time()
    mat_image = OpenCV.Mat(img)
    OpenCV.imwrite("assets/gaussian_before.png", mat_image)

    schema = Dict(:name => "gaussian1D", :sigma => 1.6, :epsilon => 0.1725)

    out2, out1 = convolve(img, schema, imgHeight, 1)

    schema1 = Dict(:name => "gaussian1D", :sigma => 1.6, :epsilon => 0.1725)
    schema2 = Dict(:name => "gaussian1D", :sigma => 2.2627, :epsilon => 0.1725)
    schema3 = Dict(:name => "gaussian1D", :sigma => 3.2, :epsilon => 0.1725)
    schema4 = Dict(:name => "gaussian1D", :sigma => 4.5254, :epsilon => 0.1725)
    schema5 = Dict(:name => "gaussian1D", :sigma => 6.4, :epsilon => 0.1725)

    iterations = 5
    for i in 1:iterations
        # try
        start_t = time()
        out2, out1 = convolve(img, schema1, imgHeight, 0)
        out2, out1 = convolve(img, schema2, imgHeight, 0)
        out2, out1 = convolve(img, schema3, imgHeight, 0)
        out2, out1 = convolve(img, schema4, imgHeight, 0)
        out2, out1 = convolve(img, schema5, imgHeight, 0)
        end_t = time()
        global time_taken += end_t - start_t
        if i == iterations
            mat_image1 = OpenCV.Mat(out1)
            mat_image2 = OpenCV.Mat(out2)
            OpenCV.imwrite("assets/gaussian_1.png", mat_image1)
            OpenCV.imwrite("assets/gaussian_2.png", mat_image2)
        end
        # if i % (iterations ÷ 20) == 0
        #     println("Iteration: ", i)
        #     CUDA.memory_status()
        # end
        # if i > 75
        #     println("Iteration: ", i)
        #     CUDA.memory_status()
        # end
        # catch e
        #     println("Error: ", e)
        #     continue
        # end
    end
    println("NO L2 CACHE: Time taken per iteration: ", time_taken / (iterations * nimages), " seconds per image when $nimages images are processed at once")
end