import OpenCV.getGaussianKernel
using OpenCV
using CUDA

function gaussian_kernel_unopt(inp, conv, out, width, height, apron)
    x, y = threadIdx().x, threadIdx().y
    X, Y = blockIdx().x, blockIdx().y

    if x <= width - 2 * apron && y <= height - 2 * apron
        out[x, y] = sum(inp[x:x+2*apron, y:y+2*apron] .* conv)
    end
    return
end

function gaussian_kernel(inp, conv, out, width, height, apron)
    # blockNum = (blockIdx().x - 1) + (blockIdx().y - 1) * gridDim().x # row first block numbering, zero-based
    blockNum::UInt16 = (blockIdx().x - 1) + (blockIdx().y - 1) * gridDim().x # row first block numbering, zero-based
    threadNum::UInt16 = (threadIdx().x - 1) + (threadIdx().y - 1) * blockDim().x # row first thread numbering in a block, zero-based

    # not sure why unsigned uint16 doesn't work
    threads::Int16 = blockDim().x * blockDim().y # total number of threads in a block

    if threadNum == 5 && blockNum == 0
        @cuprintln("threads: ", threads)
        # @cuprintln("threadNum: ", threadNum)
        # @cuprintln("blockNum: ", blockNum)
        # @cuprintln("sharedMemDim: ", (width, threads ÷ width))
        # @cuprintln("TypeOf threads: ", typeof(threads))
    end

    thisX::Int16 = 0 # one-based
    thisY::Int16 = 0 # one-based
    thisXY::Int32 = 0 # zero-based


    # Let's do the row first
    # if thread count is greater than one row
    if threads >= width

        # we'll do (width) * (threads ÷ width) pixels in a block
        data = CuDynamicSharedArray(Float32, (width, threads ÷ width))
        # XY = CuDynamicSharedArray(Int16, (threads, 2))
        # outData = CuDynamicSharedArray(Float32, threads)
        # thisPXs = CuDynamicSharedArray(Int32, (threads, 1))
        # data = @cuDynamicSharedMem(Float32, (width, threads ÷ width))
        # data = CuDynamicSharedArray(Float32, 5)
        # data = @cuDynamicSharedMem(float, 1)
        # data = @cuStaticSharedMem(Float32, 5)

        # "this" refers to the current pixel in the input image
        # "that" refers to the current pixel in the output image

        if threadNum < width * (threads ÷ width)
            thisPX = blockNum * width * (threads ÷ width) + threadNum # zero-based
            # thisPXs[threadNum+1] = blockNum * width * (threads ÷ width) + threadNum # zero-based
            if true
                this = thisPX % width + 1 # one-based
                thisY = (thisPX - thisX + 1) ÷ width + 1 # one-based
                # thisXY[threadNum+1, 1] = thisPX % width + 1
                # thisXY[threadNum+1, 2] = (thisPX - thisXY[threadNum+1, 1] + 1) ÷ width + 1


                if threadNum == 5 && blockNum == 0
                    @cuprintln("thisX: ", thisX)
                    @cuprintln("thisY: ", thisY)
                    @cuprintln("dataXY: ", (threadNum % width + 1, (threadNum - threadNum % width) ÷ width + 1))
                    # @cuprintln("img: ", inp[thisPX])
                end

                # coalesced memory access
                if (0 <= thisPX < width * height)
                    X = threadNum % width + 1
                    Y = (threadNum - (X - 1)) ÷ width + 1
                    data[X, Y] = inp[thisPX+1]
                    # data[1, 1]= inp[thisPX+1]
                end
                # data[6] = inp[thisX, thisY]

                # sync threads
                sync_threads()
            end

            # if threadNum == 5 && blockNum == 0
            #     data[1, 1]
            # end

            # thatX = thisX - apron # one-based
            # thatY = thisY # one-based
            # thatPX = (thatY - 1) * (width - 2 * apron) + thatX - 1 # zero-based

            if apron < threadNum % width + 1 <= width - apron && 0 <= thisPX < width * height
                # out[thatX, thatY] = sum(data[(threadNum%width+1-apron)*((threadNum-threadNum%width)÷width)+1:(threadNum%width+1+apron)*((threadNum-threadNum%width)÷width)+1] .* conv)
                # out[5, 5] = sum(data[1:15, 1] .* conv)
                # thisX = thisPX % width + 1
                # XY[threadNum+1, 1] = (thisPX % width + 1) - apron
                # XY[threadNum+1, 2] = (thisPX - (thisPX % width + 1) + 1) ÷ width + 1
                # outData[threadNum+1] = 0
                X = threadNum % width + 1
                Y = (threadNum - (X - 1)) ÷ width + 1
                outData = 0
                for i in -apron:apron
                    #     # don't use threadNum here
                    #     outData[threadNum+1] += 5
                    #     # outData[threadNum+1] += data[XY[threadNum+1, 1], XY[threadNum+1, 2]] * conv[i]
                    outData += data[X+i, Y] * conv[i+apron+1]

                    #     # solution to too many resources here would be to allocate a shared memory array that stores thisX and thisY or some variable.

                    #     # out[(thisPX%width+1)-apron, (thisPX-(thisPX%width+1)+1)÷width+1] += data[i, (threadNum-threadNum%width)÷width+1] * conv[i]
                end
                out[1, X-apron, blockNum*(threads÷width)+Y] = outData
                # out[XY[threadNum+1, 1], XY[threadNum+1, 2]] = 5#outData[threadNum+1]

                # out[thisX-apron, (thisPX-thisX+1)÷width+1] = sum(data[(threadNum%width+1-apron)*((threadNum-threadNum%width)÷width)+1:(threadNum%width+1+apron)*((threadNum-threadNum%width)÷width)+1] .* conv)
            end

        end
    else
        # we'll do threads pixels in a block
        data = CuDynamicSharedArray(Float32, threads)

        # total blocks in a row = width ÷ threads + 1
        # "this" refers to the current pixel in the input image
        thisY = blockNum ÷ (cld(width-apron, threads-apron)) + 1 # one-based
        thisX = (blockNum % (cld(width - apron, threads - apron))) * (threads-apron) + threadNum + 1 # one-based

        if 0 < thisX <= width && 0 < thisY <= height
            thisPX = (thisY - 1) * width + (thisX - 1) # zero-based
            data[threadNum+1] = inp[thisPX+1]
        end
        sync_threads()

        if thisX <= width-apron && 0 < thisY <= height && apron <= threadNum < threads - apron
            outData = 0.0
            for i in -apron:apron
                outData += data[threadNum+1+i] * conv[i+apron+1]
            end
            out[1, thisX-apron, thisY] = outData
        end
    end
    return
end

function convolve(img, schema)
    # function convolve(schema)
    if schema[:name] == "gaussian1D"
        sigma = convert(Float64, schema[:sigma])
        epsilon = haskey(schema, :epsilon) ? schema[:epsilon] : 0.0001
        apron = ceil(Int, sigma * sqrt(-2 * log(epsilon)))
        conv = reshape(getGaussianKernel(2 * apron + 1, sigma), 2 * apron + 1)

        println("Convolve with Gaussian1D")
        println("Sigma: ", sigma)
        println("Apron: ", apron)
        # println("Grids: ", grids)

        inp_GPU = CuArray(img)
        conv_GPU = CuArray(conv)
        width::Int32, height::Int32 = (0, 0)
        if length(size(img)) == 2
            width, height = size(img)
        else
            channel, width, height = size(img)
        end

        out_GPU = CuArray(zeros(Float32, 1, width - 2 * apron, height))

        
        blocks = (32, 1)
        grids = (cld(width, blocks[1]), cld(height, blocks[2]))
        println("Blocks: ", blocks)
        println("Grids: ", grids)
        
        if blocks[1] * blocks[2] >= width
            println("more than one row in a block")
            sharedMemSize = width * ((blocks[1] * blocks[2]) ÷ width) * (sizeof(Float32))
            println("Shared memory size: ", sharedMemSize / 1024, " KB")
            println("Convolution kernel: ", conv)
            @cuda blocks = grids threads = blocks shmem = sharedMemSize gaussian_kernel(inp_GPU, conv_GPU, out_GPU, width, height, apron)
        else
            println("many blocks in a row")
            sharedMemSize = blocks[1] * blocks[2] * (sizeof(Float32))
            println("Shared memory size: ", sharedMemSize / 1024, " KB")
            println("Convolution kernel: ", conv)
            @cuda blocks = grids threads = blocks shmem = sharedMemSize gaussian_kernel(inp_GPU, conv_GPU, out_GPU, width, height, apron)
        end

        return Array(out_GPU)
    end
end

# img = rand(1500, 512)
# load from assets/lenna.png, use OpenCV
img = OpenCV.imread("assets/lenna.png", OpenCV.IMREAD_GRAYSCALE)
println(size(img))

img = convert(Array{Float32}, img)
println(size(img))

schema = Dict(:name => "gaussian1D", :sigma => 2, :epsilon => 0.001)

out = convolve(reshape(img, size(img)...), schema)
# println(out)
println(size(out))
# save to assets/lenna_gaussian.png
# OpenCV.imwrite("assets/lenna_gaussian.png", out)
# uint8_array = UInt8.(round.(clamp.(out, 0, 1) .* 255))

# Convert the uint8 array to an OpenCV Mat
# out = reshape(out, 1, size(out, 1), size(out, 3))
mat_image = OpenCV.Mat(out)

OpenCV.imwrite("assets/lenna_gaussian.png", mat_image)
# println(out)