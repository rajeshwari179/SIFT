using NearestNeighbors

length_of_video = 1*60*60 # seconds
interest_points = rand(Float32, 128, 50 * length_of_video * 30)

start_t = time()
kdtree = KDTree(interest_points, leafsize=10, reorder=true)
end_t = time()

println("KDTree construction time: ", end_t - start_t, " seconds")