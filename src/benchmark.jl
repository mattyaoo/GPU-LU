using BenchmarkTools

include("gpu_lu.jl")


function run()
  n = 512
  A = zeros(Int, n, n)
  for i in 1:n
    A[n-i+1, i] = 1
  end
  @btime L, U, P = lupartial($A, 5)
end

run()

