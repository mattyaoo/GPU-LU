using BenchmarkTools

#include("gpu_lu.jl")


function runbenchmark(n=512)
  A = zeros(Int, n, n)
  #lower triangular matrix of ones
  #fast until about 2^14 (takes about 2s)
  for i in 1:n
    for j in 1:n
      if i >= j
        A[i, j] = 1
      end
    end
  end
  print("Starting benchmark! \n")
  bench = @benchmark Metal.@sync begin
    (L, U, P) = lupartial($A, 5)
  end
end

#test swap on CPU vs GPU
#
#Data: 
#
#   n      |  GPU / CPU
#   32     |   1924
#   64     |   1051
#   128    |   530
#   256    |   76
#   512    |   47
#   1024   |   21
#   2048   |   4.63
#   4096   |   5.17
#   8192   |   2.61
#   16384  |   .846
#   32768  |   .999

function timeswap(n=32)
  A = rand(Int, n, n)
  dA = MtlArray(A)
  P = collect(1:n)

  cpubench = @benchmark swap!($A, 1, 2, $P, 5)
  gpubench = @benchmark Metal.@sync swapgpu!($dA, 1, 2, $P, 5)

  cputime = median(cpubench).time
  gputime = median(gpubench).time

  println("CPU time: ", cputime)
  println("GPU time: ", gputime)
  println("GPU / CPU: ", gputime / cputime)
end
