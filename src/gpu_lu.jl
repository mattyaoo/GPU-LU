using LinearAlgebra
using Metal

""" 
  lupartial(A, p) 
Takes the LU decomposition with partial pivoting mod p.
Assumes that: 
  * all entries are nonnegative integers,
  * A is invertible (and thus square),
  * n % 32 = 0
"""
function lupartial(A, p)
  n = size(A, 1)
  L = collect(1I(n))
  U = copy(A)
  P = collect(1:n)
  for k in 1:n
    pivot = choosepivot(U, k, p)
    swap!(U, k, pivot, P, p)
    normalizemodp!(U, L, k, p)
    updatesubmatrix!(U, L, k)
  end
  (L, U, P)
end

function inversemod5(a)
  table = [1, 3, 2, 4]
  return table[a]
end

function choosepivot(U, k, p)
  n = size(U, 1)
 
  for i in k:n
    if U[i, k] % p != 0
      return i
    end
  end
  throw(ArgumentError("Non invertible matrix"))
  

  #U[k:n, k] .%= p
  #argmax(U[k:n, k])+k-1
end

function swap!(U, k, pivot, P, p)
  n = size(U, 1)
  P[k], P[pivot] = P[pivot], P[k]
  U[k,:], U[pivot, :] = U[pivot, :], U[k, :]
  U[k, :] .%= p
end

function normalizemodp!(U, L, k, p)
   n = size(U, 1)
   inverse = inversemod5(U[k, k])
   U[k, k:n] .*= inverse
   U[k, k:n] .%= p
   L[k, k] = inverse
   L[k+1:n, k] .= (p .- (U[k+1:n, k] .% p)) .% p
   U[k+1:n, k] .= 0
end

function updatesubmatrix!(U, L, k)
  n = size(U, 1)
  #display(L[k+1:n, k])
  #display(U[k, k+1:n])
  #display(L[k+1:n, k] .* transpose(U[k, k+1:n]))
  U[k+1:n, k+1:n] .+= L[k+1:n, k] .* transpose(U[k, k+1:n])
end


########################
#
#
# Here we start the GPU section
# 
#
########################

const NUM_THREADS_PER_GROUP = 32

function swapkernel(U, k, pivot, p)
  grouppos = threadgroup_position_in_grid_1d()
  threadind = thread_position_in_threadgroup_1d()

  # grouppos needs -1 for indexing to work (i think)
  i = (grouppos-1) * NUM_THREADS_PER_GROUP + threadind
  temp = U[k, i]
  U[k, i] = U[pivot, i] # % p
  U[pivot, i] = temp

  return nothing
end

function swapgpu!(U, k, pivot, P, p)
  n = size(U, 1)
  P[k], P[pivot] = P[pivot], P[k]

  #gpu magic via the kernel we just wrote
  @metal groups=(n÷NUM_THREADS_PER_GROUP) threads=NUM_THREADS_PER_GROUP swapkernel(U, k, pivot, p)
end
#todo: test on larger matricies


function submatrix_kernel_naive!(U, L, k)
  n = size(U, 1)
  grouppos = threadgroup_position_in_grid_1d()
  threadind = thread_position_in_threadgroup_1d()

  t = (grouppos-1) * NUM_THREADS_PER_GROUP + threadind
  i = t + k
  c = L[i, k]
  for j in k:n
    s = U[k, j]
    U[i, j] += c * s
  end

  return
end
#todo: write the wrapper

function submatrix_gpu_naive!(U, L, k)
  n = size(U, 1)
  @metal groups=(n÷NUM_THREADS_PER_GROUP) threads=NUM_THREADS_PER_GROUP rows_kernel_naive!(U, L, k)
end

function rows_kernel_shared!(U, L, k)
  n = size(U, 1)
  grouppos = threadgroup_position_in_grid_1d()
  threadind = thread_position_in_threadgroup_1d()

  t = (grouppos-1) * NUM_THREADS_PER_GROUP + threadind
  i = t + k

  sharedvals = MtlThreadGroupArray(Int, NUMS_THREADS_PER_GROUP)
  c = L[i, k]
  for j in 0:(div(n-k, 32) + 1)
    shardvals[threadind] = U[k, k + j*32 + threadind]
    threadgroup_barrier()

    for l in 1:32
      U[i, k+j*32 +threadind] += c*sharedvals[l]
    end

    threadgroup_barrier()
  end

  return
end
#homework: debug

#thing that will go wrong: 
#  pad end of matrix with zeros to avoid OOB


####################
#
# Testing code
#
####################

#N = 32

#NxN Lower triangular matrix of ones
#A = zeros(Int, N, N)
#for i in 1:N
#  for j in 1:N
#    if i >= j
#      A[i, j] = 1
#    end
#  end
#end
