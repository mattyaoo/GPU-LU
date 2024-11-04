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
  dL = MtlArray(collect(1I(n)))
  dU = MtlArray(copy(A))
  P = collect(1:n)
  for k in 1:n
    pivot = choosepivot(dU, k, p)
    swap!(dU, k, pivot, P, p)
    normalizemodp!(dU, dL, k, p)
    updatesubmatrix!(dU, dL, k)
  end
  (Matrix(dL), Matrix(dU), P)
end

function inversemod5(a)
  table = [1, 3, 2, 4]
  return table[a]
end

#not sure how fast argmax is on GPU
function choosepivot(U, k, p)
  n = size(U, 1)
  i = argmax(U[k:n, k] .% p)

  return i+k-1
end

function swap!(U, k, pivot, P, p)
  n = size(U, 1)
  P[k], P[pivot] = P[pivot], P[k]

  temp = U[k, :]
  U[k,:]= U[pivot, :] 
  U[pivot, :] = temp
  U[k, :] .%= p
end

function normalizemodp!(U, L, k, p)
   n = size(U, 1)
   
   #horrible scalar indexing hack lol
   Metal.@allowscalar inverse = inversemod5(U[k, k])
   U[k, k:n] .*= inverse
   U[k, k:n] .%= p

   #same here
   L[k:k, k:k] .= inverse
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
