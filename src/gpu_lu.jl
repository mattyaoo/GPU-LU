using LinearAlgebra

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
    #display(U)
    #display(L)
    updatesubmatrix!(U, L, k)
    #display(U)
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


