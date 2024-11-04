#toying with the indexing of thread/group position
#
#Conclusion: everything in sight is one-indexed

using Metal

A = zeros(Int, 32)
dA = MtlArray(A)

const NUM_THREADS_PER_GROUP = 4
function indextest_kernel!(A)
  grouppos = threadgroup_position_in_grid_1d()
  threadind = thread_position_in_threadgroup_1d()
  i = (grouppos-1) * NUM_THREADS_PER_GROUP + threadind

  #A[i] = threadind + (grouppos-1)*1000 
  A[i] = i

  return nothing
end

function indextest!(A)
  n = length(A)
  @metal groups=(n÷NUM_THREADS_PER_GROUP) threads=NUM_THREADS_PER_GROUP indextest_kernel!(A)
end

indextest!(dA)
dA

