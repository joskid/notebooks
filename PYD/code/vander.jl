function vander(x, N::Int)
  x = convert(AbstractVector, x)
  M = length(x)
  v = Array(promote_type(eltype(x),Int), M, N)
  if N > 0 
    v[:, 1] = 1
  end
  if N > 1
    for i = 2:N
      v[:,i] = x
    end
    accumulate(*,v,v)
  end
  return v
end

function accumulate(op, input, output)
  M, N = size(input)
  for i = 2:N
    for j = 1:M
      output[j,i] = op(input[j,i], input[j,i-1])
    end
  end
end

