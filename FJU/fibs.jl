
function fac(n::Integer)
  @assert n > 0
  return (n == 1 ? 1 : n*fac(n-1))
end

------------------------------------------------

function fib(n::Integer)
  @assert n > 0
  if n < 3 
    return 1
  else
    fib(n-1) + fib(n-2)
  end
end

function fib1(n::Integer)
  a = Array(typeof(n),n)
  a[1] = 1
  a[2] = 1
  for i = 3:n
    a[i] = a[i-1] + a[i-2]
  end
  a[n]
end

function fib2(n::Integer)
  (a, b) = big(0, 1)
  while n > 0
    (a, b) = (b, a+b)
    n -= 1
  end
  return a
end

fib_tail(a,b,n) = (n > 1) ? fib_tail(b, a+b, n-1) : a
fib3(n) = fib_tail(1, 1, n)

fib1(95)
fib1(big(95))
@time fib1(big(402))

--------------------------------

function fib(n::Integer)
  @assert n >= 0
  a = Array(typeof(n),n)
  a[0] = 0
  a[1] = 1
  i = 2:n
    a[i] = a[i-1] + a[i-2]
  end
  return a[n]
end

fib(0)

function fib(n::Integer)
  @assert n > 0
  a = Array(typeof(n),n)
  a[1] = 1
  for i = 2:n
    a[i] = a[i-1] + a[i-2]
  end
  return a[n]
end

function fib(n::Integer)
  @assert n >= 0
  a = Array(typeof(n),n)
  a[0] = 0
  a[1] = 1
  for i = 2:n
    a[i] = a[i-1] + a[i-2]
  end
  return a[n]
end

function fib(n::Integer)
  @assert n > 0
  if n < 3 
    return 1
  else
    fib(n-1) + fib(n-2)
  end
end

----------------------------------------------------------------------------------

http://www.csd.uwo.ca/~moreno/cs2101a_moreno/Parallel_computing_with_Julia.pdf


CPU_CORES;  # => 4
or
ccall(:jl_cpu_cores, Int32, ())

addprocs(3)
# addprocs(["lister","rimmer","cat","kryten"]))

@everywhere function fib(n)
    if (n < 2) then
        return n
    else 
        return fib(n-1) + fib(n-2)
    end
end

z = @spawn fib(35); fetch(z)

@everywhere function fib_parallel(n)
    if (n < 40) then
        return fib(n)
    else
        x = @spawn fib_parallel(n-1)
        y = fib_parallel(n-2)
    	return fetch(x) + y
	end 
end

# @everywhere using MyFibs

@time [fib(i) for i=35:44];
@time [fib_parallel(i) for i=35:44];



