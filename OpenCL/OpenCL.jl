
styl = String(read("style.css"))
HTML("$styl")

using Printf, Statistics
using OpenCL, PyPlot, BenchmarkTools, StatsBase

# Check on the number of OCL devices
using OpenCL
devices = cl.devices()

const ND = length(devices)

oclfile = pathof(OpenCL)

run(`head -14 $oclfile`)

oclsrc  = oclfile[1:end-10]

run(`ls -F $oclsrc`)

const mad2_kernel_src = """
__kernel void mad2(__global const float *a,
                   __global const float *b,
                   __global const float *c,
                   __global float *d)
{
  int gid = get_global_id(0);
  d[gid] = a[gid]*b[gid] + c[gid];
}
""";

a = randn(Float32, 100_000)
b = randn(Float32, 100_000)
c = randn(Float32, 100_000)
;

device, ctx, queue = cl.create_compute_context()
@printf "Device  : %s\nContext : %s\nQueue   : %s" device ctx queue

# default device returned is the first GPU

first(cl.devices(:gpu))

# create opencl buffer objects
# copies to the device initiated when the kernel function is called

a_buff = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=a)
b_buff = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=b)
c_buff = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=c)
d_buff = cl.Buffer(Float32, ctx, :w, length(a))
;

# build the program 
p = cl.Program(ctx, source=mad2_kernel_src) |> cl.build!

# construct a kernel object
mad2_kernel = cl.Kernel(p, "mad2")

# call the kernel object with global size 
# set to the size our arrays
mad2_kernel[queue, size(a)](a_buff, b_buff, c_buff, d_buff)

# perform a blocking read of the result from the device
r = cl.read(queue, d_buff);

# check to see if our result is what we expect!
[mean(r), std(r), skewness(r), kurtosis(r)]

describe(r)

run(`less ./RNGkernel.cl`);

mandel_source = """
    __kernel void mandelbrot(__global float2 *q,
                             __global ushort *output, 
                             ushort const maxiter)
{
  int gid = get_global_id(0);
  float nreal, real = 0;
  float imag = 0;
  output[gid] = 0;
  for(int curiter = 0; curiter < maxiter; curiter++) {
    nreal = real*real - imag*imag + q[gid].x;
    imag = 2*real*imag + q[gid].y;
    real = nreal;
    if (real*real + imag*imag > 4.0f)
      output[gid] = curiter;
  }
}""";

w = 4096 * 2;
h = 4096 * 2;
@printf("Size %i MB\n", sizeof(ComplexF32) * w * h / 1024 / 1024)

## q = [ComplexF32(r,i) for i=1:-(2.0/w):-1, r=-1.5:(3.0/h):1.5];

function build_q(w, h)
  y1 = -1.0
  y2 = 1.0
  x1 = -1.5
  x2 = 0.5
  q = Array{ComplexF32}(undef, h, w)
  for x in 1:w
    for y in 1:h
      xx = x1 + x * ((x2 - x1) / w)
      yy = y1 + y * ((y2 - y1) / h)
      @inbounds q[y, x] = ComplexF32(xx, yy)
    end
  end
  return q
end

q = build_q(w, h);

function mandel_opencl(q::Array{ComplexF32}, maxiter::Int64, device)
  ctx   = cl.Context(device)
  queue = cl.CmdQueue(ctx)
  out = Array{UInt16}(undef,size(q))
  q_buff = cl.Buffer(ComplexF32, ctx, (:r, :copy), hostbuf=q)
  o_buff = cl.Buffer(UInt16, ctx, :w, length(out))
  prg = cl.Program(ctx, source=mandel_source) |> cl.build!
  k = cl.Kernel(prg, "mandelbrot")
  queue(k, length(q), nothing, q_buff, o_buff, UInt16(maxiter))
  cl.copy!(queue, out, o_buff)
  return out
end

m = mandel_opencl(q, 150, device)
imshow(m, cmap="RdGy")

import OpenCL.cl.CLArray
## device, ctx, queue = cl.create_compute_context()

using LinearAlgebra

@time begin
    A = CLArray(queue, rand(Float32, 256, 128))
    B = cl.zeros(Float32, queue, 128, 256)
    ev = transpose!(B, A)
    cl.wait(ev)
end

A

aa = cl.to_host(A);
aa[1,2]

bb = cl.to_host(B);
bb[2,1]

aa' == bb

const INSTEPS = 512*512*512
const ITERS = 262144

in_nsteps = INSTEPS
niters    = ITERS

kernel = """
//------------------------------------------------------------------------------
//
// kernel:  pi
//
// Purpose: accumulate partial sums of pi comp
//
// input:  float step_size
//         int   niters per work item
//         local float* an array to hold sums from each work item
//
// output: partial_sums -- float vector of partial sums
//

void reduce(
   __local  float*,
   __global float*);

__kernel void pi(
   const int          niters,
   const float        step_size,
   __local  float*    local_sums,
   __global float*    partial_sums)
{
   int num_wrk_items  = get_local_size(0);
   int local_id       = get_local_id(0);
   int group_id       = get_group_id(0);

   float x, accum = 0.0f;
   int i,istart,iend;

   istart = (group_id * num_wrk_items + local_id) * niters;
   iend   = istart+niters;

   for(i= istart; i<iend; i++){
       x = (i+0.5f)*step_size;
       accum += 4.0f/(1.0f+x*x);
   }

   local_sums[local_id] = accum;
   barrier(CLK_LOCAL_MEM_FENCE);
   reduce(local_sums, partial_sums);
}

//------------------------------------------------------------------------------
//
// OpenCL function:  reduction
//
// Purpose: reduce across all the work-items in a work-group
//
// input : local  float*  an array to hold sums from each work item
// output: global float*  partial_sums  [float vector]
//

void reduce(
   __local  float*    local_sums,
   __global float*    partial_sums)
{
   int num_wrk_items  = get_local_size(0);
   int local_id       = get_local_id(0);
   int group_id       = get_group_id(0);

   float sum;
   int i;

   if (local_id == 0) {
      sum = 0.0f;
      for (i=0; i<num_wrk_items; i++) {
          sum += local_sums[i];
      }
      partial_sums[group_id] = sum;
   }
}
""";

pi_prog = cl.Program(ctx, source = kernel) |> cl.build!
pi_kernel = cl.Kernel(pi_prog, "pi")

work_group_size = device[:max_work_group_size]
nwork_groups = in_nsteps ÷ (work_group_size * niters)

if nwork_groups < 1
    nwork_groups = device[:max_compute_units]
    work_group_size = in_nsteps ÷ (nwork_groups * niters)
end

nsteps = work_group_size * niters * nwork_groups
step_size = 1.0 / nsteps

h_psum = Vector{Float32}(undef,nwork_groups)

println("$nwork_groups workgroup(s) of size $work_group_size.")
println("$nsteps integration steps")

d_partial_sums = cl.Buffer(Float32, ctx, :w, length(h_psum))

rtime = time()
global_size = (nwork_groups * work_group_size,)
local_size  = (work_group_size,)
localmem    = cl.LocalMem(Float32, work_group_size)

queue(pi_kernel, global_size, local_size, Int32(niters), Float32(step_size), localmem, d_partial_sums)
cl.copy!(queue, h_psum, d_partial_sums)

pi_res = round(sum(h_psum)*step_size, digits=6)
rtime = round(time() - rtime, digits=4)

println("The computation took $rtime secs")
println("pi = $pi_res for $nsteps steps")

using OpenCL, CLBlast, LinearAlgebra
const LA = LinearAlgebra

# setup data
α = 1.f0
β = 1.f0
A = rand(Float32, 10, 8)
B = rand(Float32, 8, 6)
C = zeros(Float32, 10, 6);

# transfer data
A_cl = cl.CLArray(queue, A)
B_cl = cl.CLArray(queue, B)
C_cl = cl.CLArray(queue, C);

# Compute using the CPU
LA.BLAS.gemm!('N', 'N', α, A, B, β, C)

# Compute using the GPU
CLBlast.gemm!('N', 'N', α, A_cl, B_cl, β, C_cl)

# Result is returned in C_cl
D = cl.to_host(C_cl)

C == D

ctxs = [cl.Context(devices[i]) for i = 1:ND]

const mad3_kernel = "
__kernel void mad3(
                    __global const float *a,
                    __global const float *b,
                    __global const float *x,
                    __global const float *y,
                    __global const float *z,
                    __global float *s)
    {
      int gid = get_global_id(0);
      s[gid] = a[gid]*x[gid]*x[gid] + b[gid]*y[gid] + z[gid];
    }
";

queues = [cl.CmdQueue(ctxs[i]) for i = 1:ND]
progs  = [cl.Program(ctxs[i], source=mad3_kernel) |> cl.build! for i = 1:ND]

ets = zeros(ND)
kerns  = [cl.Kernel(progs[i], "mad3") for i = 1:ND]

for n in (1, 10^3, 3*10^3, 10^4, 3*10^4, 10^5, 3*10^5, 10^6, 3*10^6, 10^7)
  
  a = rand(Float32, n);
  b = rand(Float32, n);
  x = rand(Float32, n);
  y = rand(Float32, n);
  z = rand(Float32, n);
  ## s = Array{Float32}(undef, n);

  for i = 1:ND
        
    a_buff = cl.Buffer(Float32, ctxs[i], (:r, :copy), hostbuf=a);
    b_buff = cl.Buffer(Float32, ctxs[i], (:r, :copy), hostbuf=b);
    x_buff = cl.Buffer(Float32, ctxs[i], (:r, :copy), hostbuf=x);
    y_buff = cl.Buffer(Float32, ctxs[i], (:r, :copy), hostbuf=y);
    z_buff = cl.Buffer(Float32, ctxs[i], (:r, :copy), hostbuf=z);
    s_buff = cl.Buffer(Float32, ctxs[i], :w, n)

    ets[i] = @elapsed begin
      kerns[i][queues[i], size(x)](a_buff, b_buff, x_buff, y_buff, z_buff, s_buff);
    end
  end
        
  if n == 1
    @printf "    Size   Intel HD620   NVidia MX150\n"
  else
    @printf "%8d   %9.5f     %9.5f\n" n ets[1] ets[3]
  end
end


using BenchmarkTools

device = devices[1]
@benchmark mandel_opencl(q, 150, device) samples=5

device = devices[2]
@benchmark mandel_opencl(q, 150, device) samples=5

device = devices[3]
@benchmark mandel_opencl(q, 150, device) samples=5


