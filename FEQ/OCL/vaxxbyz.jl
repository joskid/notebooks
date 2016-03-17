#! /Users/malcolm/bin/julia
#
import OpenCL
cl = OpenCL

const vaxbyz_kernel = "
__kernel void vaxbyz(
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
"

for n in (1, 1000, 10_000, 50_000, 1_000_000, 5_000_000, 25_000_000, 50_000_000)
  
  a = rand(Float32, n);
  b = rand(Float32, n);
  x = rand(Float32, n);
  y = rand(Float32, n);
  z = rand(Float32, n);
  s = Array(Float32, n);

  device, ctx, queue = cl.create_compute_context();

  a_buff = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=a);
  b_buff = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=b);
  x_buff = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=x);
  y_buff = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=y);
  z_buff = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=z);
  s_buff = cl.Buffer(Float32, ctx, :w, n)

  p = cl.Program(ctx, source=vaxbyz_kernel) |> cl.build!
  k = cl.Kernel(p, "vaxbyz");

  cl.call(queue, k, size(x), nothing, a_buff, b_buff, x_buff, y_buff, z_buff, s_buff);
  t0 = @elapsed s0 = cl.read(queue, s_buff);
  
  s1 = zeros(n);
  t1 = @elapsed begin
    for i = 1:length(x)
      s1[i] = a[i]*x[i]*x[i] + b[i]*y[i] + z[i]
    end
  end
  if n > 1
    @printf "%10d  : %8.5f  :  %8.5f\n" n t0 t1;
  else
    @printf "     Loops  :      GPU  :    Native\n";
  end
end  
println();
