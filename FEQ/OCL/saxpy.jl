import OpenCL
cl = OpenCL

const saxpy_kernel = "
__kernel void saxpy(float a,
                    __global const float *x,
                    __global const float *y,
                    __global float *z)
    {
      int gid = get_global_id(0);
      z[gid] = a*x[gid] + y[gid];
    }
"

n = 1000000
a = 1.5
x = float32(randn(n))
y = float32(randn(n))

device, ctx, queue = cl.create_compute_context()

x_buff = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=x)
y_buff = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=y)
z_buff = cl.Buffer(Float32, ctx, :w, n)

p = cl.Program(ctx, source=saxpy_kernel) |> cl.build!
k = cl.Kernel(p, "saxpy")
cl.call(queue, k, size(x), nothing, float32(a), x_buff, y_buff, z_buff)

println("\nSAXPY using OpenCL\n");
@printf "n = %d\n\n" n

tic()
cl.call(queue, k, size(x), nothing, float32(a), x_buff, y_buff, z_buff)
toc()
r = cl.read(queue, z_buff)
@printf "Mean = %7.4f\n" mean(r)
@printf "Var  = %7.4f\n\n" var(r)

println("\n..... using Native Julia");
s = zeros(n);
tic()
for i = 1:length(x)
  s[i] = a*x[i] + y[i]
end
toc()
@printf "Mean = %7.4f\n" mean(s)
@printf "Var  = %7.4f\n\n" var(s)

