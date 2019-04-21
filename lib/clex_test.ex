defmodule ClexTest do
  @moduledoc """
  Taking a crack at implementing GEMM from this OpenCL tutorial:

  https://cnugteren.github.io/tutorial/pages/page1.html
  """

  import ClexTest.Utils
  alias Clex.CL10

  @myGEMM1 ~S"""
      // First naive implementation
      __kernel void myGEMM1(const int M, const int N, const int K,
                            const __global float* A,
                            const __global float* B,
                            __global float* C) {
          const int globalRow = get_global_id(0);
          const int globalCol = get_global_id(1);
          float acc = 0.0f;
          for (int k=0; k<K; k++) {
              acc += A[k*M + globalRow] * B[globalCol*K + k];
          }
          C[globalCol*M + globalRow] = acc;
    }
  """

  def test_utils() do
    m = Matrix.ident(3)
    shape = Matrix.size(m)

    b = list_to_float_bitstring(m, :float32)
    l = float_bitstring_to_list(b, :float32)
    list_to_matrix(l, shape, 2.0) |> IO.inspect()

    b = list_to_float_bitstring(m, :float64)
    l = float_bitstring_to_list(b, :float64)
    list_to_matrix(l, shape, 2.0) |> IO.inspect()
  end

  @doc ~S"""
  Kernel 1: Naive implementation
  """
  def test_GEMM1(device_type) do
    # Define matrix dims
    cdim = 2048
    m = cdim
    n = cdim
    k = cdim

    # Set up compute context
    IO.puts("Setting up compute context")
    {:ok, [platform | _]} = CL10.get_platform_ids()
    {:ok, devices} = CL10.get_device_ids(platform, device_type)
    {:ok, context} = CL10.create_context(devices)
    device = hd(devices)

    # Create input matrices and allocate buffer for output matrix
    IO.puts("Allocating buffers")
    a_mat = Matrix.ones(m, k)
    a = a_mat |> list_to_float_bitstring(:float32)
    a_size = byte_size(a)
    {:ok, a_mem} = CL10.create_buffer(context, [:read_write], a_size)

    b_mat = Matrix.ones(k, n)
    b = b_mat |> list_to_float_bitstring(:float32)
    b_size = byte_size(b)
    {:ok, b_mem} = CL10.create_buffer(context, [:read_write], b_size)

    #c is shape {m, n} with 32-bit floats
    c = Matrix.zeros(m, n) |> list_to_float_bitstring(:float32)
    c_size = byte_size(c)
    {:ok, c_mem} = CL10.create_buffer(context, [:read_write], c_size)

    # Initialize the kernel
    {:ok, program} = CL10.create_program_with_source(context, @myGEMM1)
    :ok = CL10.build_program(program, devices)
    {:ok, kernel} = CL10.create_kernel(program, 'myGEMM1')

    CL10.set_kernel_arg(kernel, 0, m)
    CL10.set_kernel_arg(kernel, 1, n)
    CL10.set_kernel_arg(kernel, 2, k)

    CL10.set_kernel_arg(kernel, 3, a_mem)
    CL10.set_kernel_arg(kernel, 4, b_mem)
    CL10.set_kernel_arg(kernel, 5, c_mem)

    # Create command queue for the kernel to execute on
    {:ok, queue} = CL10.create_queue(context, device)

    # Queue up buffer writes
    IO.puts("Writing buffers")
    {:ok, a_write_event} = CL10.enqueue_write_buffer(queue, a_mem, 0, a_size, a, [])
    {:ok, b_write_event} = CL10.enqueue_write_buffer(queue, b_mem, 0, b_size, b, [])
    {:ok, c_write_event} = CL10.enqueue_write_buffer(queue, c_mem, 0, c_size, c, [])

    IO.puts("Enqueueing kernel op")
#    {:ok, wg_info} = CL10.get_kernel_workgroup_info(kernel, device)
#    wg_size = Keyword.get(wg_info, :work_group_size, 1) |> IO.inspect()
    local = [1, 1]
    global = [m, n]
    {:ok, kernel_event} = CL10.enqueue_nd_range_kernel(queue, kernel, global, local, [a_write_event, b_write_event, c_write_event])

    IO.puts("Enqueueing read from output buffer")
    {:ok, event} = CL10.enqueue_read_buffer(queue, c_mem, 0, c_size, [kernel_event])

    IO.puts("Flushing queue, waiting for completion")
    CL10.flush(queue)
    [{:ok, output}] = CL10.wait_for_events([event]) |> IO.inspect()
    CL10.finish(queue)
    output |> float_bitstring_to_list(:float32) |> list_to_matrix({m, n}, 0.0)
  end

end
