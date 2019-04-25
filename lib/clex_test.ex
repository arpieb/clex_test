defmodule ClexTest do
  @moduledoc """
  Taking a crack at implementing GEMM from this OpenCL tutorial:

  https://cnugteren.github.io/tutorial/pages/page1.html
  """

  @cdim 2048

  import ClexTest.Utils
  alias Clex.CL10
  alias ClexTest.Kernel.MyGEMM1

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
  def test_GEMM(device_type) do
    # Define matrix dims
    cdim = @cdim
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
    {:ok, program} = CL10.create_program_with_source(context, MyGEMM1.source())
    :ok = CL10.build_program(program, devices)
    {:ok, kernel} = CL10.create_kernel(program, MyGEMM1.name())

#    CL10.set_kernel_arg(kernel, 0, m)
#    CL10.set_kernel_arg(kernel, 1, n)
#    CL10.set_kernel_arg(kernel, 2, k)
#    CL10.set_kernel_arg(kernel, 3, a_mem)
#    CL10.set_kernel_arg(kernel, 4, b_mem)
#    CL10.set_kernel_arg(kernel, 5, c_mem)
    Clex.Utils.set_kernel_args(kernel, [m, n, k, a_mem, b_mem, c_mem])

    # Create command queue for the kernel to execute on
    {:ok, queue} = CL10.create_queue(context, device)

    num_runs = 100
    start = System.monotonic_time()
    for _ <- Range.new(1, num_runs) do
      # Queue up buffer writes
      #IO.puts("Writing buffers")
      {:ok, a_write_event} = CL10.enqueue_write_buffer(queue, a_mem, 0, a_size, a)
      {:ok, b_write_event} = CL10.enqueue_write_buffer(queue, b_mem, 0, b_size, b)
      {:ok, c_write_event} = CL10.enqueue_write_buffer(queue, c_mem, 0, c_size, c)

      #IO.puts("Enqueueing kernel op")
      {:ok, wg_info} = CL10.get_kernel_workgroup_info(kernel, device)
      local = get_local(wg_info, device_type)
      global = [m, n]
      {:ok, kernel_event} = CL10.enqueue_nd_range_kernel(queue, kernel, global, local, [a_write_event, b_write_event, c_write_event])

      #IO.puts("Enqueueing read from output buffer")
      {:ok, event} = CL10.enqueue_read_buffer(queue, c_mem, 0, c_size, [kernel_event])

      #IO.puts("Flushing queue, waiting for completion")
      CL10.flush(queue)
      [{:ok, _output}] = CL10.wait_for_events([event])
    end
    elapsed = (System.monotonic_time() - start) |> System.convert_time_unit(:native, :millisecond)
    IO.puts("Average exec time for #{num_runs} runs: #{elapsed / num_runs}ms per run")

    # Perform cleanup
    CL10.finish(queue)

    CL10.release_kernel(kernel)
    CL10.release_program(program)
    CL10.release_mem_object(a_mem)
    CL10.release_mem_object(b_mem)
    CL10.release_mem_object(c_mem)
    CL10.release_queue(queue)
    CL10.release_context(context)
    # output |> float_bitstring_to_list(:float32) |> list_to_matrix({m, n}, 0.0)

    :ok
  end

  defp get_local(wg_info, device_type) do
    #case Keyword.get(wg_info, :work_group_size, 1) do
    case device_type do
      :cpu -> [1, 1]
      _ -> [16, 16]
    end
  end

  def test_Matrex() do
    # Define matrix dims
    cdim = @cdim
    m = cdim
    n = cdim
    k = cdim

    a_mat = Matrex.magic(cdim)
    b_mat = Matrex.magic(cdim)

    num_runs = 100
    start = System.monotonic_time()
    for _ <- Range.new(1, num_runs) do
      Matrex.multiply(a_mat, b_mat)
    end
    elapsed = (System.monotonic_time() - start) |> System.convert_time_unit(:native, :millisecond)
    IO.puts("Average exec time for #{num_runs} runs: #{elapsed / num_runs}ms per run")

    :ok
  end

end
