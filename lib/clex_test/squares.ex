defmodule ClexTest.Squares do
  @moduledoc false

  import ClexTest.Utils
  alias Clex.CL10

  @square ~S"""
    __kernel void square( __global float* input,
                          __global float* output,
                          const int count)
    {
       int i = get_global_id(0);
       if (i < count)
          output[i] = input[i]*input[i];
    }
  """

  def run(device_type) do
    # Set up compute context
    IO.puts("Setting up compute context")
    {:ok, [platform | _]} = CL10.get_platform_ids()
    {:ok, devices} = CL10.get_device_ids(platform, device_type)
    {:ok, context} = CL10.create_context(devices)
    device = hd(devices)

    num_vals = 1024
    input = List.duplicate(2, num_vals) |> IO.inspect() |> list_to_float_bitstring(:float32) |> IO.inspect()
    num_bytes = byte_size(input)
    {:ok, input_mem} = CL10.create_buffer(context, [:read_only], num_bytes)
    {:ok, output_mem} = CL10.create_buffer(context, [:write_only], num_bytes)

    {:ok, queue} = CL10.create_queue(context, device)
    {:ok, program} = CL10.create_program_with_source(context, @square)
    :ok = CL10.build_program(program, devices)
    {:ok, kernel} = CL10.create_kernel(program, 'square')

    CL10.set_kernel_arg(kernel, 0, input_mem)
    CL10.set_kernel_arg(kernel, 1, output_mem)
    CL10.set_kernel_arg(kernel, 2, num_vals)

    {:ok, event1} = CL10.enqueue_write_buffer(queue, input_mem, 0, num_bytes, input, [])

    {:ok, wg_info} = CL10.get_kernel_workgroup_info(kernel, device)
    local = Keyword.get(wg_info, :work_group_size, 1)
    global = num_vals
    {:ok, event2} = CL10.enqueue_nd_range_kernel(queue, kernel, [global], [local], [event1])

    {:ok, event3} = CL10.enqueue_read_buffer(queue, output_mem, 0, num_bytes, [event2])

    CL10.flush(queue)
    CL10.wait_for_events([event1, event2])
    [event3res] = CL10.wait_for_events([event3])
    {:ok, event3res_data} = event3res
    event3res_data |> IO.inspect() |> float_bitstring_to_list(:float32)
  end

end
