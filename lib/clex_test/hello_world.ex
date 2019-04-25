defmodule ClexTest.HelloWorld do
  @moduledoc false

  import ClexTest.Utils
  alias Clex.CL10

  @hello_world ~S"""
  __kernel void HelloWorld(__global char* data) {
    data[0] = 'H';
    data[1] = 'E';
    data[2] = 'L';
    data[3] = 'L';
    data[4] = 'O';
    data[5] = ' ';
    data[6] = 'W';
    data[7] = 'O';
    data[8] = 'R';
    data[9] = 'L';
    data[10] = 'D';
    data[11] = '!';
    data[12] = '\n';
  }
  """

  def run(device_type) do
    {:ok, [platform | _]} = CL10.get_platform_ids()
    {:ok, devices} = CL10.get_device_ids(platform, device_type)
    {:ok, context} = CL10.create_context(devices)
    {:ok, queue} = CL10.create_queue(context, hd(devices))
    {:ok, program} = CL10.create_program_with_source(context, @hello_world)
    CL10.build_program(program, devices)
    {:ok, kernel} = CL10.create_kernel(program, "HelloWorld")
    {:ok, buffer} = CL10.create_buffer(context, [:read_write], 32)
    CL10.set_kernel_arg(kernel, 0, buffer)

    {:ok, k_event} = CL10.enqueue_nd_range_kernel(queue, kernel, [1], [1], [])
    {:ok, r_event} = CL10.enqueue_read_buffer(queue, buffer, 0, 32, [k_event])
    [{:ok, output}] = CL10.wait_for_events([r_event])
    output
  end

end
