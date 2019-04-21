defmodule ClexTest.Utils do
  @moduledoc false

  ########################################
  # Float conversions
  ########################################

  @doc ~S"""
  Convert a list of numeric values to a bitstring of IEEE 754 binary32 or binary64 values
  """
  @spec list_to_float_bitstring(l::[number], :float32 | :float64) :: bitstring
  def list_to_float_bitstring(l, :float32), do: _list_to_float_bitstring(l, 32)
  def list_to_float_bitstring(l, :float64), do: _list_to_float_bitstring(l, 64)

  defp _list_to_float_bitstring(l, num_bits) when is_list(l) and is_integer(num_bits) do
    for x <- l, into: <<>>, do: _list_to_float_bitstring(x, num_bits)
  end

  defp _list_to_float_bitstring(x, num_bits) when is_number(x) and is_integer(num_bits) do
    <<x::native-float-size(num_bits)>>
  end

  @doc ~S"""
  Convert a bitstring of IEEE 754 binary32 or binary64 values into a list of Elixir floats
  """
  @spec float_bitstring_to_list(data::bitstring, :float32 | :float64) :: [float]
  def float_bitstring_to_list(data, :float32), do: _float_bitstring_to_list(data, 32)
  def float_bitstring_to_list(data, :float64), do: _float_bitstring_to_list(data, 64)

  defp _float_bitstring_to_list(<<>>, _num_bits) do
    []
  end

  defp _float_bitstring_to_list(data, num_bits) when is_bitstring(data) and is_integer(num_bits) do
    <<x::native-float-size(num_bits), rest::binary>> = data
    [x | _float_bitstring_to_list(rest, num_bits)]
  end

  ########################################
  # List-to-matrix conversions
  ########################################

  @doc ~S"""
  Convert the provided flat list into a nexted list in the shape of rows x cols.
  """
  @spec list_to_matrix(l::[], shape::tuple, val::number) :: [[]]
  def list_to_matrix(l, {rows, cols}, val) do
    num_needed = (rows * cols) - length(l)
    pad_list(l, num_needed, val)
    |> Enum.chunk_every(cols)
    |> Enum.take(rows)
  end

  # List is short, pad with provided value
  defp pad_list(l, num_needed, val) when num_needed > 0 do
    l ++ List.duplicate(val, num_needed)
  end

  # List is plenty big, just return it
  defp pad_list(l, num_needed, _val) when num_needed <= 0 do
    l
  end


end
