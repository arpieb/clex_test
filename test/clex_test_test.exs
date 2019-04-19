defmodule ClexTestTest do
  use ExUnit.Case
  doctest ClexTest

  test "greets the world" do
    assert ClexTest.hello() == :world
  end
end
