defmodule ClexTest.Kernel.MyGEMM2 do
  @moduledoc false

  def name(), do: "myGEMM2"

  def source() do
    ~S"""
    // Tiled and coalesced version
    __kernel void myGEMM2(const int M, const int N, const int K,
                          const __global float* A,
                          const __global float* B,
                          __global float* C) {

        // Thread identifiers
        const int row = get_local_id(0); // Local row ID (max: TS)
        const int col = get_local_id(1); // Local col ID (max: TS)
        const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
        const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)

        // Local memory to fit a tile of TS*TS elements of A and B
        __local float Asub[TS][TS];
        __local float Bsub[TS][TS];

        // Initialise the accumulation register
        float acc = 0.0f;

        // Loop over all tiles
        const int numTiles = K/TS;
        for (int t=0; t<numTiles; t++) {

            // Load one tile of A and B into local memory
            const int tiledRow = TS*t + row;
            const int tiledCol = TS*t + col;
            Asub[col][row] = A[tiledCol*M + globalRow];
            Bsub[col][row] = B[globalCol*K + tiledRow];

            // Synchronise to make sure the tile is loaded
            barrier(CLK_LOCAL_MEM_FENCE);

            // Perform the computation for a single tile
            for (int k=0; k<TS; k++) {
                acc += Asub[k][row] * Bsub[col][k];
            }

            // Synchronise before loading the next tile
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Store the final result in C
        C[globalCol*M + globalRow] = acc;
    }
    """
  end

end
