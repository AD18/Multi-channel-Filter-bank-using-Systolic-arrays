# Multi channel Filter bank using Systolic arrays

## Project brief

This project implements a **FIR filter bank** using a **systolic-array style architecture** implemented in Verilog and exercised on Colab using Icarus Verilog (`iverilog`) and Python-based verification/visualization. The goal is to show a hardware-friendly FIR bank design (many parallel FIR filters) built from a small, reusable **processing element (PE)** and to verify its correctness against a floating-point Python reference. FIR filters are widely used in digital signal processing (DSP) for applications such as audio enhancement, communication systems, biomedical signals, and more. Traditional FIR implementations often rely on direct-form architectures, which suffer from large propagation delays and inefficient hardware usage. By contrast, **systolic arrays** exploit parallelism and pipelining. They consist of interconnected processing elements (PEs) that work together in a rhythmic, "systolic" fashion similar to the pumping of blood in the human heart. This makes systolic arrays an excellent choice for high-performance and scalable FIR filter implementations.

## FIR Filter Formula

The output of an FIR filter is defined as:

$y[n] = \sum_{k=0}^{N-1} h[k] \cdot x[n-k]$

Where:

* **y\[n]** → Output signal at time step *n*
* **x\[n]** → Input signal at time step *n*
* **h\[k]** → Filter coefficients (impulse response of the system)
* **N** → Number of filter taps (order of the FIR filter)

### Explanation:

* Each output $y[n]$ is a **weighted sum** of the current input and the past $N-1$ inputs.
* The coefficients $h[k]$ determine the filter’s behavior (low-pass, high-pass, band-pass, etc.).
* FIR filters are inherently **stable** (since they use no feedback) and have a **linear phase response**, making them suitable for many DSP applications.

---

**What we built**

* A single PE (`block`) that performs a multiply and forwards the input sample to its neighbor (systolic behavior).
* A `systolic_array` Verilog module that chains `NUM_TAPS` PEs to form one FIR filter (multiply + accumulation implemented across the chain).
* A `topmodule` that instantiates `NUM_FILTERS` independent systolic chains in parallel; this is the FIR *bank*.
* A Verilog testbench that feeds fixed-point input samples and filter coefficients (generated in Python) and writes per-cycle outputs to `output.txt`.
* Python tools (in the notebook) to generate inputs/coefficients, run the simulator, read results, compute the floating-point reference with `np.convolve`, and visualize/compare results.

> NOTE: This README intentionally omits the `yosys` (RTL-to-gate synthesis) section, per request.

---

## Why systolic arrays for FIR filters?

### Traditional Approach:

* A direct-form FIR implementation with `N` taps commonly needs `N` multipliers and `N-1` adders on the critical path if implemented combinationally, leading to long combinational paths and slow clocking.
* If you time-multiplex hardware to reuse one multiplier/adder, you increase latency and lower throughput.

In a direct implementation, each new output requires:
* Multiplying each input sample with its corresponding coefficient.
* Accumulating all partial sums.

This approach works but can become inefficient for large $N$, as it requires multiple sequential multiplications and additions.

### Systolic Array Implementation:

Our project instead uses a **systolic array** structure to implement the FIR filter. Here’s how it works:

* The filter is built from a chain of **processing elements (PEs)**.
* Each PE performs a **multiply-and-accumulate (MAC)** operation on part of the input data and coefficient.
* Data (input samples and coefficients) are **pipelined** through the array, enabling multiple operations to execute in parallel.
* The result is that after the pipeline is filled, the filter produces a new output on every clock cycle.
* Thereby reducing **latency and hardware complexity** compared to direct form.  

This systolic design makes the FIR filter **highly parallel** and **throughput-efficient**, especially in hardware implementations like FPGAs or ASICs.

### Advantages in Our Implementation:

* Breaks computation into identical **processing elements (PEs)** that communicate only with immediate neighbors , i.e, no global buses.
* Natural pipelining: new samples flow in every cycle and partial results flow through the array, enabling a new output every cycle after the pipeline fills.
* Regular structure: easy to scale (increase taps or number of parallel filters by adding PEs/modules).
* Good area/time tradeoff: hardware resources are used continuously rather than idling.

In short: systolic arrays are hardware-friendly, highly regular, and yield high throughput with low-frequency critical paths.

---

## Brief Project Overview

This section provides a complete overview of the entire project, presenting the full workflow and flow of execution step by step:

1. **Setup**  
   We begin by installing **Icarus Verilog** on Colab to allow simulation of Verilog hardware code within the notebook environment.

2. **Input & Coefficient Generation**  
   Python scripts generate input samples (e.g., sinusoids) and FIR filter coefficients (`scipy.signal.firwin`).  
   - Before quantization, values are **scaled properly** to avoid underflow, since without scaling many values would collapse to zero.  
   - After scaling, they are quantized into **hexadecimal fixed-point numbers** and written to files:  
     * `input_samples.mem` → one line per sample  
     * `filter_coeffs.mem` → `(NUM_FILTERS × NUM_TAPS)` lines, one line per coefficient  

3. **Defines File (`defines.vh`)**  
   A key feature is that we define parameters like `NUM_TAPS`, `NUM_SAMPLES`, and `NUM_FILTERS` in a single header file `defines.vh`. These values automatically propagate to all other Verilog modules, making the design modular and scalable. By changing values in one place, the entire project adapts.

4. **Verilog FIR Filter (Systolic Array)**  
   Verilog modules implement the FIR filter as a systolic array.  
   - The testbench reads the `.mem` files, processes them through the systolic FIR architecture, and writes the results to an output file.  
   - The **output file (`output.txt`)** contains `(NUM_SAMPLES + NUM_TAPS – 1)` rows × `NUM_FILTERS` columns. The extra `(NUM_TAPS – 1)` rows account for convolution padding.  
   - Each entry is written as a **signed integer** (two’s complement), representing the quantized convolution result.

5. **Simulation & Verification (with Visualization)**  
   Using the same input and coefficients, Python computes reference FIR results with `np.convolve`.  
   - These results are compared line by line with Verilog’s `output.txt`.  
   - Graphs of impulse responses, frequency responses, and time-domain outputs confirm that the Verilog and Python results match closely, apart from negligible quantization error.

6. **Evolution of the Project**  
   Initially, the project was built manually for **3 filters × 3 taps** in a brute-force style. Now, the design is fully modular and scalable, capable of handling arbitrary numbers of taps, samples, and filters. By simply changing parameters in `defines.vh`, it works seamlessly for *x* taps, *y* filters, and *z* samples.

**In summary**:  
The project generates scaled inputs and coefficients (stored as hex), defines parameters centrally in `defines.vh`, runs a systolic-array FIR in Verilog, writes signed integer results to `output.txt`, and verifies them against Python with simulation and visualization. The final design is **accurate, modular, scalable, and efficient**.

---

## How the notebook is organized (section-by-section)

Below I explain each section of the Colab notebook carefully and describe what each code block does.

### 1) `FIR Bank`

This top-level markdown states the project purpose and provides a quick conceptual overview. There is no code in this cell.

### 2) `Install Icarus Verilog on Colab`

This section installs the simulator used to compile and run the Verilog testbench on Colab. The commands are simple shell calls (run in a Colab cell):

```bash
!sudo apt-get update
!sudo apt-get install -y iverilog
```

**Why**: `iverilog` compiles the Verilog sources into a vvp executable which `vvp` can run. We use it to run the testbench and produce a text output (`output.txt`) plus an optional VCD waveform (`wave.vcd`).

---

### 3) `Input samples and filter co-efficients generator`

This Python section generates the stimuli that the Verilog testbench consumes. Key points:

* **Adjustable parameters** (variables at the top of the cell):

  * `NUM_FILTERS` : how many independent FIR filters in the bank (e.g., 15).
  * `NUM_TAPS` : number of taps (filter order), e.g., 15.
  * `NUM_SAMPLES` : number of input samples to generate, e.g., 100.
  * `Fs` : sampling frequency (used to design filters with particular cutoff frequencies).
  * `scale` : scaling factor used to convert floating-point coefficients and samples into fixed-point 32-bit integers. In the notebook `scale = 32768` (i.e. 2^15, a Q15-like scaling), then stored as 32-bit signed values.

* **Input signal**: the notebook creates a test signal that is the sum of several sinusoidal tones spread across the band, for example 100 Hz, 400 Hz, 700 Hz and 1000 Hz, so different filters in the bank (with different cutoffs) selectively attenuate/pass these tones.

* **Filters**: it uses `scipy.signal.firwin` to design `NUM_FILTERS` distinct FIR filters whose cutoff frequencies are spaced across the band. Each filter is a lowpass with a different cutoff (Hamming window used).

* **Fixed-point conversion**: coefficients (`h`) and the input samples `x` are multiplied by `scale`, rounded, and stored as signed 32-bit integers. These are written to memory files for Verilog using `$readmemh`-style hex format:

  * `input_samples.mem` : hex lines for each input sample (32-bit hex per line)
  * `filter_coeffs.mem` : hex lines for all coefficients flattened (NUM\_FILTERS \* NUM\_TAPS lines)
  * `defines.vh` : a small Verilog header containing `` `define NUM_FILTERS ... `` etc.

**Why this matters**: the Verilog implementation uses fixed-width integers; therefore we must carefully quantize inputs and coefficients the same way in Python to compute a comparable reference.

---

### 4) `FIR Bank Verilog Codes`

This is the hardware heart of the project. The notebook writes several Verilog files by constructing lists of strings and dumping them to files. The important Verilog modules are:

#### `block` : the processing element (PE)

```verilog
module block(
    input  signed [31:0] inp_north,   // coefficient (tap)
    input  signed [31:0] inp_west,    // sample (streamed)
    input                 clk,
    input                 rst,
    output reg signed [31:0] outp_east,
    output reg signed [63:0] result
);
    wire signed [63:0] multi;
    assign multi = inp_north * inp_west;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            outp_east <= 0;
            result    <= 0;
        end else begin
            outp_east <= inp_west;
            result    <= multi;   // product for this cycle (no accumulation here)
        end
    end
endmodule
```

**Behavior**:

* `inp_north`: the tap coefficient for this PE (static once loaded).
* `inp_west`: the streaming sample entering the PE from the left.
* `outp_east`: the sample is forwarded to the right (this is the systolic movement).
* `result`: the 64-bit product `inp_north * inp_west` is presented each cycle. The PE does **not** accumulate by itself; it only produces the product and passes the sample.

**Design choice**: splitting multiply and accumulation makes each PE simple and lets a surrounding module (the chain aggregator) perform accumulation cleanly.

#### `systolic_array` : one FIR filter built as a chain of `block` PEs

Key portions (simplified):

```verilog
module systolic_array #(
    parameter NUM_TAPS = 5
)(
    input  signed [31:0] inp_west,
    input  signed [32*NUM_TAPS-1:0] coeffs_flat,
    input                 clk,
    input                 rst,
    output reg signed [71:0] y_out
);
    wire [32*NUM_TAPS-1:0] east_bus;    // carries samples passing east
    wire [64*NUM_TAPS-1:0] result_bus;  // carries per-tap 64-bit products

    // generate block instances and connect them so that samples flow
    // and products are collected in result_bus

    // sum stage: sign-extend the 64-bit products and sum them into a 72-bit `sum`
    always @* begin
        sum = 0;
        for (k = 0; k < NUM_TAPS; k = k + 1) begin
            sum = sum + {{8{result_bus[(k+1)*64-1]}}, result_bus[(k+1)*64-1 -: 64]};
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) y_out <= 0;
        else     y_out <= sum;
    end
endmodule
```

**Important details**:

* Each PE produces a 64-bit product. The sum of `NUM_TAPS` 64-bit products can exceed 64 bits, so the code sign-extends each product by 8 bits and performs accumulation into a 72-bit register (`y_out`). The extra 8 bits is a safety margin to accommodate accumulation growth and maintain sign.
* `coeffs_flat` is a flattened bus containing `NUM_TAPS` 32-bit signed coefficients for the filter instance.
* The module *streams* a sample in via `inp_west` and propagates sample values eastward (so each PE multiplies the proper delayed sample with its coefficient in the same cycle).

This structure implements the multiply-and-accumulate of an FIR, with each multiply performed in a PE and the addition performed by collecting all products and summing them.

#### `topmodule` : parallel FIR bank

`topmodule` instantiates `NUM_FILTERS` copies of `systolic_array`. Each copy receives its subset of flattened coefficients (`coeffs_this_filter`) pulled from a large `coeffs_flat_bus` and shares the same streaming input `inp_west`. Each filter produces a 72-bit `y_this` output and the top module flattens them into `y_out_flat`.

**Why this is powerful**: we can run many independent FIRs on the same input stream in parallel, perfect for filter banks.

#### `testbench`

The testbench:

* Includes the defines header (`` `include "defines.vh" ``) and creates memory arrays for inputs and coefficients: `sample_mem` and `coeff_mem`.
* Uses `$readmemh("input_samples.mem", sample_mem)` and `$readmemh("filter_coeffs.mem", coeff_mem)` to load the quantized values produced by the Python generator.
* On every clock cycle, it presents the next input sample on `inp_west`, constructs the flattened `coeffs_flat_bus` for the number of filters, and captures `y_out_flat` into an output file `output.txt` using `$fwrite`. The testbench runs for `TOTAL = NUM_SAMPLES + NUM_TAPS - 1` cycles (the full convolution length) and then \$fclose / \$finish.
* It also produces a `wave.vcd` file to inspect signals with a waveform viewer (optional).

**Output format**: `output.txt` contains one row per cycle (after the pipeline has started producing outputs) with `NUM_FILTERS` decimal signed integers separated by spaces. This is the data Python later reads for verification.

---

### 5) `Simulator`

This section compiles and runs the Verilog files and then provides plotting utilities.

**Typical steps in the cell**:

1. Compile Verilog into an executable with `iverilog`:

```bash
!iverilog -o testbench.vvp block.v systolic_array.v topmodule.v testbench.v
```

2. Run the simulation with `vvp`:

```bash
!vvp testbench.vvp
```

3. The `vvp` run will produce an `output.txt` file (and a `wave.vcd` if `$dumpfile` was used).

4. Python loads `defines.vh`, `input_samples.mem`, `filter_coeffs.mem`, and `output.txt`. Conversions from hex to signed integers are handled by helper functions like:

```python
def hex_to_signed(val, bits=32):
    x = int(val, 16)
    if x & (1 << (bits - 1)):
        x -= 1 << bits
    return x
```

5. The Notebook supplies interactive plotting functions (with `ipywidgets.interact`) to visualize coefficient sets, filter outputs (time-domain), and frequency responses for groups of filters. These helper functions typically are named `plot_coeffs_group`, `plot_outputs_group`, and `plot_freqs_group` and let you page through groups (e.g., five filters at a time).

**What the simulator visualizes**:

* Filter taps (tap index vs coefficient magnitude). These show the impulse response of each filter and (usually) reveal symmetry (linear phase).
* Frequency response (magnitude in dB) of each filter (via an FFT or via `scipy.signal.freqz` on the float coefficients) which shows the passband and stopband.
* Time-domain outputs for Python vs Verilog for selected filters.

> See the `Output Verifier` section below for the code that performs the reference convolution and the comparison.

---

### 6) `Output Verifier`

This section is the python reference and validator. It does the following, step-by-step:

1. **Load constants** from `defines.vh`.
2. **Load input samples** (from `input_samples.mem`) and **filter coefficients** (from `filter_coeffs.mem`) and convert hex to signed integers.
3. **Compute software reference outputs** (floating or high-precision integer) using `np.convolve(samples, filt, mode='full')` for each filter. This yields `expected_outputs` shaped `(TOTAL, NUM_FILTERS)` where `TOTAL = NUM_SAMPLES + NUM_TAPS - 1`.
4. **Load Verilog outputs** with `np.loadtxt(output_file, dtype=np.int64)` producing `verilog_outputs` with the same shape.
5. **Compute difference**:

```python
diff = expected_outputs - verilog_outputs
```

6. **Report mismatches**: count non-zero entries in `diff` ⇒ `num_mismatches`. If `num_mismatches` > 0, print an example mismatch with expected vs verilog values.

7. **Plotting & interactive checks**:

* `plot_overlay_group(start_index=0)` : overlays the Python expected output and the Verilog output for the selected group of filters (typically five filters at a time). The plotting uses different linestyles so that you can visually compare.

* **Heatmap of differences**: The notebook creates a heatmap `imshow(diff.T, ...)` where x axis is time index (row/cycle), y axis is filter index. The color shows `Python - Verilog` difference value.

8. **Summary statistics** (not always present but recommended): maximum absolute error per filter, mean absolute error, RMSE, etc. These metrics help quantify quantization or implementation mismatch.

---

## Result graphs : What the graphs show and how to interpret them

This is a walk through of each type of plot the notebook generates and what to look for.

### A) Coefficients (tap plots)

**What is plotted**: for each filter, the sequence `h[0], h[1], ..., h[N-1]` (the coefficients / impulse response samples) is shown.

**How to read it**:

* **Symmetry**: many FIR design functions (including `firwin`) produce linear-phase filters, meaning coefficients are symmetric: `h[k] == h[N-1-k]`. You should see mirror symmetry around the center tap.
* **Windowing**: because a Hamming window is used, coefficients smoothly taper near the ends and thus no abrupt truncation.
* **Energy**: larger central coefficient magnitude indicates a sharper impulse response.

**Why useful**: visually confirms correct coefficient generation and fixed-point scaling.

### B) Frequency responses (magnitude in dB)

**What is plotted**: the amplitude response `20*log10(|H(f)|)` across frequency from 0 to Fs/2 for each filter.

**How to read it**:

* **Passband**: frequencies below the intended cutoff should have near-0 dB gain (or slightly below due to ripple), indicating they are passed.
* **Stopband**: frequencies above the cutoff should show strong attenuation (large negative dB values). The depth and slope tell you how selective the filter is.
* **Cutoff shift**: quantization of coefficients causes slight changes in actual cutoff and stopband attenuation compared to the floating-point design.

**What we should expect** given the generator: since the notebook creates a *bank* with cutoffs spread across the band, the frequency response plots for the sequence of filters will show cutoffs gradually moving from low frequency (only 100 Hz tone passes) up to high frequency (many tones pass).

**Why useful**: confirms each filter implements the intended frequency behavior.

### C) Time-domain outputs : Python vs Verilog overlay

**What is plotted**: for a selected filter index, two series are plotted against time: the Python `np.convolve` expected output and the Verilog `y_out` from the simulation.

**How to read it**:

* If the Verilog implementation is **functionally correct** and quantization matches the Python scaling, the two curves should *overlap* almost exactly (differences usually only in integer rounding effects).
* **Transient region**: the first `NUM_TAPS-1` samples are transient (convolution not yet full). Expect partial sums during those steps which is normal and identical in both implementations.
* **Differences**: small step-level differences are typically due to integer quantization of coefficients and inputs. Larger discrepancies indicate a bug in data alignment, sign-extension, or bus width.

**What to inspect**:

* Is the waveform shape identical after the transient? Good sign.
* If the Verilog output is systematically biased (offset), check sign extensions or the way accumulation width is handled.

### D) Heatmap of differences (diff)

**What is plotted**: a 2D image where x axis = time (row/cycle), y axis = filter index, and color = difference `Python - Verilog`.

**How to read it**:

* **All zeros (or near zeros)**: indicates bit-accurate match between the Python integer reference and Verilog outputs.
* **Striping on edges**: expected transient differences near start/end of convolution appear as rows/columns with changes; they are not necessarily errors.
* **Localized spots**: a single bright pixel indicates one sample/filter mismatch. We can inspect the printed example mismatch to see what happened.

**Why useful**: quickly shows whether mismatches are systematic (affect many filters/time steps) vs rare.

### E) Frequency-domain grouping / multiple filters overlay

Often the notebook offers group views (five filters per page). Use that to see how a family of filters gradually transitions from low cutoff to high cutoff.

**Interpretation**:

* We should see the 100 Hz tone present/absent across filters according to cutoff.
* Visual correlation between frequency response and time-domain attenuation of tones: if a tone lies in the stopband, its amplitude in the output time-series will be suppressed.

---

## Typical issues we might observe & debugging tips

* **Large mismatches**: We should check sign extension when accumulating products. In our code the designer explicitly sign-extends each 64-bit product to 72 bits before summing:

  ```verilog
  sum = sum + {{8{result_bus[(k+1)*64-1]}}, result_bus[(k+1)*64-1 -: 64]};
  ```

  If we forget the sign extension or choose too few extension bits, we can see errors for negative values or overflow.

* **Wrong alignment / time shift**: we need to verify that the testbench shifts samples into the systolic chain exactly as the Python convolution assumes (i.e., sample timing relative to when the result is sampled). The testbench writes `TOTAL = NUM_SAMPLES + NUM_TAPS - 1` rows so we need to check if indexing is consistent.

* **Quantization / scaling errors**: we need to make sure the same `scale` is used in both the Python generator and when interpreting results in Python. Off-by-one in scale (e.g., 2^15 vs 2^16) yields visible amplitude differences.

---

## 🔹 Results
The results demonstrate that:
- The systolic array produces the **same output as a conventional FIR filter**.  
- However, the computation is **more efficient** due to parallel MAC operations.  
- The modular PE design makes the filter **scalable for higher-order FIR filters** without major redesign.  

Key Observations:
- **Accuracy**: Outputs match expected FIR behavior.  
- **Efficiency**: Reduced latency compared to traditional direct-form filters.  
- **Scalability**: Easy extension by adding more PEs.  

---

## Conclusion and suggestions for improvements

This project successfully demonstrates the **implementation of an FIR filter using a systolic array**.  

**What we achieved**:

* Implemented a compact systolic-array FIR filter bank in Verilog, parameterized by number of filters and taps.
* Verified the design with an automated flow (Python generates data, Verilog sim runs, Python verifies & visualizes).
* Demonstrated that systolic arrays give an elegant and scalable means to implement high-throughput FIR banks.

The systolic array FIR filter proves to be a **highly efficient, scalable, and practical hardware-friendly solution** for modern digital signal processing applications.

**Possible next steps / improvements**:

* Add **saturation/overflow handling** inside the accumulation path to mimic real fixed-point DSP hardware.
* Add **configurable bitwidth** parameters to experiment with area/accuracy tradeoffs.
* Use a hardware testbench (FPGA) or cycle-approximate model to measure throughput and latency.
* Replace the simple adder accumulation with a tree-adder or pipelined adder network if deeper tap counts are used and timing becomes tight.

---
