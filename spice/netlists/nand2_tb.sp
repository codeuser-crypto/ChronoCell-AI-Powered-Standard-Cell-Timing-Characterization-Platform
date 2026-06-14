* ============================================================================
* NAND2 PROPAGATION-DELAY TESTBENCH (parametric)
* ----------------------------------------------------------------------------
* Topology: two PMOS in PARALLEL (pull-up), two NMOS in SERIES (pull-down).
* Worst-case delay is measured for the A->Z transition while B is held at VDD
* (the stacked-NMOS pull-down path is the timing-critical arc).
*
* Placeholders {VDD} {TEMP} {WP} {WN} {CL} {CORNER} filled by run_sweep.py.
* ============================================================================

.title CMOS NAND2 Worst-Case Propagation Delay

.param VDD  = {VDD}
.param TEMP = {TEMP}
.param WP   = {WP}
.param WN   = {WN}
.param CL   = {CL}
.param LMIN = 150n
.param TR   = 20p
.param PER  = 2n

.temp {TEMP}
.lib "../models/sky130_pmos_nmos.lib" {CORNER}

* ---- Supplies / stimuli -----------------------------------------------------
Vdd vdd 0 DC {VDD}
* B held high so the A arc is exercised; A is the switching input
Vb  b   0 DC {VDD}
Va  a   0 PULSE(0 {VDD} 0 {TR} {TR} {PER}/2-{TR} {PER})

* ---- Pull-up: two PMOS in parallel (A and B gates) -------------------------
Mpa z a vdd vdd pmos_mod W={WP} L={LMIN}
Mpb z b vdd vdd pmos_mod W={WP} L={LMIN}

* ---- Pull-down: two NMOS in series (A on top, B to gnd) --------------------
Mna z  a nint 0 nmos_mod W={WN} L={LMIN}
Mnb nint b 0  0 nmos_mod W={WN} L={LMIN}

* ---- Load -------------------------------------------------------------------
Cload z 0 {CL}

.tran 1p 4n

* A rising -> Z falling (through series NMOS stack = worst case)
.measure tran tpHL  TRIG v(a) VAL='VDD/2' RISE=1 TARG v(z) VAL='VDD/2' FALL=1
* A falling -> Z rising
.measure tran tpLH  TRIG v(a) VAL='VDD/2' FALL=1 TARG v(z) VAL='VDD/2' RISE=1
.measure tran tf_out TRIG v(z) VAL='0.9*VDD' FALL=1 TARG v(z) VAL='0.1*VDD' FALL=1
.measure tran iavg AVG i(Vdd) FROM=0 TO=4n

.control
run
let pwr = abs(iavg) * VDD
print tpHL tpLH tf_out pwr
.endc

.end
