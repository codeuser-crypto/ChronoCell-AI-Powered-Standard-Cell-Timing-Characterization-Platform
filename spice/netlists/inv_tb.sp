* ============================================================================
* INVERTER PROPAGATION-DELAY TESTBENCH (parametric)
* ----------------------------------------------------------------------------
* Standard CMOS inverter: PMOS pull-up + NMOS pull-down driving a cap load.
* Parametric over: VDD, TEMP, Wp, Wn, CL  (substituted by run_sweep.py)
*
* Placeholders {VDD} {TEMP} {WP} {WN} {CL} {CORNER} are filled by the sweep
* driver via Python str.format(); a self-contained default set is also given
* so the file simulates standalone with:  ngspice -b inv_tb.sp
* ============================================================================

.title CMOS Inverter Propagation Delay

* ---- Parameters (overridden by run_sweep.py via -D / template subst) --------
.param VDD  = {VDD}
.param TEMP = {TEMP}
.param WP   = {WP}
.param WN   = {WN}
.param CL   = {CL}
.param LMIN = 150n
.param TR   = 20p
.param PER  = 2n

.temp {TEMP}

* ---- Transistor models (corner-selectable) ---------------------------------
.lib "../models/sky130_pmos_nmos.lib" {CORNER}

* ---- Supplies ---------------------------------------------------------------
Vdd  vdd 0 DC {VDD}
Vin  in  0 PULSE(0 {VDD} 0 {TR} {TR} {PER}/2-{TR} {PER})

* ---- Inverter ---------------------------------------------------------------
* M<name> drain gate source bulk model  W=..  L=..
Mp out in vdd vdd pmos_mod W={WP} L={LMIN}
Mn out in 0   0   nmos_mod W={WN} L={LMIN}

* ---- Load capacitor ---------------------------------------------------------
Cload out 0 {CL}

* ---- Analysis ---------------------------------------------------------------
.tran 1p 4n

* ---- Delay measurements (50% crossings) ------------------------------------
* tpHL: input rising -> output falling  (HL transition of output)
.measure tran tpHL  TRIG v(in)  VAL='VDD/2' RISE=1 TARG v(out) VAL='VDD/2' FALL=1
* tpLH: input falling -> output rising
.measure tran tpLH  TRIG v(in)  VAL='VDD/2' FALL=1 TARG v(out) VAL='VDD/2' RISE=1
* Output slew (10%-90%) on the falling edge
.measure tran tf_out TRIG v(out) VAL='0.9*VDD' FALL=1 TARG v(out) VAL='0.1*VDD' FALL=1
* Average supply current -> dynamic power
.measure tran iavg AVG i(Vdd) FROM=0 TO=4n

.control
run
* Emit machine-readable line for the parser
let pwr = abs(iavg) * VDD
print tpHL tpLH tf_out pwr
.endc

.end
