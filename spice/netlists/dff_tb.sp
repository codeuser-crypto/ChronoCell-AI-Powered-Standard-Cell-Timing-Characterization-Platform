* ============================================================================
* D FLIP-FLOP TESTBENCH (parametric) -- master/slave CMOS, ~12T
* ----------------------------------------------------------------------------
* Built from transmission-gate latches + inverters (classic master-slave DFF).
* Measures clock-to-Q delay (clk rising -> Q settling). Setup/hold are swept
* externally by run_sweep.py by shifting the D edge relative to the clock.
*
* Clock frequency = 500 MHz (period 2 ns). Placeholders filled by run_sweep.py.
* ============================================================================

.title CMOS Master-Slave D Flip-Flop Clock-to-Q

.param VDD  = {VDD}
.param TEMP = {TEMP}
.param WP   = {WP}
.param WN   = {WN}
.param CL   = {CL}
.param LMIN = 150n
.param TR   = 20p
.param PER  = 2n          ; 500 MHz clock

.temp {TEMP}
.lib "../models/sky130_pmos_nmos.lib" {CORNER}

Vdd  vdd 0 DC {VDD}

* Clock: 500 MHz
Vclk clk 0 PULSE(0 {VDD} 0 {TR} {TR} {PER}/2-{TR} {PER})
* Inverted clock
Xinvclk clk clkb vdd 0 INVX

* Data: rises well before the capturing clock edge (>> setup time)
Vd   d 0 PULSE(0 {VDD} 0.3n {TR} {TR} 5n 10n)

* ---- Master latch -----------------------------------------------------------
* TG passes D when clk=0
XtgM   d   qm  clk clkb vdd 0 TGATE
Xinvm1 qm  qmb vdd 0 INVX
Xinvm2 qmb qm  vdd 0 INVX        ; feedback (weak keeper modelled as inv)

* ---- Slave latch ------------------------------------------------------------
* TG passes master output when clk=1
XtgS   qmb qs  clkb clk vdd 0 TGATE
Xinvs1 qs  qsb vdd 0 INVX
Xinvs2 qsb qs  vdd 0 INVX
Xinvq  qsb q   vdd 0 INVX        ; output buffer -> Q

Cload q 0 {CL}

* ---- Subcircuits ------------------------------------------------------------
.subckt INVX a y vdd vss
Mp y a vdd vdd pmos_mod W={WP} L={LMIN}
Mn y a vss vss nmos_mod W={WN} L={LMIN}
.ends

.subckt TGATE in out g gb vdd
* transmission gate: g controls NMOS, gb controls PMOS
Mn in out g  0   nmos_mod W={WN} L={LMIN}
Mp in out gb vdd pmos_mod W={WP} L={LMIN}
.ends

.tran 1p 8n

* clock-to-Q: capturing clock edge (~2 ns) rising -> Q rising
.measure tran tpHL  TRIG v(clk) VAL='VDD/2' RISE=1 TD=1.5n TARG v(q) VAL='VDD/2' FALL=1
.measure tran tpLH  TRIG v(clk) VAL='VDD/2' RISE=1 TD=1.5n TARG v(q) VAL='VDD/2' RISE=1
.measure tran tf_out TRIG v(q) VAL='0.9*VDD' FALL=1 TARG v(q) VAL='0.1*VDD' FALL=1
.measure tran iavg AVG i(Vdd) FROM=0 TO=8n

.control
run
let pwr = abs(iavg) * VDD
print tpHL tpLH tf_out pwr
.endc

.end
