#!/usr/bin/env python3 -I
"""gguard.py TAG -- cmd...   Runs cmd under a thermal watchdog on the P104. Trip: >=TRIP_C (guard.env, 88) or hw_thermal/sw_thermal/hw_slowdown.
Waits for temp<=COOL_C (guard.env, 50; env COOL_C overrides) before start. Writes runs/TAG.trace.csv, runs/TAG.out, runs/TAG.meta.json. Exit 2 on trip."""
import sys,os,subprocess,time,json,signal
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
BUS = os.environ.get("GGUARD_GPU") or (_cvd if _cvd.startswith(("GPU-", "0000")) else "<gpu-uuid>")
def _genv():
    c={"TRIP_C":88,"WARN_C":83,"COOL_C":50,"COOL_TIMEOUT_S":900}
    try:
        for l in open(os.path.expanduser("~/p104-bench/guard.env")):
            k,_,v=l.strip().partition("=")
            if k in c and v.strip().isdigit(): c[k]=int(v.strip())
    except OSError: pass
    if os.environ.get("COOL_C","").isdigit(): c["COOL_C"]=int(os.environ["COOL_C"])
    return c
G=_genv(); LIM=G["TRIP_C"]; WARN=G["WARN_C"]; COOL=G["COOL_C"]
Q="temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_slowdown,clocks_event_reasons.sw_power_cap"
def samp():
    o=subprocess.run(["nvidia-smi","-i",BUS,"--query-gpu="+Q,"--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip().split(", ")
    return o
tag=sys.argv[1]; cmd=sys.argv[sys.argv.index("--")+1:]
d=os.path.expanduser("~/p104-bench/runs")
t0=time.time()
while True:
    s=samp()
    if int(s[0])<=COOL: break
    if time.time()-t0>G["COOL_TIMEOUT_S"]: print("cooldown timeout",s); sys.exit(3)
    time.sleep(5)
start=s; warned=False
out=open(f"{d}/{tag}.out","w"); tr=open(f"{d}/{tag}.trace.csv","w"); tr.write("t,"+Q+"\n")
p=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
tstart=time.time(); trip=None; mx=0
while p.poll() is None:
    s=samp(); tr.write(f"{time.time()-tstart:.1f},"+",".join(s)+"\n"); tr.flush()
    try:
        mx=max(mx,int(s[0]))
        if mx>=WARN and not warned: warned=True; print(f"WARN temp {mx}C >= {WARN}C",flush=True); out.write(f"[gguard WARN t={time.time()-tstart:.1f}s temp {mx}C >= {WARN}C]\n"); out.flush()
        if int(s[0])>=LIM or "Active" in s[6:9]: trip=s
    except Exception: pass
    if trip:
        os.killpg(p.pid,signal.SIGKILL); break
    time.sleep(0.5)
end=samp()
json.dump({"tag":tag,"cmd":cmd,"start":start,"end":end,"max_temp":mx,"warn":warned,"trip_c":LIM,"warn_c":WARN,"cool_c":COOL,"start_temp":int(start[0]),"trip":trip,"rc":p.poll(),"secs":time.time()-tstart,"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())},open(f"{d}/{tag}.meta.json","w"))
if trip: print("TRIP",trip); sys.exit(2)
sys.exit(p.returncode or 0)
