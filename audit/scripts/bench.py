#!/usr/bin/env python3
"""P104-100 llama-server benchmark with thermal watchdog. Aborts on thermal throttle / temp limit."""
import json, subprocess, threading, time, sys, urllib.request, os, statistics, datetime
URL="http://127.0.0.1:18100"
GPU="<pci-bus-id>"
ABORT_TEMP=88
try:
    for _l in open(os.path.expanduser("~/p104-bench/guard.env")):
        _k,_,_v=_l.strip().partition("=")
        if _k=="TRIP_C" and _v.strip().isdigit(): ABORT_TEMP=int(_v.strip())
except OSError: pass
OUT=os.path.expanduser("~/p104-bench")
stop=threading.Event(); abort_reason=[None]; samples=[]
Q="--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_slowdown,clocks_event_reasons.sw_power_cap"
def sample():
    r=subprocess.run(["nvidia-smi","-i",GPU,Q,"--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip().split(", ")
    t,sm,mem,pw,ut,mu=[float(x) for x in r[:6]]
    fl=[x.strip()=="Active" for x in r[6:]]
    return dict(t=time.time(),temp=t,sm=sm,mem=mem,pw=pw,util=ut,memused=mu,hwth=fl[0],swth=fl[1],hwslow=fl[2],pcap=fl[3])
def watchdog():
    while not stop.is_set():
        s=sample(); samples.append(s)
        if s["hwth"] or s["swth"] or s["hwslow"]:
            abort_reason[0]=f"THERMAL/HW SLOWDOWN flag active at {s['temp']}C (hw_th={s['hwth']} sw_th={s['swth']} hw_slow={s['hwslow']})"
        elif s["temp"]>=ABORT_TEMP:
            abort_reason[0]=f"temp {s['temp']}C >= {ABORT_TEMP}C limit"
        if abort_reason[0]:
            stop.set()
            json.dump(dict(results=results,samples=samples,abort=abort_reason[0]),open(f"{OUT}/result-ABORT-{datetime.datetime.now():%H%M%S}.json","w"))
            print("ABORTED:",abort_reason[0],flush=True); os._exit(2)
        time.sleep(0.5)
def complete(prompt_tokens=None,prompt=None,n_predict=128,slot=-1):
    body=dict(prompt=prompt,n_predict=n_predict,temperature=0.7,cache_prompt=False,stream=False,id_slot=slot,ignore_eos=True)
    req=urllib.request.Request(URL+"/completion",json.dumps(body).encode(),{"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(req,timeout=600))
BASE=("The quick brown fox jumps over the lazy dog while the engineers measure token throughput on an old mining card. ")
def mkprompt(ntok):
    return BASE*max(1,ntok//22)  # ~22 tok per repetition
def run(label,fn):
    if stop.is_set(): return None
    r=fn()
    t=r.get("timings",{})
    row=dict(label=label,pp_n=t.get("prompt_n"),pp_tps=round(t.get("prompt_per_second",0),1),tg_n=t.get("predicted_n"),tg_tps=round(t.get("predicted_per_second",0),2),
             tg_ms_tok=round(t.get("predicted_per_token_ms",0),2),temp=samples[-1]["temp"] if samples else None,sm=samples[-1]["sm"] if samples else None,pw=samples[-1]["pw"] if samples else None)
    print(json.dumps(row),flush=True); results.append(row); return row
results=[]
def main():
    s0=sample(); samples.append(s0)
    print("start:",s0,flush=True)
    threading.Thread(target=watchdog,daemon=True).start()
    run("warmup",lambda:complete(prompt=mkprompt(64),n_predict=16))
    for n in (128,512,1024,2048,4096,8192):
        for rep in range(2): run(f"pp{n}_tg128_r{rep}",lambda:complete(prompt=mkprompt(n),n_predict=128))
    for n in (0,4096,10500):
        run(f"longgen_ctx{n}_tg512",lambda:complete(prompt=mkprompt(max(n,64)),n_predict=512))
    # concurrency: 2 parallel slots
    if not stop.is_set():
        out=[None,None]
        def w(i): out[i]=complete(prompt=mkprompt(1024),n_predict=256,slot=i)
        th=[threading.Thread(target=w,args=(i,)) for i in range(2)]; t0=time.time()
        [x.start() for x in th]; [x.join() for x in th]; dt=time.time()-t0
        tgs=[o["timings"]["predicted_per_second"] for o in out]
        row=dict(label="2x_parallel_pp1024_tg256",per_stream_tg=[round(x,2) for x in tgs],aggregate_tg=round(sum(tgs),2),wall_s=round(dt,1)); print(json.dumps(row)); results.append(row)
    # sustained soak: ~5 min continuous generation to see thermal plateau
    soak=int(sys.argv[1]) if len(sys.argv)>1 else 300
    t0=time.time(); k=0
    while time.time()-t0<soak and not stop.is_set():
        run(f"soak{k}",lambda:complete(prompt=mkprompt(512),n_predict=256)); k+=1
    stop.set()
    time.sleep(0.6)
    json.dump(dict(results=results,samples=samples,abort=abort_reason[0]),open(f"{OUT}/result-{datetime.datetime.now():%H%M%S}.json","w"))
    if abort_reason[0]: print("ABORTED:",abort_reason[0]); sys.exit(2)
    tt=[s["temp"] for s in samples]
    print(f"temp start {tt[0]} max {max(tt)} end {tt[-1]}; max power {max(s['pw'] for s in samples)}W; min SM clock under load {min(s['sm'] for s in samples if s['util']>50)}")
if __name__=="__main__":
    main()
