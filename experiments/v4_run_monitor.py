"""Run the standalone Task 1 pipeline and sample aggregate process-tree RSS."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/v4'


def rss_tree(pid):
    pending=[pid];seen=set();total=0
    while pending:
        current=pending.pop()
        if current in seen:continue
        seen.add(current)
        try:
            stat=(Path('/proc')/str(current)/'statm').read_text().split()
            total+=int(stat[1])*os.sysconf('SC_PAGE_SIZE')//1024
            children=(Path('/proc')/str(current)/'task'/str(current)/'children').read_text().split()
            pending.extend(map(int,children))
        except (FileNotFoundError,ProcessLookupError):pass
    return total,len(seen)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='2')
    started=time.monotonic();peak=0;nprocess=0
    with (OUT/'training.log').open('w') as stdout,(OUT/'stderr.log').open('w') as stderr:
        p=subprocess.Popen([sys.executable,str(ROOT/'assignment1.py'),'--tasks','1','--rating-workers','3',
                            '--data-dir',str(ROOT/'assignment1_2026'),'--output-dir',str(OUT)],
                           cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
        print('Standalone process PID',p.pid,flush=True)
        while p.poll() is None:
            rss,count=rss_tree(p.pid);peak=max(peak,rss);nprocess=max(nprocess,count);time.sleep(.25)
    result=dict(exit_code=p.returncode,wall_seconds=time.monotonic()-started,
                peak_sampled_aggregate_rss_kib=peak,max_processes=nprocess,sampling_interval_seconds=.25)
    (OUT/'runtime.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result,flush=True)
    raise SystemExit(p.returncode)


if __name__=='__main__':main()
