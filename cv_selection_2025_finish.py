"""Continue an already reviewed pilot through full CV, audit, figures and report."""
import argparse,subprocess,sys,time,traceback,shutil,os
from pathlib import Path
from datetime import datetime,timezone
from cv_selection_2025 import read,save,sha,ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--workers',type=int,default=12);a=p.parse_args();out=a.output.resolve()
    assert read(out/'pilot_review.json')['continue_full']
    started=time.time();status=dict(state='running',started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),workers=a.workers)
    logpath=out/'full_workflow.log'
    try:
        with logpath.open('a',encoding='utf-8',buffering=1) as log:
            commands=[('full_cv',['cv_selection_2025.py','--stage','full','--output',str(out),'--workers',str(a.workers)]),('saved_output_audit',['cv_selection_2025_audit_outputs.py',str(out),'--stage','full']),('external_evaluation_and_figures',['cv_selection_2025_report.py',str(out),'--stage','full'])]
            for stage,args in commands:
                status.update(stage=stage,updated_utc=datetime.now(timezone.utc).isoformat());save(out/'RUN_STATUS.json',status)
                subprocess.run([sys.executable,'-B','-X','utf8',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert read(out/'full/saved_output_audit.json')['passed']
        assert read(out/'full/cv_complete.json')['fits']==122528
        assert read(out/'full/evaluation_complete.json')['all_reuse_checks_passed']
        for name in ['cv_selection_2025_report.py','cv_selection_2025_audit_outputs.py','cv_selection_2025_finish.py']:shutil.copy2(ROOT/name,out/'code'/name)
        status['postprocessing_hashes']={name:sha(ROOT/name) for name in ['cv_selection_2025_report.py','cv_selection_2025_audit_outputs.py','cv_selection_2025_finish.py']}
        status.update(state='completed',stage='completed',seconds=time.time()-started,completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        status.update(state='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),seconds=time.time()-started)
        save(out/'RUN_STATUS.json',status);raise
    save(out/'RUN_STATUS.json',status)
if __name__=='__main__':main()
