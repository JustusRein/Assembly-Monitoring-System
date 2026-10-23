#!/usr/bin/env python3
"""Evaluate the deployed YOLO weights on the independent TEST split in data.yaml.

This intentionally refuses to run if the YAML has no explicit 'test:' entry. It will not
silently substitute the training or validation split. Ultralytics also saves PR/confusion-matrix
plots under the generated run directory when supported by the installed version.
"""
from __future__ import annotations
import argparse, csv, json, sys
from datetime import datetime
from pathlib import Path

IMAGE_EXTS={'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}

def yaml_names(names):
    if isinstance(names,dict):
        try: return [str(names[k]) for k in sorted(names,key=lambda x:int(x))]
        except Exception: return [str(v) for v in names.values()]
    if isinstance(names,list): return [str(x) for x in names]
    return []

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent,
                    help='Assembly_monitor project root')
    ap.add_argument('--data',type=Path,default=Path(r'D:\Thesis\Blender\synthetic_dataset\data.yaml'),
                    help='Dataset YAML that must contain an explicit test: split')
    ap.add_argument('--weights',type=Path,default=None,help='Defaults to <project-root>/weights/best.pt')
    ap.add_argument('--imgsz',type=int,default=640)
    ap.add_argument('--device',default='0',help="GPU index '0' or 'cpu'; use '0' on the RTX 4060")
    args=ap.parse_args()
    root=args.project_root.resolve(); data_yaml=args.data.resolve()
    weights=(args.weights or root/'weights'/'best.pt').resolve()
    if not data_yaml.is_file():
        print(f'ERROR: dataset YAML not found: {data_yaml}\nOpen the actual data.yaml under synthetic_dataset and pass its path with --data.',file=sys.stderr); return 2
    if not weights.is_file():
        print(f'ERROR: model weights not found: {weights}',file=sys.stderr); return 2
    try:
        import yaml
    except ImportError:
        print('ERROR: PyYAML is missing from this environment. Install with: python -m pip install pyyaml',file=sys.stderr); return 2
    try:
        cfg=yaml.safe_load(data_yaml.read_text(encoding='utf-8-sig')) or {}
    except Exception as e:
        print(f'ERROR: could not read dataset YAML: {e}',file=sys.stderr); return 2
    test_entry=cfg.get('test')
    if not test_entry:
        print("ERROR: data.yaml has no explicit non-empty 'test:' entry. No test-set evaluation was run. Add/choose the correct held-out test split YAML; do not substitute val or train.",file=sys.stderr); return 2
    base=Path(str(cfg.get('path') or data_yaml.parent))
    if not base.is_absolute(): base=(data_yaml.parent/base).resolve()
    test_paths=test_entry if isinstance(test_entry,list) else [test_entry]
    missing=[]; image_count=0
    for entry in test_paths:
        p=Path(str(entry))
        if not p.is_absolute(): p=(base/p).resolve()
        if not p.exists(): missing.append(str(p)); continue
        if p.is_file() and p.suffix.lower()=='.txt':
            try:
                lines=[x.strip() for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
                image_count += len(lines)
            except Exception: pass
        elif p.is_dir():
            image_count += sum(1 for q in p.rglob('*') if q.is_file() and q.suffix.lower() in IMAGE_EXTS)
        elif p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            image_count += 1
    if missing:
        print('ERROR: the test split in data.yaml points to missing path(s):\n  '+'\n  '.join(missing),file=sys.stderr); return 2
    try:
        from ultralytics import YOLO
    except ImportError:
        print('ERROR: Ultralytics is missing from this environment. Use the same Python environment used to run main.py.',file=sys.stderr); return 2
    outbase=root/'validation_output'; outbase.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S'); out=outbase/f'yolo_test_evaluation_{stamp}'; out.mkdir()
    print('='*64); print('YOLO HELD-OUT TEST-SPLIT EVALUATION'); print('='*64)
    print(f'Weights: {weights}'); print(f'Dataset YAML: {data_yaml}'); print(f'Test image/file count estimated from YAML paths: {image_count}')
    print('Running Ultralytics validation on split="test" ...')
    try:
        model=YOLO(str(weights))
        metrics=model.val(data=str(data_yaml), split='test', imgsz=args.imgsz, device=args.device,
                          plots=True, verbose=True, project=str(out), name='ultralytics_test_run', exist_ok=True)
    except Exception as e:
        print(f'ERROR during YOLO evaluation: {type(e).__name__}: {e}',file=sys.stderr)
        print('No metric is claimed. Check the data.yaml paths and the Ultralytics version that supports these weights.',file=sys.stderr)
        return 2
    box=getattr(metrics,'box',None)
    def number(attr):
        v=getattr(box,attr,None) if box is not None else None
        try: return float(v)
        except (TypeError,ValueError): return None
    vals={
        'precision':number('mp'),
        'recall':number('mr'),
        'mAP50':number('map50'),
        'mAP50_95':number('map'),
    }
    vals={k:v for k,v in vals.items() if v is not None}
    run_dir=Path(getattr(metrics,'save_dir',out/'ultralytics_test_run'))
    all_names=yaml_names(cfg.get('names')) or yaml_names(getattr(metrics,'names',None)) or yaml_names(getattr(model,'names',None))
    maps=getattr(box,'maps',None) if box is not None else None
    per_class=[]
    if maps is not None:
        try: maps=list(maps)
        except TypeError: maps=[]
        for i,v in enumerate(maps):
            try: f=float(v)
            except (TypeError,ValueError): continue
            per_class.append({'class_id':i,'class_name':all_names[i] if i<len(all_names) else f'class_{i}','mAP50_95':f})
    result={
        'evaluation_timestamp':datetime.now().isoformat(timespec='seconds'),
        'evaluation_type':'YOLO object detector on held-out synthetic TEST split',
        'weights':str(weights),'data_yaml':str(data_yaml),'split':'test','image_size':args.imgsz,
        'device':args.device,'estimated_test_images_from_yaml_paths':image_count,
        'aggregate_metrics':vals,'per_class_map50_95':per_class,
        'ultralytics_output_directory':str(run_dir),
        'interpretation_note':'These are object-detector metrics on the declared test split. They do not establish real-camera assembly-stage accuracy or wrong-part/wrong-order detection.'
    }
    (out/'yolo_test_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (out/'yolo_test_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['metric','value','split','weights'])
        for k,v in vals.items(): w.writerow([k,v,'test',str(weights)])
    with (out/'yolo_per_class_map.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['class_id','class_name','mAP50_95']); w.writeheader(); w.writerows(per_class)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        if vals:
            fig,ax=plt.subplots(figsize=(8,4.5)); keys=list(vals); vs=[vals[k] for k in keys]
            bars=ax.bar(keys,vs); ax.bar_label(bars,labels=[f'{v:.4f}' for v in vs],padding=4)
            ax.set_ylim(0,1.08); ax.set_ylabel('Score'); ax.set_title('YOLO held-out test-split metrics')
            ax.grid(axis='y',alpha=.25); fig.text(.5,-.03,'Object detection only; not complete physical-workflow accuracy.',ha='center',fontsize=8)
            fig.tight_layout(); fig.savefig(out/'fig_yolo_test_metrics.png',dpi=240,bbox_inches='tight'); fig.savefig(out/'fig_yolo_test_metrics.pdf',bbox_inches='tight'); plt.close(fig)
        if per_class:
            fig,ax=plt.subplots(figsize=(8,max(4,.42*len(per_class))))
            names=[r['class_name'] for r in per_class]; vs=[r['mAP50_95'] for r in per_class]
            bars=ax.barh(names,vs); ax.bar_label(bars,labels=[f'{v:.3f}' for v in vs],padding=3)
            ax.set_xlim(0,1.08); ax.set_xlabel('mAP@0.50:0.95'); ax.set_title('Per-class YOLO test performance')
            ax.grid(axis='x',alpha=.25); fig.tight_layout(); fig.savefig(out/'fig_yolo_test_per_class.png',dpi=240,bbox_inches='tight'); fig.savefig(out/'fig_yolo_test_per_class.pdf',bbox_inches='tight'); plt.close(fig)
    except ImportError:
        print('NOTE: metrics saved, but matplotlib is missing; plots were not created.')
    print('Aggregate test metrics:')
    for k,v in vals.items(): print(f'  {k}: {v:.4f}')
    print(f'Output folder: {out}')
    print(f'Ultralytics confusion matrix / PR plots (if supported): {run_dir}')
    print('Use the result only as detector performance on the held-out synthetic test set.')
    return 0

if __name__=='__main__': raise SystemExit(main())
