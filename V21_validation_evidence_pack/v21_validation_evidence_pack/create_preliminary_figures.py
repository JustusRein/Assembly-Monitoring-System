"""Create thesis-ready figures from the user's reported V2.1 log summary and thesis Table 5.9."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / 'preliminary_figures'
OUT.mkdir(parents=True, exist_ok=True)

sessions = [
    {'log_file':'process_20261009_160234.jsonl','time':'16:02:34','states':14,'transitions':7,'warnings':1,
     'interpretation':'Earlier pilot; test condition not confirmed from the supplied notes.'},
    {'log_file':'process_20261009_161029.jsonl','time':'16:10:29','states':6,'transitions':2,'warnings':0,
     'interpretation':'Unknown object was not detected; run reached Stage_02 only.'},
    {'log_file':'process_20261009_161525.jsonl','time':'16:15:25','states':12,'transitions':3,'warnings':1,
     'interpretation':'WRONG_ORDER warning reported for nut before Stage_03 was complete.'},
]
with (OUT/'v21_preliminary_log_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w=csv.DictWriter(f, fieldnames=list(sessions[0].keys())); w.writeheader(); w.writerows(sessions)

# Thesis-ready LaTeX table (counts only; interpretation is not an accuracy claim).
with (OUT/'tab_v21_preliminary_log_summary.tex').open('w', encoding='utf-8') as f:
    f.write('% Preliminary exploratory logs; inspect the source CSV before thesis use.\n')
    f.write(r'\begin{tabular}{lrrr}' + '\n')
    f.write(r'\hline' + '\n')
    f.write(r'Session & State records & Transitions & Warnings \\' + '\n')
    f.write(r'\hline' + '\n')
    for item in sessions:
        f.write(f"{item['time']} & {item['states']} & {item['transitions']} & {item['warnings']}" + r' \\' + '\n')
    f.write(r'\hline' + '\n')
    f.write(r'\end{tabular}' + '\n')

# Figure A: counts read from analyser console output. Descriptive counts only.
plt.rcParams.update({'font.size':10, 'axes.titlesize':12, 'axes.labelsize':10})
fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.8), gridspec_kw={'width_ratios':[1.25,1]})
labels = [f"{s['time']}\n{('pilot' if i==0 else 'unknown object' if i==1 else 'wrong order')}" for i,s in enumerate(sessions)]
x=np.arange(len(labels)); width=.24
for off, key, label in [(-width,'states','State records'),(0,'transitions','Stage-transition records'),(width,'warnings','Warning records')]:
    bars=axs[0].bar(x+off,[s[key] for s in sessions],width,label=label)
    axs[0].bar_label(bars,fmt='%d',padding=2,fontsize=8)
axs[0].set_xticks(x,labels)
axs[0].set_ylabel('Recorded count')
axs[0].set_title('(a) Process-log event counts')
axs[0].grid(axis='y',alpha=.25); axs[0].legend(fontsize=8,loc='upper right')
# Outcome summary, only two specific tests described.
outcomes=['Unknown-object\ntest','Out-of-order\nknown part']
counts=[0,1]
bars=axs[1].bar(outcomes,counts)
axs[1].set_ylim(0,1.35); axs[1].set_yticks([0,0.5,1.0])
axs[1].set_ylabel('Warning emitted (0 = no, 1 = yes)')
axs[1].set_title('(b) Observed outcomes in described tests')
axs[1].bar_label(bars,labels=['No warning','WRONG_ORDER'],padding=4,fontsize=9)
axs[1].grid(axis='y',alpha=.25)
fig.suptitle('V2.1 preliminary functional-test evidence (9 October 2026)',fontweight='bold',y=1.02)
fig.text(.5,-.035,'Descriptive exploratory evidence only. Counts are software-log records, not accuracy; session 16:02:34 is not assigned a test condition.',ha='center',fontsize=8)
fig.tight_layout()
fig.savefig(OUT/'fig_v21_preliminary_process_log_counts.png',dpi=260,bbox_inches='tight')
fig.savefig(OUT/'fig_v21_preliminary_process_log_counts.pdf',bbox_inches='tight')
plt.close(fig)

# Figure B: metrics already present in thesis Table 5.9 (synthetic validation split).
metrics=[('Precision',0.9420),('Recall',0.8918),('mAP@0.50',0.9578),('mAP@0.50:0.95',0.7655)]
with (OUT/'yolo_synthetic_validation_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f); w.writerow(['metric','value','source']);
    for n,v in metrics: w.writerow([n,v,'Thesis Table 5.9; synthetic validation split, 560 images'])
fig, ax=plt.subplots(figsize=(8.4,4.5))
labels=[x[0] for x in metrics]; vals=[x[1] for x in metrics]
bars=ax.bar(labels,vals)
ax.bar_label(bars,labels=[f'{v:.4f}' for v in vals],padding=4,fontsize=9)
ax.set_ylim(0,1.08); ax.set_ylabel('Score'); ax.set_title('YOLO26s detector results on the synthetic validation split (n = 560 images)',fontweight='bold')
ax.grid(axis='y',alpha=.25); ax.tick_params(axis='x',labelrotation=12)
fig.text(.5,-.035,'These are detector metrics on synthetic validation images; they are not real-workflow or assembly-stage accuracy.',ha='center',fontsize=8)
fig.tight_layout()
fig.savefig(OUT/'fig_yolo_synthetic_validation_metrics.png',dpi=260,bbox_inches='tight')
fig.savefig(OUT/'fig_yolo_synthetic_validation_metrics.pdf',bbox_inches='tight')
plt.close(fig)

# Table-like thesis-ready narrative.
(OUT/'v21_preliminary_results_text.txt').write_text('''Suggested thesis text (label as preliminary functional testing, not quantitative accuracy):

Three process logs reported by the V2.1 analysis tool contained 14, 6, and 12 state records; 7, 2, and 3 stage-transition records; and 1, 0, and 1 warning records, respectively. The totals were 32 state records, 12 stage-transition records, and 2 warning records. These are descriptive software-log counts and must not be interpreted as stage accuracy or error-detection rate.

In the unknown-object trial, the object outside the seven trained YOLO classes was not detected and no warning was generated. In the known-part out-of-order trial, the system generated a WRONG_ORDER warning when a nut was detected before Stage_03 had been completed. This is a positive observation for that specific warning condition, while the unknown-object result is a missed detection. Independent trial ground truth and error-injection timestamps were not complete for all sessions; aggregate accuracy and response latency are therefore not reported.

Figure captions:
Fig. X: Preliminary V2.1 process-log record counts and the outcomes of two described functional tests. Counts are descriptive only and do not represent accuracy.
Fig. Y: YOLO26s precision, recall, mAP@0.50, and mAP@0.50:0.95 on the synthetic validation split of 560 images. These detector metrics are separate from end-to-end physical workflow validation.
''',encoding='utf-8')
print('Created preliminary figures at', OUT)
for p in sorted(OUT.iterdir()): print(p.name, p.stat().st_size)
