from pathlib import Path
import json, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'Milestone_2_Overleaf'
OUT = Path(__file__).resolve().parent
rows = list(json.loads((ROOT/'parallel_c/data/scaling_results.json').read_text())['results'].values())

def wrap_image(source, name):
    image = Image.open(source)
    w, h = image.size
    dest = OUT / (name + '.pdf')
    c = canvas.Canvas(str(dest), pagesize=(w*.24,h*.24),pageCompression=1)
    c.drawImage(ImageReader(image),0,0,width=w*.24,height=h*.24,mask='auto')
    c.showPage();c.save()

for name in ['sph_neighbourhood','sequential_flow','parallel_flow','spatial_hash_grid']:
    wrap_image(OUT/(name+'.jpg'),name)
for name in ['breakdown_vs_n_sequential_t1','comparison_16t_nohash_vs_hash','comparison_hash_1t_vs_16t']:
    wrap_image(ROOT/'parallel_c/data'/(name+'.png'),name)
for name in ['dambreak_filmstrip','pressure_vs_depth_hydrostatic','momentum_drift_vs_time']:
    wrap_image(ROOT/'sequential_c/data/plots'/(name+'.png'),name)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
for variant, name in [('prehash','thread_scaling_dense'),('hash','hash_thread_scaling_dense')]:
    data=[r for r in rows if r['variant']==variant]
    base={r['particles_requested']:r['ms_per_step'] for r in data if r['threads']==1}
    fig,axs=plt.subplots(2,1,figsize=(8.2,8.0),layout='constrained')
    for t in sorted({r['threads'] for r in data}):
        pts=sorted([r for r in data if r['threads']==t and r['particles_requested'] in base],key=lambda r:r['n_total'])
        axs[0].plot([r['n_total'] for r in pts],[base[r['particles_requested']]/r['ms_per_step'] for r in pts],'.-',ms=3,lw=1.2,label=f'{t} threads')
    axs[0].set(xscale='log',xlabel='Actual particle count N (log scale)',ylabel='Speedup relative to the same solver at 1 thread',title=f'(a) Full saved sweep: {len(base)} particle counts per thread setting')
    axs[0].legend(ncol=4,fontsize=8)
    for requested in ([1000,10000,30000,100000] if variant=='prehash' else [2000,10000,30000,100000]):
        pts=sorted([r for r in data if r['particles_requested']==requested],key=lambda r:r['threads'])
        axs[1].plot([r['threads'] for r in pts],[base[requested]/r['ms_per_step'] for r in pts],'o-',ms=4,label=f"N = {pts[0]['n_total']:,}")
    axs[1].plot([1,32],[1,32],'--',color='.55',lw=1,label='Ideal linear speedup')
    axs[1].set(xlabel='OpenMP threads',ylabel='Speedup relative to 1 thread',title='(b) Strong scaling at fixed particle counts',xticks=[1,2,4,8,16,24,32])
    axs[1].legend(ncol=3,fontsize=8)
    for ax in axs:ax.grid(alpha=.2)
    fig.savefig(OUT/(name+'.pdf'));fig.savefig(OUT/(name+'.png'),dpi=150);plt.close(fig)

p=REPORT/'Milestone_2.tex'
s=p.read_text(encoding='utf-8')
# Preserve the six-section narrative; remove only the requested closing subsection.
s=s[:s.index('\\subsection{Data and implementation')].rstrip()+'\n\n\\end{document}\n'
s=re.sub(r'Figure (\d+)',lambda m:'Figure~\\ref{fig:'+m.group(1)+'}',s)
s=s.replace('\\usepackage[section]{placeins}','\\usepackage{float}')
s=s.replace('\\begin{figure}', '\\begin{figure}[H]')
s=s.replace('\\clearpage\n\\begin{figure}[H]\n\\centering\n\\includegraphics[width=\\linewidth,height=0.48\\textheight,keepaspectratio]{figures/breakdown_vs_n_prehash_t24.pdf}', '\\begin{figure}[H]\n\\centering\n\\includegraphics[width=\\linewidth,height=0.48\\textheight,keepaspectratio]{figures/breakdown_vs_n_prehash_t24.pdf}')

def figure(name, caption, label, height='.48', width='1'):
    return '\n\\begin{figure}[H]\n\\centering\n'+f'\\includegraphics[width={width}\\linewidth,height={height}\\textheight,keepaspectratio]{{figures/{name}.pdf}}\n'+f'\\caption{{{caption}}}\\label{{fig:{label}}}\n\\end{{figure}}\n'

def before(marker, content):
    global s
    assert marker in s, marker
    s=s.replace(marker,content+'\n'+marker,1)

before('\\section{The SPH computation',figure('breakdown_vs_n_sequential_t1',r'Sequential stage costs across the saved particle-count sweep. The two neighbourhood searches account for almost all of the growth, showing that the bottleneck persists across resolutions rather than being peculiar to the 30,016-particle example. Source: \path{parallel_c/data/breakdown_vs_n_sequential_t1.png}.','seq-breakdown-sweep'))
before('\\subsection{Initialization and',figure('sph_neighbourhood',r'Finite-support particle interactions, reproduced from slide 2 of the presentation. Only particles inside the support circle contribute to the target particle. The source illustration writes the support radius as $kh$; in this implementation the corresponding radius is $R=2H$.','sph-support',height='.38'))
before('\\subsection{Parallelism and',figure('sequential_flow',r'The sequential SPH flow diagram from slide 3. This schematic introduces the density--pressure--force dependency and the recurrence between timesteps. The kick--drift--kick pseudocode above gives the more precise integration and boundary-update order used by the benchmark driver.','sequential-flow',height='.58'))
before('\\subsection{Parallel pressure and force',figure('parallel_flow',r'Particle-level work sharing and stage barriers, reproduced from slide 7. The illustrated particle tasks are distributed over a fixed pool of OpenMP threads; the implementation does not create a separate operating-system thread for every particle. Pressure is shown as one stage here, and its elementwise vector operations are also threaded in the code.','parallel-flow',height='.62'))

start=s.index('\\subsection{Why small particle counts behave')
end=s.index('\\subsection{Breakdown after',start)
s=s[:start]+r'''\subsection{Thread scaling across the full particle-count sweep}\label{why-small-particle-counts-behave-differently}

The fuller saved sweep contains 49 particle counts for each of seven OpenMP thread settings, from 1 to 32 threads. Figure~\ref{fig:5} replaces the earlier five-count speedup plot with all 343 all-pairs measurements from the common benchmark dataset. For every particle count, the speedup is calculated as the one-thread time divided by the time at the chosen thread count. No new solver runs are used.
'''+figure('thread_scaling_dense',r'OpenMP all-pairs thread scaling from the complete saved sweep. Panel (a) shows all 49 particle counts for each thread setting; panel (b) holds the particle count fixed and varies the number of threads. Speedups use mean per-step times in \path{parallel_c/data/scaling_results.json}. Lines join measured values; the dashed reference in panel (b) is ideal linear speedup.','5',height='.72')+r'''
The two views answer complementary questions. The first shows how much useful work is available to amortize thread-management and synchronization costs as the particle count increases. The second shows diminishing returns when more threads are assigned to a fixed problem. Thread count alone cannot guarantee a speedup: the workload size, serial stages, memory behavior, and measurement variation all influence the result. The 9,933-particle values agree with the detailed table in Section 4.2 because both use the same dataset.

'''+s[end:]

before('\\subsection{Mapping a position',figure('spatial_hash_grid',r'Spatial-hashing illustration from slide 9. The green particle is the query target; blue particles pass the radius test; grey particles lie in candidate cells but still require a distance check; red particles lie outside the searched cell neighbourhood. The source diagram labels its interaction radius $h$; this report uses $R=2H$ for that radius and the cell side. The queried region comprises the target cell and its eight adjacent cells, nine cells in total.','spatial-hash-grid',height='.45'))
before('\\subsection{Comparing sequential,',figure('hash_thread_scaling_dense',r'Thread scaling after spatial hashing using all 50 saved particle counts for each of seven thread settings (350 measurements). Panel (a) uses every saved point; panel (b) shows fixed-size strong scaling. The one-thread spatial-hash solver is the baseline, so these ratios isolate threading within the optimized algorithm. Source: \path{parallel_c/data/scaling_results.json}.','hash-thread-scaling',height='.72')+figure('comparison_hash_1t_vs_16t',r'One versus 16 threads with the same spatial-hash search. Keeping the algorithm fixed isolates the benefit of multithreading after the candidate set has already been reduced. This complementary runtime view also makes the measured variation near intermediate particle counts visible. Source: \path{parallel_c/data/comparison_hash_1t_vs_16t.png}.','hash-1t-16t'))
before('\\subsection{Does the breakdown confirm',figure('comparison_16t_nohash_vs_hash',r'All-pairs versus spatial-hash search at the same 16-thread setting. Holding the thread count fixed isolates the algorithmic improvement. The spatial-hash curve stays close to the horizontal axis on this shared linear scale because its runtime is much smaller. Source: \path{parallel_c/data/comparison_16t_nohash_vs_hash.png}.','hash-fixed-threads'))
before('The existing sequential validation report records',figure('dambreak_filmstrip',r'Dam-break particle snapshots from the saved sequential validation run. The initially confined column spreads and interacts with the tank boundaries. This is an illustration of the test scene and its simulated motion; it is not an experimental comparison or a validation of the parallel solver. Source: \path{sequential_c/data/plots/dambreak_filmstrip.png}.','dambreak-motion',height='.32'))
before('The next compute direction',r'''
\begin{figure}[H]
\centering
\includegraphics[width=.49\linewidth]{figures/momentum_drift_vs_time.pdf}\hfill
\includegraphics[width=.49\linewidth]{figures/pressure_vs_depth_hydrostatic.pdf}
\caption{Two existing sequential validation diagnostics. Left: relative momentum drift remains near floating-point roundoff in the free-fluid test. Right: the hydrostatic pressure-depth relation differs substantially from the theoretical line. Together they illustrate why performance improvement and physical validation must be assessed separately. Sources: the corresponding plots in \path{sequential_c/data/plots/}.}\label{fig:validation-diagnostics}
\end{figure}
''')
p.write_bytes(s.encode('utf-8'))
print('Figures:',s.count('\\begin{figure}'))
print('New PDF assets:',[x.name for x in OUT.glob('*.pdf')])

