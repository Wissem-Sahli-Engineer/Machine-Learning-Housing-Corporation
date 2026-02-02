from pathlib import Path
import matplotlib.pyplot as plt

if not Path("images").is_dir():
    Path("images").mkdir(parents=True, exist_ok=True)

def save_fig(fig_id,tight_Layout = True, fig_ext = ".png", res= 300):
    path = Path("images") / f"{fig_id}.{fig_ext}"
    if tight_Layout :
        plt.tight_layout()
    plt.savefig(path,format = fig_ext,dpi=res)

def settings():
    plt.rc('font', size=14)
    plt.rc('axes', labelsize=14, titlesize=14)
    plt.rc('legend', fontsize=14)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)