"""Plot typography and colors for metalloprotease measurements."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

cm = 1 / 2.54


class styling:
    paper_colors = {
        "paper_teal": "#4FB9AF", "paper_navaho": "#FFE0AC",
        "paper_melon": "#FFC6B2", "paper_blue": "#6686C5",
        "paper_pink": "#FFACB7", "paper_darkblue": "#4B5FAA",
        "paper_amaranth": "#D59AB5", "paper_coolgrey": "#9596C6",
        "paper_indigo": "#08415C",
    }

    @staticmethod
    def despline(ax=None, left_visible=True):
        ax = plt.gca() if ax is None else ax
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(left_visible)

    @staticmethod
    def set_spine_width(ax, width=1.0):
        for spine in ax.spines.values():
            spine.set_linewidth(width)


def apply_style():
    sns.set_style("white")
    sns.set_style("ticks")
    sns.set_palette(styling.paper_colors.values())
    mpl.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    mpl.rcParams["axes.linewidth"] = 1.15


def init_fig():
    plt.figure(figsize=(6.6 * cm, 7.5 * cm), dpi=300)
