from matplotlib import pyplot as plt

def view_slice(img, title="", vmax = None, savepath=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.tick_params(axis='both', which='both', top=False,
                    bottom=False, left=False, right=False,
                    labelbottom=False, labelleft=False)

    ax.set_title(title, fontsize=30)
    if vmax == None:
        vmax = img.max()
    ax.imshow(img, cmap="gray", vmax=vmax)
    if savepath:
        plt.savefig(savepath, format='jpeg')
    plt.show()