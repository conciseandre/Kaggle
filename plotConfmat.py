
# coding: utf-8

# In[ ]:

def plot_confmat(confmat):

    import matplotlib.pyplot as plt
    get_ipython().magic('matplotlib inline')

    fig, ax = plt.subplots(figsize = [2.5, 2.5])
    ax.matshow(confmat, cmap = plt.cm.Reds, alpha = 0.3)
    
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i,
                    s= confmat[i, j],
                    va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.show()

