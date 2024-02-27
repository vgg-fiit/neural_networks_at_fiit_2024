import numpy as np
import plotly.express as px

def dataset_Circles(m=10, radius=0.7, noise=0.0, verbose=False):

    # Hodnoty X budu v intervale <-1; 1>
    X = (np.random.rand(2, m) * 2.0) - 1.0
    if (verbose): print('X: \n', X, '\n')

    # Element-wise nasobenie nahodnym sumom
    N = (np.random.rand(2, m)-0.5) * noise
    if (verbose): print('N: \n', N, '\n')
    Xnoise = X + N
    if (verbose): print('Xnoise: \n', Xnoise, '\n')

    # Spocitame polomer
    # Element-wise druha mocnina
    XSquare = Xnoise ** 2
    if (verbose): print('XSquare: \n', XSquare, '\n')

    # Spocitame podla prvej osi. Ziskame (1, m) array.
    RSquare = np.sum(XSquare, axis=0, keepdims=True)
    if (verbose): print('RSquare: \n', RSquare, '\n')
    R = np.sqrt(RSquare)
    if (verbose): print('R: \n', R, '\n')

    # Y bude 1, ak je polomer vacsi ako argument radius
    Y = (R > radius).astype(float)
    if (verbose): print('Y: \n', Y, '\n')

    # Vratime X, Y
    return X, Y

def draw_dataset(x, y):
    if x.shape[0] == 2:
        fig = px.scatter(x=x[0], y=x[1], color=y[0], width=700, height=700)
    elif x.shape[1] == 2:
        xx = x.reshape(-1,2).T
        yy = y.reshape(1,-1)
        fig = px.scatter(x=xx[0], y=xx[1], color=yy[0], width=700, height=700)
    else:
        return
    fig.show()


if __name__ == '__main__':
    x,y = dataset_Circles(128)
    print(x.shape)
    print(y.shape)
    draw_dataset(x, y)