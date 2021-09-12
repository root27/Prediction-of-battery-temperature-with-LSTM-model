import pandas  as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

"""
style.use("fivethirtyeight")
fig = plt.figure()
fig.figsize = (8,4)

ax1 = fig.add_subplot(1,1,1)


def animate(i):
    df = pd.read_csv("real time temp.csv")
    ys = df.iloc[:,0].values
    ys1 = df.iloc[:,1].values

    if len(ys)>=120:
        ys = df.iloc[-120:,0].values

        ys1 = df.iloc[-120:,1].values

    xs = list(range(1, len(ys)+1))
    ax1.clear()
    ax1.plot(xs, ys)
    ax1.plot(xs,ys1)

    ax1.set_title("Prediction", fontsize=32)
    ax1.legend([ "Actual","Forecast"], loc="lower right")

ani = animation.FuncAnimation(fig, animate, interval=500)

plt.tight_layout()
plt.show()
"""
df = pd.read_csv("real time temp.csv")
ys = df.iloc[:,0].values
ys1 = df.iloc[:,1].values
print(ys[0])
fig = plt.figure()
fig.figsize = (8,4)
xs = list(range(1, len(ys)+1))
ax1 = fig.add_subplot(1,1,1)
plt.yticks(fontsize=5)
ax1.plot(xs,ys)
plt.tight_layout()
plt.show()