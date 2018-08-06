import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

app = tk.Tk()
app.wm_title("Graphs")

fig = Figure(figsize=(6, 4), dpi=96)
a = np.array([1,2,3])
ax = fig.add_subplot(111)

line, = ax.plot(a,np.array([0,0.5,2]))
line2, = ax.plot(a,0.55*a)

graph = FigureCanvasTkAgg(fig, master=app)
canvas = graph.get_tk_widget()
canvas.grid(row=0, column=0, rowspan = 11, padx =10, pady =5)

def updateScale(value):
   print("scale is now %s" % (value))
   b = float(value)*a
   # set new data to the line
   line2.set_data(a,b)
   # rescale the axes
   ax.relim()
   ax.autoscale()
   #draw canvas
   fig.canvas.draw_idle()


value = tk.DoubleVar()
scale = tk.Scale(app, variable=value, orient="horizontal",length = 100, 
                 from_=0.55, to=2.75, resolution = 0.01,command=updateScale)
scale.grid(row=0, column=1)

app.mainloop()