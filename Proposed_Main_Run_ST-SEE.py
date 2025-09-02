import tkinter as tk
import tkinter.ttk as ttk
from tkinter import *
from tkinter import messagebox
import sys
import os

root = tk.Tk()
root.title("")
root.geometry('1350x768+0+0')

tk.Label(root, 
		 text="SEQUENTIAL ELLIPTICAL ENVELOPE OUTLIER DETECTION FOR CONTINUOUS MONITORING OF ",
		 fg = "light green",
		 bg = "dark green",
		 font = "Helvetica 16 bold italic").pack(pady=30,padx=0)
		 
tk.Label(root, 
		 text="CARDIOVASCULAR ACTIVITIES USING PPG ",
		 fg = "light green",
		 bg = "dark green",
		 font = "Helvetica 16 bold italic").pack(pady=1,padx=0)
		 
def b1():
    
    import tkinter 
    import tkinter.filedialog
    import getpass
    from tkinter.filedialog import askdirectory
   
    import os

    import ntpath

    os.getcwd()
       
    directory= askdirectory(initialdir=os.getcwd())

    print(directory)

    print(ntpath.basename(directory))
    
    messagebox.showinfo('Message Info', 'Success'+"\n"+directory)
    
    import ntpath

    os.getcwd()
    f = open("Info.txt", "w")
    f.write(ntpath.basename(directory))
    f.close()
    
    os.system('python PPG_Plot.py')

def b2():

    os.system('python Butterworth_Bandpass_filter_signal_preprocessing.py')

def b3():
    
    os.system('python Hijorth_Statistical_TD_Hilbert_Transform_FD_feature_extraction.py')

def b31():    

    os.system('python Sequential_Elliptic_Envelope_Outlier_Detection.py')

def b4():
    
    os.system('python Training_time.py')   
	
def b5():    
    
    os.system('python MAE_outlier_detection.py')

def b6():
    
    os.system('python PRA.py')  


b1=Button(root,text="PPG-DaLiA dataset ",command=b1,bg="black",fg="white",font = "Helvetica 13 bold italic")

b1.place(x=200,y=200)

b1.configure(width=55,height=1)


b2=Button(root,text="Butterworth Bandpass Normalization-based pre-processing",command=b2,bg="black",fg="white",font = "Helvetica 13 bold italic")

b2.place(x=200,y=250)

b2.configure(width=55,height=1)


b3=Button(root,text="Hijorth_Statistical_TD_Hilbert Transform_FD_feature_extraction ",command=b3,bg="black",fg="white",font = "Helvetica 13 bold italic")

b3.place(x=200,y=300)

b3.configure(width=55,height=1)


b31=Button(root,text="Sequential Elliptical Envelope-based Outlier detection",command=b31,bg="black",fg="white",font = "Helvetica 13 bold italic")

b31.place(x=200,y=350)

b31.configure(width=55,height=1)



l2=tk.Label(root,text="Performance",fg = "light green",bg = "dark green",font = "Helvetica 16 bold italic")
l2.place(x=980,y=400)

b4=Button(root,text=" Training time (sec)",command=b4,bg="black",fg="white",font = "Helvetica 13 bold italic")
b4.place(x=800,y=450)
b4.configure(width=45,height=1)

b5=Button(root,text="MAE of outlier detection (%)",command=b5,bg="black",fg="white",font = "Helvetica 13 bold italic")
b5.place(x=800,y=500)
b5.configure(width=45,height=1)

b6=Button(root,text="Precision, recall and accuracy",command=b6,bg="black",fg="white",font = "Helvetica 13 bold italic")
b6.place(x=800,y=550)
b6.configure(width=45,height=1)


root.mainloop()
