import tkinter as tk
from utils import process_text, make_predict
import pickle

with open('RidgeClassifier.sav', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.sav', 'rb') as f:
    vectorizer = pickle.load(f)

predict = make_predict(model, vectorizer)
def show_entry_fields():
    text = e1.get()
    score = predict(text)

    output_text = f"{score:.2f}"
    output_string.set(output_text)


master = tk.Tk()
master.geometry("400x240")
master.winfo_toplevel().title("Review Score Predictor")

tk.Label(master,
         text="Review")

e1 = tk.Entry(master)
e1.pack()

label = tk.Label(master, text="Prediction", pady=10)
label.pack()

output_string = tk.StringVar()
output = tk.Label(master, textvariable=output_string, pady=10)
output.config(font=("Roboto", 44))
output.pack()
output_string.set('0')


submitBtn = tk.Button(master,
          text='Predict', command=show_entry_fields)
submitBtn.pack()
        
tk.mainloop()
