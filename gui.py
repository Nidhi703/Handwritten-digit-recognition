# gui.py
from tkinter import *
import win32gui
from PIL import ImageGrab, Image
import numpy as np
from recognition import predict_digit
from model import load_model

class App(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.x = self.y = 0

        # Load the pre-trained model
        try:
            self.model = load_model('mnist.h5')
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

        # Creating elements
        self.canvas = Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = Label(self, text="Thinking..", font=("Helvetica", 48))
        self.classify_btn = Button(self, text="Recognise", command=self.classify_handwriting)
        self.button_clear = Button(self, text="Clear", command=self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        # Bind events
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()  # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND)  # get the coordinate of the canvas
        im = ImageGrab.grab(rect)
        digit, acc = predict_digit(self.model, im)
        self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
