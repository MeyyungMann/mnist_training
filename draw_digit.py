import tkinter as tk
from PIL import Image, ImageDraw
import os

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Draw a digit")
        self.canvas = tk.Canvas(self, width=280, height=280, bg='black')
        self.canvas.pack()
        self.image = Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.button = tk.Button(self, text="Save", command=self.save)
        self.button.pack()
    def paint(self, event):
        x1, y1 = (event.x - 6), (event.y - 6)
        x2, y2 = (event.x + 6), (event.y + 6)
        self.canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')
        self.draw.ellipse([x1, y1, x2, y2], fill=255)
    def save(self):
        # Create the folder if it doesn't exist
        os.makedirs('drawn_digits', exist_ok=True)
        img = self.image.resize((28, 28)).convert('L')
        img.save(os.path.join('drawn_digits', 'drawn_digit.png'))
        print("Digit saved as drawn_digits/drawn_digit.png")
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()