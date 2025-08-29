import os
import struct
import random
import numpy as np
import tkinter as tk
from tkinter import messagebox
from neuralscratchpy import NeuralNetwork, ActivationFunctionType


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def load_idx_images_np(filename: str, normalize: bool = True, flatten: bool = True) -> np.ndarray:
	with open(filename, 'rb') as f:
		magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
		if magic != 2051:
			raise ValueError(f"{filename} is not a valid MNIST image file (magic={magic})")
		data = np.frombuffer(f.read(), dtype=np.uint8).astype(np.float64)
		if normalize:
			data /= 255.0
		if flatten:
			data = data.reshape(num_images, rows * cols)
		else:
			data = data.reshape(num_images, rows, cols)
		return data


def load_idx_labels_np(filename: str) -> np.ndarray:
	with open(filename, 'rb') as f:
		magic, num_labels = struct.unpack('>II', f.read(8))
		if magic != 2049:
			raise ValueError(f"{filename} is not a valid MNIST label file (magic={magic})")
		labels = np.frombuffer(f.read(), dtype=np.uint8)
		return labels


def one_hot_encode(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
	out = np.zeros((labels.size, num_classes), dtype=np.float64)
	out[np.arange(labels.size), labels] = 1.0
	return out


def load_mnist_small(sample_per_class: int = 200):
	"""Load a small balanced subset for quick demo training."""
	train_images = load_idx_images_np(os.path.join(DATA_DIR, 'train-images.idx3-ubyte'))
	train_labels = load_idx_labels_np(os.path.join(DATA_DIR, 'train-labels.idx1-ubyte'))
	train_oh = one_hot_encode(train_labels, 10)

	# Collect indices per class
	indices = []
	for c in range(10):
		cls_idx = np.where(train_labels == c)[0]
		chosen = np.random.choice(cls_idx, size=min(sample_per_class, len(cls_idx)), replace=False)
		indices.extend(chosen.tolist())
	random.shuffle(indices)
	imgs = train_images[indices]
	lbls = train_oh[indices]
	return imgs, lbls

def load_all_mnist():
	"""Load the entire MNIST dataset."""
	train_images = load_idx_images_np(os.path.join(DATA_DIR, 'train-images.idx3-ubyte'))
	train_labels = load_idx_labels_np(os.path.join(DATA_DIR, 'train-labels.idx1-ubyte'))
	train_oh = one_hot_encode(train_labels, 10)

	test_images = load_idx_images_np(os.path.join(DATA_DIR, 't10k-images.idx3-ubyte'))
	test_labels = load_idx_labels_np(os.path.join(DATA_DIR, 't10k-labels.idx1-ubyte'))
	test_oh = one_hot_encode(test_labels, 10)

	return (train_images, train_oh), (test_images, test_oh)


class DigitApp:
    def __init__(self, master: tk.Tk, net: NeuralNetwork):
        self.master = master
        self.net = net
        self.master.title('MNIST Handwritten Digit - MatrixNetwork Demo')

        # Grid settings
        self.grid_size = 28  # MNIST grid
        self.cell_size = 28  # display pixels per cell
        self.canvas_size = self.grid_size * self.cell_size

        # Brush settings
        self.brush_radius = 1  # in grid units
        self.brush_strength = 0.5  # 0..1 intensity per stroke

        # Pixel array (float for grayscale)
        self.pixels = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)

        # Tkinter canvas
        self.canvas = tk.Canvas(master, width=self.canvas_size, height=self.canvas_size, bg='white')
        self.canvas.grid(row=0, column=0, columnspan=4, padx=10, pady=10)
        self.canvas.bind('<B1-Motion>', self.on_draw)
        self.canvas.bind('<Button-1>', self.on_draw)

        # Buttons
        tk.Button(master, text='Predict', command=self.predict).grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        tk.Button(master, text='Clear', command=self.clear).grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        tk.Button(master, text='Quit', command=master.quit).grid(row=1, column=2, sticky='ew', padx=5, pady=5)

        # Prediction label
        self.pred_var = tk.StringVar(value='Draw a digit')
        tk.Label(master, textvariable=self.pred_var, font=('Arial', 16, 'bold')).grid(row=1, column=3, padx=5, pady=5)

    def on_draw(self, event):
        x, y = event.x, event.y
        gx = x / self.cell_size
        gy = y / self.cell_size

        # Draw a soft brush in a circular radius
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dx = gx - j
                dy = gy - i
                dist = np.sqrt(dx * dx + dy * dy)
                if dist <= self.brush_radius:
                    # increase pixel value smoothly based on distance
                    intensity = self.brush_strength * (1 - dist / (self.brush_radius + 1))
                    self.pixels[i, j] = min(1.0, self.pixels[i, j] + intensity)
                    self.fill_cell(j, i)

    def fill_cell(self, gx: int, gy: int):
        """Draw a square filled with grayscale based on pixel value"""
        val = self.pixels[gy, gx]
        shade = int((1 - val) * 255)
        color = f'#{shade:02x}{shade:02x}{shade:02x}'
        x0 = gx * self.cell_size
        y0 = gy * self.cell_size
        x1 = (gx + 1) * self.cell_size
        y1 = (gy + 1) * self.cell_size
        self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline='')

    def clear(self):
        self.canvas.delete('all')
        self.pixels.fill(0.0)
        self.pred_var.set('Draw a digit')

    def predict(self):
        # flatten and normalize
        flat = self.pixels.reshape(-1).tolist()
        try:
            output = self.net.predict(flat)
        except Exception as e:
            messagebox.showerror('Error', f'Prediction failed: {e}')
            return
        pred_class = int(np.argmax(output))
        confidence = np.max(output) * 100
        print(f"Predicted with {confidence:.2f}% confidence")
        self.pred_var.set(f'Predicted: {pred_class}')



def train_network():
	print('[INFO] Loading MNIST subset...')
	# images, labels = load_mnist_small(sample_per_class=1000)  # 150*10 = 1500 samples
	(images, labels), (test_images, test_labels) = load_all_mnist()
	print('[INFO] Subset loaded:', images.shape, labels.shape)
	net = NeuralNetwork([784, 64, 128, 64, 10], ActivationFunctionType.SIGMOID, ActivationFunctionType.SIGMOID)
	print('[INFO] Training network (epochs=10)...')
	net.train(images.tolist(), labels.tolist(), epochs=10, learning_rate=0.1, verbose=True)
	loss, error = net.evaluate(test_images.tolist(), test_labels.tolist())
	print('[INFO] Evaluation complete:', loss, error)
	print('[INFO] Training complete.')
	return net


def main():
	# Basic existence check for data files
	required = [
		'train-images.idx3-ubyte',
		'train-labels.idx1-ubyte',
	]
	for fname in required:
		if not os.path.exists(os.path.join(DATA_DIR, fname)):
			print(f'[ERROR] Missing data file: {fname} in {DATA_DIR}')
			return
	net = train_network()
	root = tk.Tk()
	app = DigitApp(root, net)
	root.mainloop()


if __name__ == '__main__':
	main()

