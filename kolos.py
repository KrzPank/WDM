import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy import fftpack

# Wczytanie obrazu
image = imageio.imread('obraz.png')  # zakładając, że obraz jest monochromatyczny

# a. Histogram składowej luminancji (Y)
# Dla obrazu monochromatycznego luminancja Y to po prostu wartość szarości
plt.figure(figsize=(12, 4))

# Histogram składowej luminancji (Y)
plt.subplot(1, 3, 1)
plt.hist(image.ravel(), bins=256, range=(0, 256), color='black', alpha=0.7)
plt.title("Histogram składowej luminancji (Y)")
plt.xlabel("Intensywność")
plt.ylabel("Liczba pikseli")

# b. Histogram po normalizacji
normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalizacja
plt.subplot(1, 3, 2)
plt.hist(normalized_image.ravel(), bins=256, range=(0, 1), color='black', alpha=0.7)
plt.title("Histogram po normalizacji")
plt.xlabel("Znormalizowana intensywność")
plt.ylabel("Liczba pikseli")

# c. Histogram po wyrównaniu
# Obliczenie wyrównania histogramu
cumulative_hist = np.cumsum(np.histogram(image, bins=256, range=(0, 256))[0])  # Funkcja dystrybuanty
equalized_image = np.interp(image.ravel(), np.arange(256), (cumulative_hist - cumulative_hist.min()) / cumulative_hist.max() * 255)
equalized_image = equalized_image.reshape(image.shape)

plt.subplot(1, 3, 3)
plt.hist(equalized_image.ravel(), bins=256, range=(0, 256), color='black', alpha=0.7)
plt.title("Histogram po wyrównaniu")
plt.xlabel("Intensywność")
plt.ylabel("Liczba pikseli")

plt.tight_layout()
# Zapisanie histogramów do pliku
plt.savefig('histogramy.png')
plt.close()

# 2. Widmo amplitudowe obrazu
# Przekształcenie Fouriera (FFT) obrazu
f_image = fftpack.fft2(image)
f_image_shifted = fftpack.fftshift(f_image)  # Przesunięcie do centrum
amplitude_spectrum = np.abs(f_image_shifted)  # Amplituda widma

# Rysowanie widma amplitudowego
plt.figure(figsize=(6, 6))
plt.imshow(np.log(amplitude_spectrum + 1), cmap='gray')
plt.title("Przybliżone widmo amplitudowe")
plt.colorbar()

# Zapisanie widma amplitudowego do pliku
plt.savefig('widmo_amplitudowe.png')
plt.close()

# 3. Histogram składowych Y, Cb, Cr

# Funkcja konwersji RGB do YCbCr
def rgb_to_ycbcr(image):
    # Jeśli obraz jest w 3 kanałach (RGB)
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    
    # Obliczanie składowych Y, Cb, Cr
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128
    
    return Y, Cb, Cr

# Konwersja obrazu na YCbCr
Y, Cb, Cr = rgb_to_ycbcr(image)

# Rysowanie histogramów składowych Y, Cb, Cr
plt.figure(figsize=(12, 4))

# Y
plt.subplot(1, 3, 1)
plt.hist(Y.ravel(), bins=256, range=(0, 256), color='yellow', alpha=0.7)
plt.title("Histogram składowej Y")
plt.xlabel("Intensywność")
plt.ylabel("Liczba pikseli")

# Cb
plt.subplot(1, 3, 2)
plt.hist(Cb.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
plt.title("Histogram składowej Cb")
plt.xlabel("Intensywność")
plt.ylabel("Liczba pikseli")

# Cr
plt.subplot(1, 3, 3)
plt.hist(Cr.ravel(), bins=256, range=(0, 256), color='red', alpha=0.7)
plt.title("Histogram składowej Cr")
plt.xlabel("Intensywność")
plt.ylabel("Liczba pikseli")

plt.tight_layout()
# Zapisanie histogramów składowych Y, Cb, Cr do pliku
plt.savefig('histogramy_Y_Cb_Cr.png')
plt.close()
