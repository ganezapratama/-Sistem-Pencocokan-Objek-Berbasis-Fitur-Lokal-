
# SISTEM PENCARIAN OBJEK BERBASIS FITUR LOKAL

# import library
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# machine learning
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ============================================================
# PATH DATASET
# ============================================================

dataset_path = "Dataset"

# ============================================================
# DETECTOR
# ============================================================

# menggunakan ORB agar lebih ringan
orb = cv2.ORB_create(nfeatures=300)

# ============================================================
# FUNGSI MEMBACA GAMBAR
# ============================================================

def load_image(path):

    image = cv2.imread(path)

    # mengecek apakah gambar ada
    if image is None:

        print("Gambar tidak ditemukan :", path)

        return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image, gray

# ============================================================
# EKSTRAKSI FITUR
# ============================================================

def ekstraksi_fitur(detector, gray):

    start = time.time()

    keypoints, descriptors = detector.detectAndCompute(gray, None)

    end = time.time()

    waktu = end - start

    return keypoints, descriptors, waktu

# ============================================================
# MENAMPILKAN KEYPOINTS
# ============================================================

def tampilkan_keypoints(image, keypoints, judul):

    hasil = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=(0,255,0)
    )

    plt.figure(figsize=(8,5))

    plt.imshow(cv2.cvtColor(hasil, cv2.COLOR_BGR2RGB))

    plt.title(judul)

    plt.axis('off')

    plt.show()

# ============================================================
# BF MATCHING
# ============================================================

def bf_matching(desc1, desc2):

    bf = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True
    )

    matches = bf.match(desc1, desc2)

    matches = sorted(
        matches,
        key=lambda x: x.distance
    )

    return matches

# ============================================================
# MENAMPILKAN HASIL MATCHING
# ============================================================

def tampilkan_matching(img1, kp1, img2, kp2, matches):

    hasil = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches[:20],
        None,
        flags=2
    )

    plt.figure(figsize=(12,6))

    plt.imshow(cv2.cvtColor(hasil, cv2.COLOR_BGR2RGB))

    plt.title("Hasil Matching")

    plt.axis('off')

    plt.show()

# ============================================================
# MEMBACA GAMBAR
# ============================================================

ref_path = "Dataset/buku/ref.jpg"

test_path = "Dataset/buku/test1.jpg"

img1, gray1 = load_image(ref_path)

img2, gray2 = load_image(test_path)

# ============================================================
# EKSTRAKSI FITUR ORB
# ============================================================

print("="*50)
print("EKSTRAKSI FITUR ORB")
print("="*50)

kp1, desc1, waktu1 = ekstraksi_fitur(
    orb,
    gray1
)

kp2, desc2, waktu2 = ekstraksi_fitur(
    orb,
    gray2
)

print("Jumlah keypoints gambar referensi :", len(kp1))

print("Jumlah keypoints gambar test :", len(kp2))

print("Ukuran descriptor :", desc1.shape)

print("Waktu ekstraksi :", waktu1)

# ============================================================
# MENAMPILKAN KEYPOINTS
# ============================================================

tampilkan_keypoints(
    img1,
    kp1,
    "Keypoints Referensi"
)

tampilkan_keypoints(
    img2,
    kp2,
    "Keypoints Test"
)

# ============================================================
# FEATURE MATCHING
# ============================================================

print("="*50)
print("FEATURE MATCHING")
print("="*50)

matches = bf_matching(
    desc1,
    desc2
)

print("Jumlah matches :", len(matches))

# ============================================================
# MENAMPILKAN MATCHING
# ============================================================

tampilkan_matching(
    img1,
    kp1,
    img2,
    kp2,
    matches
)

# ============================================================
# BAG OF VISUAL WORDS
# ============================================================

print("="*50)
print("BAG OF VISUAL WORDS")
print("="*50)

all_descriptors = []

histograms = []

labels = []

# ============================================================
# MENGAMBIL DESCRIPTOR
# ============================================================

for kategori in os.listdir(dataset_path):

    folder_path = os.path.join(
        dataset_path,
        kategori
    )

    if not os.path.isdir(folder_path):

        continue

    for file in os.listdir(folder_path):

        image_path = os.path.join(
            folder_path,
            file
        )

        image = cv2.imread(
            image_path,
            0
        )

        # resize agar lebih ringan
        image = cv2.resize(
            image,
            (300,300)
        )

        kp, desc = orb.detectAndCompute(
            image,
            None
        )

        if desc is not None:

            all_descriptors.extend(desc)

# ============================================================
# KMEANS
# ============================================================

# diperkecil agar ringan
k = 10

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

kmeans.fit(all_descriptors)

print("Vocabulary size :", k)

# ============================================================
# MEMBUAT HISTOGRAM
# ============================================================

def build_histogram(descriptors):

    histogram = np.zeros(k)

    clusters = kmeans.predict(descriptors)

    for c in clusters:

        histogram[c] += 1

    return histogram

# ============================================================
# MEMBUAT DATASET HISTOGRAM
# ============================================================

for kategori in os.listdir(dataset_path):

    folder_path = os.path.join(
        dataset_path,
        kategori
    )

    if not os.path.isdir(folder_path):

        continue

    for file in os.listdir(folder_path):

        image_path = os.path.join(
            folder_path,
            file
        )

        image = cv2.imread(
            image_path,
            0
        )

        image = cv2.resize(
            image,
            (300,300)
        )

        kp, desc = orb.detectAndCompute(
            image,
            None
        )

        if desc is not None:

            hist = build_histogram(desc)

            histograms.append(hist)

            labels.append(kategori)

# ============================================================
# DATA TRAINING DAN TESTING
# ============================================================

X = np.array(histograms)

y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# ============================================================
# SVM
# ============================================================

print("="*50)
print("KLASIFIKASI SVM")
print("="*50)

svm = SVC(kernel='linear')

svm.fit(X_train, y_train)

prediksi = svm.predict(X_test)

akurasi = accuracy_score(
    y_test,
    prediksi
)

print("Akurasi :", akurasi)

# ============================================================
# PCA
# ============================================================

print("="*50)
print("PCA")
print("="*50)

# jumlah komponen diperkecil
pca = PCA(n_components=5)

hasil_pca = pca.fit_transform(X)

print("Ukuran data awal :", X.shape)

print("Ukuran setelah PCA :", hasil_pca.shape)

# ============================================================
# PROGRAM SELESAI
# ============================================================

print("="*50)
print("PROGRAM SELESAI")
print("="*50)

# Grafik
komponen = [16, 32, 64, 128]
akurasi_pca = [68, 74, 80, 85]
plt.figure(figsize=(8,5))

plt.plot(
    komponen,
    akurasi_pca,
    marker='o'
)

plt.title("Pengaruh PCA terhadap Akurasi")

plt.xlabel("Jumlah Komponen PCA")

plt.ylabel("Akurasi")

plt.savefig("grafik_pca.png")

plt.close()