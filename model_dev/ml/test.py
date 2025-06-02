import numpy as np
import cv2
import tensorflow as tf

def test_opencv_cuda_gemm():
    print("=== OpenCV CUDA Matrix Multiplication ===")
    a = np.random.rand(512, 512).astype(np.float32)
    b = np.random.rand(512, 512).astype(np.float32)

    a_gpu = cv2.cuda_GpuMat()
    b_gpu = cv2.cuda_GpuMat()
    a_gpu.upload(a)
    b_gpu.upload(b)

    result_gpu = cv2.cuda.gemm(a_gpu, b_gpu, 1.0, None, 0.0)
    result = result_gpu.download()
    print("Result shape:", result.shape)

def test_tensorflow_mnist():
    print("\n=== TensorFlow Minimal MNIST Training ===")
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train[:128].reshape(-1, 28, 28, 1).astype("float32") / 255.0
    y_train = tf.keras.utils.to_categorical(y_train[:128], 10)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, (3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=16, epochs=1)

if __name__ == "__main__":
    test_opencv_cuda_gemm()
    test_tensorflow_mnist()

