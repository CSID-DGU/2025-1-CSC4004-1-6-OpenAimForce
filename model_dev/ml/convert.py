import tensorflow as tf

model = tf.keras.models.load_model('result/final_aimhack_model.keras')
model.export('compat_model')  # <-- Directory, no extension!
