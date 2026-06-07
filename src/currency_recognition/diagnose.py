import cv2, numpy as np, sys, tensorflow as tf

MODEL_PATH = r'C:\projects\vispay\src\currency_recognition\currency_recognition\models\currency\VisPay_currency_model.tflite'
CLASS_MAP  = {'0':'10','1':'100','2':'20','3':'200','4':'2000','5':'50','6':'500','7':'Background'}

interp = tf.lite.Interpreter(model_path=MODEL_PATH)
interp.allocate_tensors()
inp = interp.get_input_details()
out = interp.get_output_details()

cap = cv2.VideoCapture(0)
print('Hold the 20 rupee note steady... press S to capture, Q to quit')

while True:
    ret, frame = cap.read()
    cv2.imshow('Capture', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, 0)
        interp.set_tensor(inp[0]['index'], tensor)
        interp.invoke()
        probs  = interp.get_tensor(out[0]['index'])[0]

        print('\n--- All class probabilities ---')
        for i, p in enumerate(probs):
            label = CLASS_MAP[str(i)]
            arrow = ' <<<' if p == max(probs) else ''
            print(f'  Class {i}  ({label:>10} rupees):  {p:.4f}{arrow}')
        print()

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()