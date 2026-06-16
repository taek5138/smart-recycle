import cv2
import lgpio
import time
import glob
import adafruit_dht
import board
from RPLCD.i2c import CharLCD
from gpiozero import AngularServo
from ultralytics import YOLO

# LCD
lcd = CharLCD(i2c_expander='PCF8574', address=0x27,
              port=1, cols=16, rows=2)

def lcd_print(line1, line2=""):
    lcd.clear()
    lcd.cursor_pos = (0, 0)
    lcd.write_string(line1[:16])
    lcd.cursor_pos = (1, 0)
    lcd.write_string(line2[:16])

# DHT11
dht_sensor = adafruit_dht.DHT11(board.D4)

def read_dht():
    try:
        temp = dht_sensor.temperature
        humi = dht_sensor.humidity
        if temp is not None and humi is not None:
            return temp, humi
    except RuntimeError:
        pass  # DHT11 간헐적 오류 무시
    return None, None

# YOLO
model_path = glob.glob('/home/eun/model/*/models/yolo11n_recycle_ncnn_model')[0]
model = YOLO(model_path, task='detect')
NAMES = {0: "plastic", 1: "can", 2: "glass", 3: "paper"}

# Settings
TRIG = 23
ECHO = 24
DETECT_DIST = 30
CLASSIFY_INTERVAL = 5
DHT_INTERVAL = 10  # 온습도 표시 간격 (초)

LED_PINS = {
    "plastic": 17,
    "can":     27,
    "paper":   22
}

CATEGORY_MAP = {
    "plastic": "plastic",
    "can":     "can",
    "glass":   None,
    "paper":   "paper"
}

LCD_MSG = {
    "plastic": ("Plastic Bottle", "Green bin"),
    "can":     ("Can / Metal",    "Red bin"),
    "paper":   ("Paper",          "Yellow bin"),
    "glass":   ("Glass",          "Glass bin")
}

BOX_COLOR = {
    "plastic": (0, 255, 0),
    "can":     (0, 0, 255),
    "glass":   (255, 255, 0),
    "paper":   (0, 255, 255)
}

# lgpio
h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(h, TRIG)
lgpio.gpio_claim_input(h, ECHO)
for pin in LED_PINS.values():
    lgpio.gpio_claim_output(h, pin)

# Servo
servo = AngularServo(18, min_angle=0, max_angle=180,
                     min_pulse_width=0.0005, max_pulse_width=0.0025)

def open_lid():
    servo.angle = 90
    time.sleep(0.5)
    print("lid open")

def close_lid():
    servo.angle = 0
    time.sleep(0.5)
    print("lid closed")

# LED
def all_leds_off():
    for pin in LED_PINS.values():
        lgpio.gpio_write(h, pin, 0)

def blink_led(category, times=3):
    if category not in LED_PINS:
        return
    pin = LED_PINS[category]
    all_leds_off()
    for _ in range(times):
        lgpio.gpio_write(h, pin, 1)
        time.sleep(0.3)
        lgpio.gpio_write(h, pin, 0)
        time.sleep(0.3)
    lgpio.gpio_write(h, pin, 1)

# Ultrasonic
def get_distance():
    lgpio.gpio_write(h, TRIG, 1)
    time.sleep(0.00001)
    lgpio.gpio_write(h, TRIG, 0)
    timeout = time.time() + 0.05
    pulse_start = pulse_end = time.time()
    while lgpio.gpio_read(h, ECHO) == 0:
        pulse_start = time.time()
        if time.time() > timeout:
            return 999
    while lgpio.gpio_read(h, ECHO) == 1:
        pulse_end = time.time()
        if time.time() > timeout:
            return 999
    return round((pulse_end - pulse_start) * 34300 / 2, 1)

# Camera
cap = cv2.VideoCapture(0)

def capture_and_classify():
    ret, frame = cap.read()
    if not ret:
        lcd_print("Camera Error", "Try again")
        return False

    results = model.predict(frame, imgsz=640, device="cpu",
                            conf=0.35, verbose=False)

    detected = None
    display = frame.copy()

    if results and results[0].boxes and len(results[0].boxes):
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = NAMES.get(cls_id, "unknown")
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            color = BOX_COLOR.get(name, (255, 255, 255))
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, f"{name} {conf:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

        cls_id = int(results[0].boxes.cls[0])
        detected = NAMES.get(cls_id, None)
        print(f"detected: {detected} (conf: {float(results[0].boxes.conf[0]):.2f})")

    dist = get_distance()
    cv2.putText(display, f"dist: {dist}cm",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Smart Recycle", display)
    cv2.waitKey(1)
    cv2.imwrite(f"waste_{int(time.time())}.jpg", display)

    if detected == "glass":
        lcd_print("Glass", "Glass bin")
        all_leds_off()
        # 분류 후 온습도 표시
        time.sleep(2)
        show_dht_on_lcd()
        return True
    elif detected and CATEGORY_MAP.get(detected):
        category = CATEGORY_MAP[detected]
        line1, line2 = LCD_MSG[category]
        lcd_print(line1, line2)
        blink_led(category)
        # 분류 후 온습도 표시
        time.sleep(2)
        show_dht_on_lcd()
        return True
    else:
        lcd_print("No item found", "Try again")
        all_leds_off()
        return False

def show_dht_on_lcd():
    temp, humi = read_dht()
    if temp is not None:
        lcd_print(f"Temp: {temp}C", f"Humi: {humi}%")
        print(f"DHT11 - Temp: {temp}C, Humi: {humi}%")
    else:
        lcd_print("DHT11 Error", "Retrying...")

# Main loop
try:
    print("loading model...")
    lcd_print("Smart Recycle", "Ready...")
    all_leds_off()
    close_lid()
    lid_open = False
    last_classify_time = 0
    last_dht_time = 0
    print("system ready!")

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Smart Recycle", frame)
        cv2.waitKey(1)

        dist = get_distance()
        print(f"distance: {dist} cm")

        if dist < DETECT_DIST:
            now = time.time()
            if now - last_classify_time > CLASSIFY_INTERVAL:
                lcd_print("Scanning...", "Please wait")
                success = capture_and_classify()
                last_classify_time = now
                last_dht_time = now  # 분류 직후 DHT 타이머 리셋

                if success and not lid_open:
                    open_lid()
                    lid_open = True

        else:
            if lid_open:
                close_lid()
                all_leds_off()
                lcd_print("Smart Recycle", "Ready...")
                lid_open = False

            # 대기 중 주기적 온습도 표시
            now = time.time()
            if now - last_dht_time > DHT_INTERVAL:
                show_dht_on_lcd()
                last_dht_time = now
                time.sleep(2)
                lcd_print("Smart Recycle", "Ready...")

        time.sleep(0.5)

except KeyboardInterrupt:
    print("exit")
    close_lid()
    all_leds_off()
    lcd.clear()
    cap.release()
    cv2.destroyAllWindows()
    lgpio.gpiochip_close(h)
    dht_sensor.exit()
