import pyttsx3
import threading

def speak_text(text):
    """
    Hàm nhận vào văn bản và phát ra âm thanh trong một luồng riêng.
    """
    def run_speech():
        # Khởi tạo engine bên trong luồng
        engine = pyttsx3.init()
        
        # 1. Chỉnh tốc độ đọc (Words per minute)
        # Mặc định thường là 200 hơi nhanh, giảm xuống 150 cho rõ chữ
        engine.setProperty('rate', 150) 
        
        # 2. Chỉnh âm lượng (từ 0.0 đến 1.0)
        engine.setProperty('volume', 1.0)
        
        # 3. Chọn giọng đọc (Nam/Nữ)
        voices = engine.getProperty('voices')
        # Trên Windows, thường voices[0] là Nam (David), voices[1] là Nữ (Zira)
        if len(voices) > 1:
            engine.setProperty('voice', voices[1].id) # Đổi thành voices[0].id nếu muốn giọng nam
            
        # Ra lệnh đọc
        engine.say(text)
        engine.runAndWait()

    # Tạo và khởi chạy một luồng (thread) mới rẽ nhánh từ chương trình chính
    speech_thread = threading.Thread(target=run_speech)
    speech_thread.start()