from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')

    results = model.train(
        data='C:/WebAI/Dataset/data.yaml',
        epochs=80,                 
        imgsz=640,                 
        batch=16,                  
        device='cpu',               
        project='C:/WebAI/Training_Results', 
        name='asl_yolov8_run',     
        save=True                  
    )

if __name__ == '__main__':
    main()