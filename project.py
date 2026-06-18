import os
import cv2
import albumentations as A
class YoloAugmenter:
    def __init__(self, train_img_dir, train_label_dir):
        self.img_dir = train_img_dir
        self.label_dir = train_label_dir

        self.transform = A.Compose([
            A.Affine(scale=(1.0, 1.2), rotate=(-5, 5), p=1.0),
            A.ToGray(p=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.2, p=0.5),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3)

        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=['class_labels']))

    def read_yolo_label(self, label_path):
        bboxes, classes = [], []
        if not os.path.exists(label_path): return bboxes, classes
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    classes.append(int(float(parts[0])))
                    bboxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
        return bboxes, classes

    def save_yolo_label(self, bboxes, classes, save_path):
        with open(save_path, 'w') as f:
            for i in range(len(bboxes)):
                line = f"{classes[i]} {' '.join([str(round(x, 6)) for x in bboxes[i]])}\n"
                f.write(line)

    def process_dataset(self, outputs_per_image=3):
        image_files = [f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"Founded {len(image_files)} images. Duplicating {outputs_per_image}x...")

        for img_name in image_files:
            img_path = os.path.join(self.img_dir, img_name)
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(self.label_dir, label_name)

            image = cv2.imread(img_path)
            if image is None: continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes, classes = self.read_yolo_label(label_path)

            if len(bboxes) == 0: continue

            for i in range(outputs_per_image):
                try:
                    transformed = self.transform(image=image, bboxes=bboxes, class_labels=classes)

                    if len(transformed['bboxes']) > 0:
                        new_base_name = f"{os.path.splitext(img_name)[0]}_aug_{i}"
                        new_img_path = os.path.join(self.img_dir, new_base_name + '.jpg')
                        new_label_path = os.path.join(self.label_dir, new_base_name + '.txt')

                        cv2.imwrite(new_img_path, cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))
                        self.save_yolo_label(transformed['bboxes'], transformed['class_labels'], new_label_path)
                except Exception as e:
                    pass
        print("Success!")
TRAIN_IMAGES = "C:/WebAI/Dataset/train/images"
TRAIN_LABELS = "C:/WebAI/Dataset/train/labels"

augmenter = YoloAugmenter(train_img_dir=TRAIN_IMAGES, train_label_dir=TRAIN_LABELS)
augmenter.process_dataset(outputs_per_image=3)