import cv2
from config import cfg
import os
from util.DeepSnake import DeepSnake

weight_path = 'data/wheel_896*448.pth'
imgs_dir = 'data/test_images'
#imgs_dir = 'data1'

snake = DeepSnake(cfg.num_layers, cfg.heads, cfg.head_conv, cfg.down_ratio, weight_path)

for img_name in os.listdir(imgs_dir):
    img_path = os.path.join(imgs_dir, img_name)
    print('Test {}'.format(img_path))

    cv_img = cv2.imread(img_path)
    if cv_img is None:
        print('cannot load {}'.format(img_path))
        continue

    result = snake.predict(cv_img)
    snake.visualize_box(cv_img, result)
