import cv2
import numpy as np
import torch
from config import cfg
from lib.network import Network


class DeepSnake:
    def __init__(self, num_layers, heads, head_conv, down_ratio, weight_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.net = Network(num_layers, heads, head_conv, down_ratio).to(self.device)
        # self.net.load_state_dict(torch.load(weight_path))

        # torch.save(self.net, 'full.pth')
        self.net = torch.load('full.pth').to(self.device)

        self.net.eval()
        self.down_ratio = down_ratio

    @staticmethod
    def normalize_image(inp):
        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - cfg.mean) / cfg.std
        inp = inp.transpose(2, 0, 1)
        return inp

    @staticmethod
    def get_3rd_point(a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    @staticmethod
    def get_dir(src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs
        return src_result

    @staticmethod
    def cuda_tensor_to_cv(cuda_tensor):
        cv_img = cuda_tensor.detach().cpu().numpy()
        cv_img = cv_img.transpose(1, 2, 0)
        cv_img = cv_img * cfg.std + cfg.mean
        cv_img *= 255
        cv_img = cv_img.astype(np.uint8).copy()
        return cv_img

    def get_affine_transform(self, center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale], dtype=np.float32)
        scale_tmp = scale
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = self.get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

        src[2:, :] = self.get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = self.get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return trans

    def process_img(self, img):
        width, height = img.shape[1], img.shape[0]
        # center = np.array([width // 2, height // 2])
        # scale = np.array([width, height])
        x = 32
        input_w = (width + x - 1) // x * x
        input_h = (height + x - 1) // x * x

        width_shift = input_w - width
        height_shift = input_h - height
        processed_img = cv2.copyMakeBorder(img, 0, height_shift, 0, width_shift, cv2.BORDER_CONSTANT, 128)
        # trans_input = self.get_affine_transform(center, scale, 0, [input_w, input_h])
        # processed_img = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
        processed_img = self.normalize_image(processed_img)
        return processed_img

    def predict(self, img):
        input_data = self.process_img(img)
        input_data = torch.from_numpy(input_data).unsqueeze(dim=0).to(self.device)
        # batch['inp'] = torch.from_numpy(batch['inp']).unsqueeze(dim=0).cuda()
        with torch.no_grad():
            output = self.net(input_data)

        # decode the output
        # output is a dict which contains 'detection', 'py', 'it_ex', 'it_py', , 'ct_hm', 'wh', 'ct' and so on
        # 'detection' contains x1, y1, x2, y2, score, class
        # 'py' contains bounding points of the target mask
        boxes = output['detection'][:, :4].detach().cpu().numpy() * self.down_ratio  # num * 4
        scores = output['detection'][:, 4:5].detach().cpu().numpy()                  # num * 1
        class_ids = output['detection'][:, 5:].detach().cpu().numpy()                # num * 1

        polys = output['py']
        polys = polys[-1] if isinstance(polys, list) else polys
        polys = polys.detach().cpu().numpy() * self.down_ratio                       # num * 128 * 2

        # recover data from tensor
        # img = self.cuda_tensor_to_cv(input_data[0])
        result = {'polys': polys, 'boxes': boxes, 'scores': scores, 'class_ids': class_ids}
        return result

    @staticmethod
    def visualize_box(input_img, output_result):
        polys = output_result['polys']
        boxes = output_result['boxes']
        scores = output_result['scores']
        class_ids = output_result['class_ids']

        for i in range(len(polys)):
            color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            poly, box, score, class_id = polys[i], boxes[i], scores[i].item(), int(class_ids[i].item())
            point_size = len(poly)
            for idx in range(point_size):
                next_idx = (idx + 1) % point_size
                cv2.line(input_img, (poly[idx][0], poly[idx][1]), (poly[next_idx][0], poly[next_idx][1]), color, 3)
            x1, x2, x3, x4 = box
            cv2.rectangle(input_img, (x1, x2), (x3, x4), (0, 0, 255))
            print('class_id: {}, score: {:.2f}'.format(class_id, score))
        print()
        cv2.imshow("frame", input_img)
        if cv2.waitKey(0) == 27:
            exit()

