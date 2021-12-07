import cv2, os, random, colorsys, onnxruntime, time, argparse, uuid, logging
from PIL import Image, ImageDraw, ImageFont
import numpy as np




parser = argparse.ArgumentParser("YOLOX Inference")
parser.add_argument("-m", "--model", type=str, default="tiny.onnx", help="please model name .. onnx", required = True)
parser.add_argument("-size", "--size", type=int, default = 640, help="size", required = False)
parser.add_argument('--i', type = str, required = True, default = False)
parser.add_argument('--o', type = str, required = True, default = False)






class Detection(object):
    def __init__(self, path_model, path_classes, image_shape):
        self.session = onnxruntime.InferenceSession(path_model)
        self.class_labels, self.num_names = self.get_classes(path_classes)
        self.image_shape = image_shape
        self.font = ImageFont.truetype('font.otf', 10)
        self.colors()
    

    
    def colors(self):
        hsv_tuples = [(x / len(self.class_labels), 1., 1.) for x in range(len(self.class_labels))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        class_colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        np.random.seed(43)
        np.random.shuffle(colors)
        np.random.seed(None)
        self.class_colors = np.tile(class_colors, (16, 1))
    
    
    def cvtColor(self, image):
        if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
            return image 
        else:
            image = image.convert('RGB')
            return image 
    
    def get_classes(self, classes_path):
        with open(classes_path, encoding='utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names, len(class_names)


    def preprocess_input(self, image):
        image /= 255.0
        image -= np.array([0.485, 0.456, 0.406])
        image /= np.array([0.229, 0.224, 0.225])
        return image
    
    
    def resize_image(self, image, size):
        iw, ih  = image.size
        w, h    = size
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)
        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return new_image
    
    
    def inference(self, image, size):
        ort_inputs = {self.session.get_inputs()[0].name:image, self.session.get_inputs()[1].name:size}
        box_out, scores_out, classes_out = self.session.run(None, ort_inputs)
        return box_out, scores_out, classes_out
    

    
    def draw_detection(self, image, boxes_out, scores_out, classes_out):
        image_pred = image.copy()
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.image_shape), 1))
        for i, c in reversed(list(enumerate(classes_out))):
            draw = ImageDraw.Draw(image_pred)
            predicted_class = self.class_labels[c]
            box = boxes_out[i]
            score = scores_out[i]
            label = '{}:{:.2f}%'.format(predicted_class, score*100)
            label_size = draw.textsize(label, self.font)
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            if top - label_size[1] >= 0:
                  text_origin = np.array([left, top - label_size[1]])
            else:
                  text_origin = np.array([left, top + 1])
            draw.rectangle([left, top, right, bottom], outline= tuple(self.class_colors[c]), width=2)
            draw.text(text_origin, label, fill = (255,255,0), font = self.font)      
            del draw
        return np.array(image_pred)
    
    
    
    def predict_image(self, image, video = False):
        if not video: image = Image.open(image)
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)       
        image = self.cvtColor(image)
        image_data  = self.resize_image(image, (self.image_shape[1], self.image_shape[0]))
        image_data  = np.expand_dims(self.preprocess_input(np.array(image_data, dtype='float32')), 0)       
        box_out, scores_out, classes_out = self.inference(image_data,input_image_shape)
        image_pred = self.draw_detection(image, box_out, scores_out, classes_out)
        return np.array(image_pred)
        
        
        
    def detection_video(self, video_path:str, output_path:str, fps = 25):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frame_height, frame_width, _ = frame.shape
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width,frame_height))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            output = self.predict_image(frame, True)
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            out.write(output)        
        out.release()



logging.basicConfig(filename=f'log/app.log', filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S')
args_parser = parser.parse_args()

args = {"path_model":args_parser.model, "path_classes":'classes.txt', "image_shape":(args_parser.size,args_parser.size)}
cls = Detection(**args)

start_time = time.time()
logging.info(f'load model {args_parser.model}')

# video
#cls.detection_video(video_path = args_parser.i, output_path = f"{args_parser.o}.avi", fps = 25)



end_time =  time.time() - start_time
end_time = end_time/60


image = cls.predict_image(args_parser.i)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imwrite(args_parser.o, image)
logging.info(f'[INFO]: save out {args_parser.o}')

