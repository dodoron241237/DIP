import os.path
import cv2
import json
def get_resize_ratio(file_path):
    #read image to resize
    file_path = file_path.replace('label','image')
    file_path = file_path[:-5] + ".png"
    image = cv2.imread(file_path)
    #resize to 1000*1000
    resized_image = cv2.resize(image, (480, 480))  
    #get resize ratio      
    resize_ratio = resized_image.shape[0] / image.shape[0]
    #write to file
    cv2.imwrite(file_path, resized_image)
    return resize_ratio


def get_data_dicts(directory):
    for i, filename in enumerate([file for file in os.listdir(directory) if file.endswith('.json')]):
        json_file = os.path.join(directory, filename)
        
        #要處理每場圖片的大小 要resize
        
        resize_ratio = get_resize_ratio(json_file)
        print(resize_ratio)
        with open(json_file) as f:
            img_anns = json.load(f)
      
        annos = img_anns["shapes"]
        for anno in annos:
            px = [a[0] *resize_ratio for a in anno['points']] # x coord
            py = [a[1] *resize_ratio for a in anno['points']] # y-coord
            for i, a in enumerate(anno['points']):
                anno['points'][i] = (a[0] * resize_ratio, a[1] * resize_ratio)
            with open(json_file, 'w') as f:
                json.dump(img_anns, f)
    