import cv2
import os

_store_flag = False
_iteration = 0
_output_dir = "out"


def debug_set_output_directory(d):
    global _output_dir
    global _store_flag
    _output_dir = d
    _store_flag = True
    if not os.path.exists(_output_dir):
        os.makedirs(_output_dir)
    

def debug_set_iteration(i):
    global _iteration
    _iteration = i

    
def debug_store_image(label, image, format):
    global _iteration
    global _output_dir
    if _store_flag:
        cv2.imwrite(f"{_output_dir}/{_iteration:05d}-{label}.{format}", image)

        
def debug_store_svg(label, content):
    global _iteration
    global _output_dir
    with open(f"{_output_dir}/{_iteration:05d}-{label}.svg", "w") as f:
        f.write(content)

        
def debug_store_images(value):
    global _store_flag
    _store_flag = value
