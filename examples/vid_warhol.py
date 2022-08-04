import argparse

from PIL import Image
import numpy as np
import cv2

from wrapify.connect.wrapper import MiddlewareCommunicator
from wrapify.config.manager import ConfigManager

"""
Warhol effect on captures from camera

A large portion of this code is from https://github.com/mosesschwartz/warhol_effect/blob/master/warhol_effect.py

Here we demonstrate:
1. Using the Image messages
2. Partial execution of the same script on different machines or as different process
3. Demonstrating single return, multiple returns, and list returns
4. Demonstrating the disabling of a function
5. Using the ConfigManager to load a configuration file

Note: This is more of a stress test playground, with many operations that are unnecessary
        and depending on your connection speed, the output may be distorted. Consider reducing the transformations
        or transmissions (remove wrapped functions). 

Run: (in any order)
    # On machine 1 (or process 1): The capturing is done here
    python3 vid_warhol.py --cfg-file vid_warhol_cfg1.yml
    # On machine 2 (or process 2): The plotted image should appear here
    python3 vid_warhol.py --cfg-file vid_warhol_cfg2.yml
    # On machine 3 (or process 3): Without this, the other two processes halt
    python3 vid_warhol.py --cfg-file vid_warhol_cfg3.yml
"""

COLORSET = [
    {
        'bg' : (255,255,0,255),
        'fg' : (50,9,125,255),
        'skin': (118,192,0,255)
    },
    {
        'bg' : (0,122,240,255),
        'fg' : (255,0,112,255),
        'skin': (255,255,0,255)
    },
    {
        'bg' : (50,0,130,255),
        'fg' : (255,0,0,255),
        'skin': (243,145,192,255)
    },
    {
        'bg' : (255,126,0,255),
        'fg' : (134,48,149,255),
        'skin': (111,185,248,255)
    },
    {
        'bg' : (255,0,0,255),
        'fg' : (35,35,35,255),
        'skin': (255,255,255,255)
    },
    {
        'bg' : (122,192,0,255),
        'fg' : (255,89,0,255),
        'skin': (250,255,160,255)
    },
    {
        'bg' : (0,114,100,255),
        'fg' : (252,0,116,255),
        'skin': (250,250,230,255)
    },
    {
        'bg' : (250,255,0,255),
        'fg' : (254,0,0,255),
        'skin': (139,198,46,255)
    },
    {
        'bg' : (253,0,118,255),
        'fg' : (51,2,126,255),
        'skin': (255,105,0,255)
    }
]


class Warholify(MiddlewareCommunicator):
    def __init__(self, vid_src, img_width, img_height):
        super(Warholify, self).__init__()
        self.vid_cap = None
        self.vid_src = vid_src
        self.img_width = img_width
        self.img_height = img_height

    @staticmethod
    def cv2_to_pil(img):
        if isinstance(img, Image.Image):
            return img
        else:
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    @staticmethod
    def pil_to_cv2(img):
        if isinstance(img, Image.Image):
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        else:
            return img

    @MiddlewareCommunicator.register("Image", "yarp", "Warholify", "/vid_warhol/darken_bg", carrier="tcp", width="$img_width", height="$img_height", rgb=True)
    def darken_bg(self, img, color, img_width, img_height):
        img = self.cv2_to_pil(img)
        # composite image on top of a single-color image, effectively turning all transparent parts to that color
        color_layer = Image.new('RGBA', (img_width, img_height), color)
        masked_img = Image.composite(img, color_layer, img)
        masked_img = self.pil_to_cv2(masked_img)
        return masked_img,

    @MiddlewareCommunicator.register("Image", "yarp", "Warholify", "/vid_warhol/color_bg_fg", carrier="tcp", width="$img_width", height="$img_height", rgb=True)
    def color_bg_fg(self, img, bg_color, fg_color, img_width, img_height):
        img = self.cv2_to_pil(img)
        # change transparent background to bg_color and change everything non-transparent to fg_color
        fg_layer = Image.new('RGBA', (img_width, img_height), fg_color)
        bg_layer = Image.new('RGBA', (img_width, img_height), bg_color)
        masked_img = Image.composite(fg_layer, bg_layer, img)
        masked_img = self.pil_to_cv2(masked_img)
        return masked_img,

    @MiddlewareCommunicator.register("Image", "yarp", "Warholify", "/vid_warhol/white_to_color", carrier="tcp", width="$img_width", height="$img_height", rgb=True)
    def white_to_color(self, img, color, img_width, img_height):
        img = self.cv2_to_pil(img).convert('RGBA')
        # change all colors close to white and non-transparent (alpha > 0) to be color
        threshold = 50
        dist = 10
        arr = np.array(np.asarray(img))
        r, g, b, a = np.rollaxis(arr, axis=-1)
        mask = ((r > threshold)
                & (g > threshold)
                & (b > threshold)
                & (np.abs(r - g) < dist)
                & (np.abs(r - b) < dist)
                & (np.abs(g - b) < dist)
                & (a > 0)
                )
        arr[mask] = color
        img = Image.fromarray(arr, mode='RGBA')
        img = self.pil_to_cv2(img)
        return img,

    def make_warhol_single(self, img, bg_color, fg_color, skin_color):
        # create a single warhol-serigraph-style image
        bg_fg_layer, = self.color_bg_fg(img, bg_color, fg_color, img_width=img.size[0], img_height=img.size[1])
        bg_fg_layer = self.cv2_to_pil(bg_fg_layer).convert('RGBA')
        temp_dark_image, = self.darken_bg(img, (0, 0, 0, 255), img_width=img.size[0], img_height=img.size[1])
        temp_dark_image = self.cv2_to_pil(temp_dark_image).convert('RGBA')
        skin_mask, = self.white_to_color(temp_dark_image, (0, 0, 0, 0), img_width=temp_dark_image.size[0], img_height=temp_dark_image.size[1])
        skin_mask = self.cv2_to_pil(skin_mask).convert('RGBA')

        skin_layer = Image.new('RGBA', img.size, skin_color)
        out = Image.composite(bg_fg_layer, skin_layer, skin_mask)
        return out

    @MiddlewareCommunicator.register([["Image", "yarp", "Warholify", "/vid_warhol/warhol_" + str(x), {"carrier": "tcp", "width": "$img_width", "height": "$img_height"}] for x in range(9)])
    @MiddlewareCommunicator.register("Image", "Warholify", "/vid_warhol/warhol_combined", carrier="tcp", width="$img_width", height="$img_height")
    def combine_to_one(self, warhols, img_width, img_height):
        warhols_new = []
        for warhol in warhols:
            warhols_new.append(self.cv2_to_pil(warhol))
        warhols = warhols_new
        blank_img = Image.new("RGB", (img_width * 3, img_height * 3))
        blank_img.paste(warhols[0], (0, 0))
        blank_img.paste(warhols[1], (img_width, 0))
        blank_img.paste(warhols[2], (img_width * 2, 0))
        blank_img.paste(warhols[3], (0, img_height))
        blank_img.paste(warhols[4], (img_width, img_height))
        blank_img.paste(warhols[5], (img_width * 2, img_height))
        blank_img.paste(warhols[6], (0, img_height * 2))
        blank_img.paste(warhols[7], (img_width, img_height * 2))
        blank_img.paste(warhols[8], (img_width * 2, img_height * 2))
        blank_img = self.pil_to_cv2(blank_img)
        warhols_new = []
        for warhol in warhols:
            warhols_new.append(self.pil_to_cv2(warhol))
        warhols = warhols_new
        return warhols, blank_img

    def warholify_images(self, img):
        img = self.cv2_to_pil(img).convert('RGBA')
        warhols = []
        for colors in COLORSET:
            bg = colors['bg']
            fg = colors['fg']
            skin = colors['skin']
            warhols.append(self.make_warhol_single(img, bg, fg, skin))

        img_width, img_height = img.size[:2]

        (_, blank_img) = self.combine_to_one(warhols, img_width=img_width, img_height=img_height)

        return blank_img

    @MiddlewareCommunicator.register("Image", "yarp", "Warholify", "/vid_warhol/feed", carrier="tcp", width="$img_width", height="$img_height")
    def capture_vid(self, img_width, img_height):
        if self.vid_cap is None:
            self.vid_cap = cv2.VideoCapture(self.vid_src)
        if self.vid_cap.isOpened():
            grabbed, img = self.vid_cap.read()
            if grabbed:
                return img,
            return np.random.random((img_width, img_height, 3)) * 255,
        return np.random.random((img_width, img_height, 3)) * 255,

    def transform(self, img_width=640, img_height=480):
        img, = self.capture_vid(img_width=img_width, img_height=img_height)
        if img is not None:
            img = self.warholify_images(img)
            img = self.pil_to_cv2(img)
            img = cv2.resize(img, dsize=(img_width, img_height), interpolation=cv2.INTER_CUBIC)
            return img


    @MiddlewareCommunicator.register("Image", "yarp", "Warholify", "/vid_warhol/final_img", carrier="tcp", width="$img_width", height="$img_height", rgb=True)
    def display(self, img, img_width, img_height):
        cv2.imshow("Warhol", img)
        cv2.waitKey(1)
        return img,

    def run(self):
        while True:
            img_width, img_height = self.img_width, self.img_height
            img = self.transform(img_width=img_width, img_height=img_height)
            if img is not None:
                self.display(img, img_width=img_width, img_height=img_height)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.vid_cap.release()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-file", type=str, default="vid_warhol_cfg1.yml",
                        help="The transmission mode configuration file")
    parser.add_argument("--vid-src", default=0,
                        help="The video source. A string indicating a filename or an integer indicating the device id")
    parser.add_argument("--img-width", type=int, default=640, help="The image width")
    parser.add_argument("--img-height", type=int, default=480, help="The image height")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args.cfg_file)
    ConfigManager(args.cfg_file)
    warholify = Warholify(vid_src=args.vid_src, img_width=args.img_width, img_height=args.img_height)
    warholify.run()