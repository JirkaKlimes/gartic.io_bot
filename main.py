import pyautogui
import cv2 as cv
import numpy as np
import keyboard
import time
from math import sqrt
from PIL import ImageGrab
import win32api, win32con

def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv.kmeans(samples,
            clusters, 
            None,
            (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

class GarticBot:

    def __init__(self, DEBUG=False):
        self.debug = DEBUG
    

    BOARD_ORIGIN = (692, 170)
    BOARD_RESOLUTION = (962, 530)
    PENCIL = (-150, 25)
    PENCIL_SLIDER = (-147, 772)
    PENCIL_SLIDER_MIN_RANGE = (790, 665)

    PALLETE = (-100, 570)

    # DRAWING_RESOLUTION = (120, 66)
    # DRAWING_RESOLUTION = (150, 82)
    DRAWING_RESOLUTION = (200, 110)
    COLOR_VARIANCE = 128
    WHITE_THRESHOLD = 55
    CLICK_DELAY = 1e-10
    CLICK_DELAY_INTERVAL = 5
    pyautogui.PAUSE = CLICK_DELAY

    def _getRelativePos(self, pos):
        return (pos[0] + self.BOARD_ORIGIN[0], pos[1] + self.BOARD_ORIGIN[1])

    def _downScale(self, image):
        f1 = self.DRAWING_RESOLUTION[0] / image.shape[1]
        f2 = self.DRAWING_RESOLUTION[1] / image.shape[0]
        dim = (int(image.shape[1] * min(f1, f2)), int(image.shape[0] * min(f1, f2)))
        resized = cv.resize(image, dim)
        
        downscaled = kmeans_color_quantization(resized, clusters=self.COLOR_VARIANCE, rounds=1)

        if self.debug:
            cv.imshow("IMAGE", cv.resize(image, (600, int(image.shape[0]*600/image.shape[1])), interpolation=cv.INTER_AREA))
            cv.waitKey(600)
            cv.imshow("IMAGE", cv.resize(resized, (600, int(resized.shape[0]*600/resized.shape[1])), interpolation=cv.INTER_AREA))
            cv.waitKey(600)
            cv.imshow("IMAGE", cv.resize(downscaled, (600, int(downscaled.shape[0]*600/downscaled.shape[1])), interpolation=cv.INTER_AREA))
            cv.waitKey(600)
            cv.destroyAllWindows()

        return downscaled
    
    def _getColorClusters(self, image):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        clusters = {}
        for j in range(len(image)):
            for i in range(len(image[0])):
                color = f"{image[j][i][0]},{image[j][i][1]},{image[j][i][2]}"
                if color in clusters:
                    clusters[color].append((i, j))
                else:
                    clusters.update({color: [(i, j)]})
        return clusters
    
    def _equipPencil(self):
        pyautogui.click(self._getRelativePos(self.PENCIL))
    
    def _setColor(self, color):
        pyautogui.click(self._getRelativePos(self.PALLETE))
        time.sleep(0.1)
        color = color.split(",")
        keyboard.send('tab')
        time.sleep(0.01)
        keyboard.send('tab')
        time.sleep(0.01)
        keyboard.send('tab')
        time.sleep(0.01)
        keyboard.write(color[0])
        time.sleep(0.01)
        keyboard.send('tab')
        time.sleep(0.01)
        keyboard.write(color[1])
        time.sleep(0.01)
        keyboard.send('tab')
        time.sleep(0.01)
        keyboard.write(color[2])
        time.sleep(0.01)
        keyboard.send('enter')
        time.sleep(0.1)

    def _getClickPosition(self, pos):
        upscale_factor_x = self.BOARD_RESOLUTION[0] / self.DRAWING_RESOLUTION[0]
        upscale_factor_y = self.BOARD_RESOLUTION[1] / self.DRAWING_RESOLUTION[1]
        pos = (int(pos[0]*upscale_factor_x), int(pos[1]*upscale_factor_y))
        return pos

    def _setPencilThickness(self, thickness):
        pyautogui.moveTo(self._getRelativePos(self.PENCIL_SLIDER))

    def draw(self, image):
        print("DOWNSCALING")
        downscaled = self._downScale(image)
        clusters = self._getColorClusters(downscaled)
        while True:
            if keyboard.is_pressed('alt+s'):
                print("STOPPING")
                return
            if keyboard.is_pressed('alt+q'):
                quit()
            if keyboard.is_pressed('alt+d'):
                break
        time.sleep(0.2)
        print("DRAWING")
        self._equipPencil()


        for color in clusters:
            channels = color.split(",")
            dist = sqrt(pow(int(channels[0])-255, 2) + pow(int(channels[1])-255, 2) + pow(int(channels[2])-255, 2))
            if dist < self.WHITE_THRESHOLD:
                continue
            print(f'Color: {color}')
            self._setColor(color)
            for i, pixel in enumerate(clusters[color]):
                pos = self._getClickPosition(pixel)
                pos = self._getRelativePos(pos)

                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, int(pos[0]/win32api.GetSystemMetrics(0)*65535), int(pos[1]/win32api.GetSystemMetrics(1)*65535) ,0 ,0)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                if i%self.CLICK_DELAY_INTERVAL==0: time.sleep(self.CLICK_DELAY)
                
                if keyboard.is_pressed('alt+s'):
                    print("STOPED")
                    return
        print("DONE")
    def run(self):
        while True:
            if keyboard.is_pressed('alt+q'):
                break
            if keyboard.is_pressed('alt+c'):
                image = np.array(ImageGrab.grabclipboard())[:,:,:3]
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                self.draw(image)

bot = GarticBot(DEBUG=True)
bot.run()


