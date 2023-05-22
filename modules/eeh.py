# File: eeh.py
# Description: Edge Enhancement
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np
import os.path as op
from scipy import ndimage
import cv2
from .basic_module import BasicModule, register_dependent_modules
from .helpers import generic_filter, gen_gaussian_kernel
class tracking:
    def __init__(self, tr):
        self.im = tr[0]
        strongs = tr[1]

        self.vis = np.zeros(self.im.shape, bool)
        self.dx = [1, 0, -1,  0, -1, -1, 1,  1]
        self.dy = [0, 1,  0, -1,  1, -1, 1, -1]
        for s in strongs:
            if not self.vis[s]:
                self.dfs(s)
        for i in range(self.im.shape[0]):
            for j in range(self.im.shape[1]):
                self.im[i, j] = 1.0 if self.vis[i, j] else 0.0

    def dfs(self, origin):
        q = [origin]
        while len(q) > 0:
            s = q.pop()
            self.vis[s] = True
            self.im[s] = 1
            for k in range(len(self.dx)):
                for c in range(1, 16):
                    nx, ny = s[0] + c * self.dx[k], s[1] + c * self.dy[k]
                    if self.exists(nx, ny) and (self.im[nx, ny] >= 0.5) and (not self.vis[nx, ny]):
                        q.append((nx, ny))
        pass

    def exists(self, x, y):
        return x >= 0 and x < self.im.shape[0] and y >= 0 and y < self.im.shape[1]

@register_dependent_modules('csc')
class EEH(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        kernel = gen_gaussian_kernel(kernel_size=5, sigma=1.2)
        self.gaussian = (1024 * kernel / kernel.max()).astype(np.int32)  # x1024

        t1, t2 = self.params.flat_threshold, self.params.edge_threshold
        threshold_delta = np.clip(t2 - t1, 1E-6, None)
        self.middle_slope = np.array(self.params.edge_gain * t2 / threshold_delta, dtype=np.int32)  # x256
        self.middle_intercept = -np.array(self.params.edge_gain * t1 * t2 / threshold_delta, dtype=np.int32)  # x256
        self.edge_gain = np.array(self.params.edge_gain, dtype=np.int32)  # x256
    
    def gradient(self, im):
         # Sobel operator
         op1 = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]])
         op2 = np.array([[-1, -2, -1],
                      [ 0,  0,  0],
                      [ 1,  2,  1]])
         kernel1 = np.zeros(im.shape)
         kernel1[:op1.shape[0], :op1.shape[1]] = op1
         kernel1 = np.fft.fft2(kernel1)

         kernel2 = np.zeros(im.shape)
         kernel2[:op2.shape[0], :op2.shape[1]] = op2
         kernel2 = np.fft.fft2(kernel2)

         fim = np.fft.fft2(im)
         Gx = np.real(np.fft.ifft2(kernel1 * fim)).astype(float)
         Gy = np.real(np.fft.ifft2(kernel2 * fim)).astype(float)

         G = np.sqrt(Gx**2 + Gy**2)
         Theta = np.arctan2(Gy, Gx) * 180 / np.pi
         return G, Theta
    
    def NMS(self, det, phase):
        gmax = np.zeros(det.shape)
        for i in range(gmax.shape[0]):
          for j in range(gmax.shape[1]):
            if phase[i][j] < 0:
              phase[i][j] += 360

            if ((j+1) < gmax.shape[1]) and ((j-1) >= 0) and ((i+1) < gmax.shape[0]) and ((i-1) >= 0):
              # 0 degrees
              if (phase[i][j] >= 337.5 or phase[i][j] < 22.5) or (phase[i][j] >= 157.5 and phase[i][j] < 202.5):
                if det[i][j] >= det[i][j + 1] and det[i][j] >= det[i][j - 1]:
                  gmax[i][j] = det[i][j]
              # 45 degrees
              if (phase[i][j] >= 22.5 and phase[i][j] < 67.5) or (phase[i][j] >= 202.5 and phase[i][j] < 247.5):
                if det[i][j] >= det[i - 1][j + 1] and det[i][j] >= det[i + 1][j - 1]:
                  gmax[i][j] = det[i][j]
              # 90 degrees
              if (phase[i][j] >= 67.5 and phase[i][j] < 112.5) or (phase[i][j] >= 247.5 and phase[i][j] < 292.5):
                if det[i][j] >= det[i - 1][j] and det[i][j] >= det[i + 1][j]:
                  gmax[i][j] = det[i][j]
              # 135 degrees
              if (phase[i][j] >= 112.5 and phase[i][j] < 157.5) or (phase[i][j] >= 292.5 and phase[i][j] < 337.5):
                if det[i][j] >= det[i - 1][j - 1] and det[i][j] >= det[i + 1][j + 1]:
                  gmax[i][j] = det[i][j]
        return gmax
    
    def thresholding(self, im):
        thres  = np.zeros(im.shape)
        strong = 1.0
        weak   = 0.5
        mmax = np.max(im)
        lo, hi = 0.1 * mmax, 0.8 * mmax
        strongs = []
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                px = im[i][j]
                if px >= hi:
                    thres[i][j] = strong
                    strongs.append((i, j))
                elif px >= lo:
                    thres[i][j] = weak
        return thres, strongs
    
    def SobelFilter(self,img, direction):
        if(direction == 'x'):
            Gx = np.array([[-2,0,+2], [-4,0,+4],  [-2,0,+2]])
            Res = ndimage.convolve(img, Gx)
            #Res = ndimage.convolve(img, Gx, mode='constant', cval=0.0)
        if(direction == 'y'):
            Gy = np.array([[-2,-4,-2], [0,0,0], [+2,+4,+2]])
            Res = ndimage.convolve(img, Gy)
            #Res = ndimage.convolve(img, Gy, mode='constant', cval=0.0)

        return Res
    
    # Normalize the pixel array, so that values are <= 1
    def Normalize(self, img):
        #img = np.multiply(img, 255 / np.max(img))
        img = img/np.max(img)
        return img
    
    def NonMaxSupWithInterpol(self, Gmag, Grad, Gx, Gy):
        NMS = np.zeros(Gmag.shape)

        for i in range(1, int(Gmag.shape[0]) - 1):
            for j in range(1, int(Gmag.shape[1]) - 1):
                if((Grad[i,j] >= 0 and Grad[i,j] <= 45) or (Grad[i,j] < -135 and Grad[i,j] >= -180)):
                    yBot = np.array([Gmag[i,j+1], Gmag[i+1,j+1]])
                    yTop = np.array([Gmag[i,j-1], Gmag[i-1,j-1]])
                    x_est = np.absolute(Gy[i,j]/Gmag[i,j])
                    if (Gmag[i,j] >= ((yBot[1]-yBot[0])*x_est+yBot[0]) and Gmag[i,j] >= ((yTop[1]-yTop[0])*x_est+yTop[0])):
                        NMS[i,j] = Gmag[i,j]
                    else:
                        NMS[i,j] = 0
                if((Grad[i,j] > 45 and Grad[i,j] <= 90) or (Grad[i,j] < -90 and Grad[i,j] >= -135)):
                    yBot = np.array([Gmag[i+1,j] ,Gmag[i+1,j+1]])
                    yTop = np.array([Gmag[i-1,j] ,Gmag[i-1,j-1]])
                    x_est = np.absolute(Gx[i,j]/Gmag[i,j])
                    if (Gmag[i,j] >= ((yBot[1]-yBot[0])*x_est+yBot[0]) and Gmag[i,j] >= ((yTop[1]-yTop[0])*x_est+yTop[0])):
                        NMS[i,j] = Gmag[i,j]
                    else:
                        NMS[i,j] = 0
                if((Grad[i,j] > 90 and Grad[i,j] <= 135) or (Grad[i,j] < -45 and Grad[i,j] >= -90)):
                    yBot = np.array([Gmag[i+1,j] ,Gmag[i+1,j-1]])
                    yTop = np.array([Gmag[i-1,j] ,Gmag[i-1,j+1]])
                    x_est = np.absolute(Gx[i,j]/Gmag[i,j])
                    if (Gmag[i,j] >= ((yBot[1]-yBot[0])*x_est+yBot[0]) and Gmag[i,j] >= ((yTop[1]-yTop[0])*x_est+yTop[0])):
                        NMS[i,j] = Gmag[i,j]
                    else:
                        NMS[i,j] = 0
                if((Grad[i,j] > 135 and Grad[i,j] <= 180) or (Grad[i,j] < 0 and Grad[i,j] >= -45)):
                    yBot = np.array([Gmag[i,j-1] ,Gmag[i+1,j-1]])
                    yTop = np.array([Gmag[i,j+1] ,Gmag[i-1,j+1]])
                    x_est = np.absolute(Gy[i,j]/Gmag[i,j])
                    if (Gmag[i,j] >= ((yBot[1]-yBot[0])*x_est+yBot[0]) and Gmag[i,j] >= ((yTop[1]-yTop[0])*x_est+yTop[0])):
                        NMS[i,j] = Gmag[i,j]
                    else:
                        NMS[i,j] = 0

        return NMS
    
    def DoThreshHyst(self, img):
        highThresholdRatio = 0.2  
        lowThresholdRatio = 0.15 
        GSup = np.copy(img)
        h = int(GSup.shape[0])
        w = int(GSup.shape[1])
        highThreshold = np.max(GSup) * highThresholdRatio
        lowThreshold = highThreshold * lowThresholdRatio    
        x = 0.1
        oldx=0

        # The while loop is used so that the loop will keep executing till the number of strong edges do not change, i.e all weak edges connected to strong edges have been found
        while(oldx != x):
            oldx = x
            for i in range(1,h-1):
                for j in range(1,w-1):
                    if(GSup[i,j] > highThreshold):
                        GSup[i,j] = 1
                    elif(GSup[i,j] < lowThreshold):
                        GSup[i,j] = 0
                    else:
                        if((GSup[i-1,j-1] > highThreshold) or 
                            (GSup[i-1,j] > highThreshold) or
                            (GSup[i-1,j+1] > highThreshold) or
                            (GSup[i,j-1] > highThreshold) or
                            (GSup[i,j+1] > highThreshold) or
                            (GSup[i+1,j-1] > highThreshold) or
                            (GSup[i+1,j] > highThreshold) or
                            (GSup[i+1,j+1] > highThreshold)):
                            GSup[i,j] = 1
            x = np.sum(GSup == 1)

        GSup = (GSup == 1) * GSup # This is done to remove/clean all the weak edges which are not connected to strong edges

        return GSup

    def execute(self, data):
        y_image = data['y_image'].astype(np.int32)
        #  Gaussian filtering
        # gaussian_y_image = generic_filter(y_image, self.gaussian)
        # grim, gphase = self.gradient(gaussian_y_image)
        # gmax = self.NMS(grim, gphase)
        # thres = self.thresholding(gmax)
        # edge = tracking(thres)
        # data['y_image'] = (y_image + edge.im).astype(np.uint8)
        # data['edge_map'] = edge.im

        # OUTPUT_DIR = './output'
        # edge_map = data['edge_map']
        # edge_map_path = op.join(OUTPUT_DIR, 'edge.png')
        # print(data['edge_map'])
        # cv2.imwrite(edge_map_path, edge_map)

        # y_gray = np.dot(y_image[..., :3], [0.299, 0.587, 0.114])
        # y_gray_

       

        dx = self.SobelFilter(y_image, 'x')
        dy = self.SobelFilter(y_image, 'y')

        # dx = ndimage.sobel(y_image, axis=1) # horizontal derivative
        # dy = ndimage.sobel(y_image, axis=0) # vertical derivative

        # Mag = np.hypot(dx, dy)
        # Mag = self.Normalize(Mag)
        # gradient = np.degrees(np.arctan2(dy,dx))

        # NMS = self.NonMaxSupWithInterpol(Mag, gradient, dx, dy)
        # NMS = self.Normalize(NMS)
    
        
        edge = dx + dy
        edge = np.clip(edge, self.params.flat_threshold, self.params.edge_threshold * 2)



        #  Cal Gradient
        OUTPUT_DIR = './output'
        edg_img_path = op.join(OUTPUT_DIR, 'edg_img.png')
        cv2.imwrite(edg_img_path, edge)
        data['y_image'] = (y_image).astype(np.uint8)
        data['edge_map'] = edge
        # delta = y_image - generic_filter(y_image, self.gaussian)
        # sign_map = np.sign(delta)
        # abs_delta = np.abs(delta)

        # middle_delta = np.right_shift(self.middle_slope * abs_delta + self.middle_intercept, 8)
        # edge_delta = np.right_shift(self.edge_gain * abs_delta, 8)
        # enhanced_delta = (
        #         (abs_delta > self.params.flat_threshold) * (abs_delta <= self.params.edge_threshold) * middle_delta +
        #         (abs_delta > self.params.edge_threshold) * edge_delta
        # )

        # enhanced_delta = sign_map * np.clip(enhanced_delta, 0, self.params.delta_threshold)
        # eeh_y_image = np.clip(y_image + enhanced_delta, 0, self.cfg.saturation_values.sdr)

        # data['y_image'] = eeh_y_image.astype(np.uint8)
        # data['edge_map'] = delta
