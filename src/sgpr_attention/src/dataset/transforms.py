import numpy as np
from torchvision import transforms
import random
from sklearn.preprocessing import normalize

class RotatePointCloud:
    def __call__(self,xyz):
        rotated_xyz = np.zeros(xyz.shape, dtype=np.float32)
        for k in range(xyz.shape[0]):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            # along y
            # rotation_matrix = np.array([[cosval, 0, sinval],
            #               [0, 1, 0],
            #               [-sinval, 0, cosval]])
            # shape_pc = xyz[k, ...]
            # along z
            rotation_matrix = np.array([[cosval, -sinval, 0],
                                        [sinval, cosval, 0],
                                        [0, 0, 1]])
            shape_pc = xyz[k, ...]
            rotated_xyz[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        return rotated_xyz


class JitterPointCloud:
    def __call__(self,xyz, sigma=0.01, clip=0.05):
        B, N, C = xyz.shape
        assert (clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
        jittered_data += xyz
        return jittered_data

class RandomScalePointCloud:
    def __call__(self,batch_data, scale_low=0.8, scale_high=1.25):
        B, N, C = batch_data.shape
        scales = np.random.uniform(scale_low, scale_high, B)
        for batch_index in range(B):
            batch_data[batch_index,:,:] *= scales[batch_index]
        return batch_data

class RotatePerturbationPointCloud:
    def __call__(self,batch_data, angle_sigma=0.015, angle_clip=0.045):

        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        for k in range(batch_data.shape[0]):
            angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
            Rx = np.array([[1,0,0],
                     [0,np.cos(angles[0]),-np.sin(angles[0])],
                     [0,np.sin(angles[0]),np.cos(angles[0])]])
            Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                     [0,1,0],
                     [-np.sin(angles[1]),0,np.cos(angles[1])]])
            Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                     [np.sin(angles[2]),np.cos(angles[2]),0],
                     [0,0,1]])
            R = np.dot(Rz, np.dot(Ry,Rx))
            shape_pc = batch_data[k, ...]
            rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
        return rotated_data

class ShiftPointCloud:
    def __call__(self,batch_data, shift_range=0.3):
        B, N, C = batch_data.shape
        shifts = np.random.uniform(-shift_range, shift_range, (B,3))
        for batch_index in range(B):
            batch_data[batch_index,:,:] += shifts[batch_index,:]
        return batch_data

class FlipPointCloud:
    def __call__(self,batch_data):

        if random.random() > 0.5:
            batch_data[:, :, 0] = -batch_data[:, :, 0]
        return batch_data

class PerturbFeature:
    def __call__(self,feature):
        perturbed_feature = np.random.normal(0, scale=1, size=feature.shape) + feature
        return perturbed_feature

class ShuffleFeature:
    def __call__(self,feature):
        shuffled_feature=np.random.permutation(feature[0,:,:])
        return np.expand_dims(shuffled_feature,axis=0)

class MaskFeature:
    def __call__(self,feature):
      mask=np.random.randn(*feature.shape)
      # print(np.where(mask<0.5))
      feature[np.where(mask<0.5)]=0
      return feature

class NodeAug:
    def __call__(self,feature):
        feature1=np.expand_dims(np.random.permutation(feature[0,:,:]),axis=0)
        sigma=np.random.random()
        return sigma*feature+(1-sigma)*feature1

if __name__=="__main__":

    x=np.random.randn(3,10,24)
    mask=np.random.randn(*x.shape)
    print(mask)