import os
import glob
import time
import torch
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize, ToPILImage
from torchvision import transforms
import torchvision.transforms.functional as F
import random
import numpy as np
import h5py


class gtav_city(data.Dataset):
    CS_PALETTE = {
    (128, 64, 128): 'cs',
    (244, 35, 232): 'cs',
    (70, 70, 70): 'cs',
    (102, 102, 156): 'cs',
    (190, 153, 153): 'cs',
    (153, 153, 153): 'cs',
    (250, 170, 30): 'cs',
    (220, 220, 0): 'cs',
    (107, 142, 35): 'cs',
    (152, 251, 152): 'cs',
    (70, 130, 180): 'cs',
    (220, 20, 60): 'cs',
    (255, 0, 0): 'cs',
    (0, 0, 142): 'cs',
    (0, 0, 70): 'cs',
    (0, 60, 100): 'cs',
    (0, 80, 100): 'cs',
    (0, 0, 230): 'cs',
    (119, 11, 32): 'cs',
    }
    
    CLASSES = { (220,20,60):'person',
                (255,0,0): 'rider',
                (0,0,142):'car',
                (0,0,70):'truck',
                (0,60,100):'bus',
                (0,80,100):'train',
                (0,0,230):'motorcycle',
                (119,11,32):'bicycle',
                (70,130,180):'sky',
                (152,251,152):'terrain',
                (107,142, 35):'vegetation',
                (220,220,  0):'traffic sign',
                (250,170, 30):'traffic light',
                (153,153,153):'pole',
                (190,153,153):'fence',
                (102,102,156):'wall',
                (70, 70, 70):'building',
                (244, 35,232):'sidewalk',
                (128, 64,128):'road'
                }
    
    
    @staticmethod
    def num_classes():
        return len(gtav_city.CLASSES.keys())
    
    def __init__(self, train, opts, split='test'):
        self.opts = opts
        self.train = train
        self.split = split
    
        # A
        if self.train or split=='val':
            self.A_anno = glob.glob(os.path.join(os.sep, 'sinergia', 'ozaydin', 'gta5', 'labels', '*.png'))
            self.A = [anno.replace('labels', 'images').replace('ozaydin', 'cvpr_dunit') for anno in self.A_anno]
            self.B_anno = glob.glob(os.path.join(os.sep, 'sinergia', 'cvpr_dunit', 'cityscapes', 'gtFine', 'train', '**', '*color.png'))
            self.B = [os.path.join(os.sep, *anno.split('/')[:4], anno.split('/')[-1].replace('gtFine_color', 'leftImg8bit')) for anno in self.B_anno]
        else:
            self.A_anno = glob.glob(os.path.join(os.sep, 'sinergia', 'ozaydin', 'gta5', 'labels', 'test', '*.png'))
            self.A = [anno.replace('labels', 'images').replace('ozaydin', 'cvpr_dunit') for anno in self.A_anno]
            self.B_anno = glob.glob(os.path.join(os.sep, 'sinergia', 'cvpr_dunit', 'cityscapes', 'gtFine', 'val', '**', '*color.png'))
            self.B = [os.path.join(os.sep, 'sinergia', 'ozaydin', 'cityscapes', 'leftImg8bit', *anno.split('/')[-3:-1], anno.split('/')[-1].replace('gtFine_color', 'leftImg8bit')) for anno in self.B_anno]

        # B
        
        #len(gta_train)=22966,len(cityscape_train)=2975
        #len(gta_test)=2000,len(cityscape_test)=500
        

        self.A_size = len(self.A)
        self.B_size = len(self.B)
        self.dataset_size = max(self.A_size, self.B_size) if self.train else self.B_size
        self.dataset_size = min(self.A_size, self.B_size) if split=='val' else self.B_size
        #when doing evaluation(for sth like beamsearch), we use just part of the train set(dependent on the min size of the two datasets)
        
        # print(self.dataset_size,self.A_size,self.B_size)


        self.classes = gtav_city.CLASSES

    def __getitem__(self, index):
        try:
            
            if self.train:
                idx_a = index if self.dataset_size == self.A_size else random.randint(0, self.A_size - 1)
                idx_b = index if self.dataset_size == self.B_size else random.randint(0, self.B_size - 1)
            #during val and test, we remove randomness by uniformly sampling the dataset
            elif self.split=='test':
                idx_a = index*4  #gta test set is 4 times larger than cityscape test set
                idx_b = index
            else:
                idx_a = index*7  #gta train set is about 7.7 times larger than cityscape train set
                idx_b = index
            
            data_A, anno_A = self.load_img(self.A[idx_a], self.A_anno[idx_a])
            data_B, anno_B = self.load_img(self.B[idx_b], self.B_anno[idx_b])
        except(IOError, OSError, IndexError, AssertionError) as e:
            #print(e)
            return self.__getitem__(random.randrange(0, self.dataset_size-1))
        return data_A, anno_A, data_B, anno_B
  

    def load_img(self, img_name, anno_name):
        img = Image.open(img_name).convert('RGB')
        anno = Image.open(anno_name).convert('RGB')
        assert img.size==anno.size#, 'img.size={}, anno.size={}'.format(img.size, anno.size)
        img, anno = self.transform(img, anno, self.opts)

        masks = torch.zeros((len(self.classes)+1, *anno.shape[-2:]), dtype=torch.uint8)
        for i, clas in enumerate(self.classes.keys()):
            mask = (anno*255).to(torch.uint8)==torch.tensor(clas).unsqueeze(1).unsqueeze(2)
            mask = (mask.sum(0)==3).to(torch.uint8)
            masks[i,] = mask
        masks[-1,] = (masks.sum(0)==0).to(torch.uint8)
        return img, masks
    
    def __len__(self):
        return self.dataset_size
  
    def transform(self, input, target, opts):
        input = F.resize(input, opts['resize_size'], transforms.InterpolationMode.BICUBIC)
        target = F.resize(target, opts['resize_size'], transforms.InterpolationMode.NEAREST)

        if self.split == 'val':
            center_crop = CenterCrop((opts['crop_size'], opts['crop_size']))
            input = center_crop(input)
            target = center_crop(target)
            # print(input,target)
        else:
            if self.train:
                i, j, h, w = transforms.RandomCrop.get_params(input, (opts['crop_size'], opts['crop_size']))
            else:
                i, j, h, w = 0, 0, 7.0/8 * opts['crop_size'], 14.0/8 * opts['crop_size']
            input = F.crop(input, i, j, h, w)
            target = F.crop(target, i, j, h, w)

        ##do random flip only during training
        if random.random() > 0.5 and self.train and self.split != 'val':
            input = F.hflip(input)
            target = F.hflip(target)
        input = F.to_tensor(input)
        input = F.normalize(input, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        target = F.to_tensor(target)
        return input, target
    
    def get_image(self, index, domain='A'):
        
        if domain == 'A':
            data, anno = self.load_img(self.A[index], self.A_anno[index])
        elif domain == 'B':
            data, anno = self.load_img(self.B[index], self.B_anno[index])
            
        return data, anno
    
    def get_random_image(self, domain='A'):
        idx = random.randint(0, self.A_size - 1) if domain == 'A' \
         else random.randint(0, self.B_size - 1)
        return self.get_image(idx, domain)

class shapes3d(data.Dataset):
    
    _FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                         'orientation']
    _NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 
                              'scale': 8, 'shape': 4, 'orientation': 15}
    
    def __init__(self, auto=True):
        # load dataset
        dataset = h5py.File('../../3dshapes.h5', 'r')
        self.images = dataset['images']  # array shape [480000,64,64,3], uint8 in range(256)
        self.labels = dataset['labels']  # array shape [480000,6], float64
        #image_shape = images.shape[1:]  # [64,64,3]
        #label_shape = labels.shape[1:]  # [6]
        self.n_samples = self.labels.shape[0]  # 10*10*10*8*4*15=480000
        self.auto = auto
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        rnd_index = random.randint(0, self.n_samples - 1)
        factor1, factor2 = self.get_factor(index), self.get_factor(rnd_index)
        if not self.auto:
            factor1[0] = 0
            factor1[1] = 3
            factor2[0] = 5
            factor2[1] = 8
        idx1, idx2 = self.get_index(factor1), self.get_index(factor2)
        image_A = self.images[idx1]
        label_A = self.labels[idx1]
        image_B = self.images[idx2]
        label_B = self.labels[idx2]

        image_A = self.transform(image_A)
        image_B = self.transform(image_B)
        return image_A, label_A, image_B, label_B
            
    
    def transform(self, image):
        image = image / 255. # normalise values to range [0,1]
        image = image.astype(np.float32)
        im_tensor = torch.from_numpy(image)
        im_tensor = im_tensor.permute(2,0,1).unsqueeze(0)
        im_tensor = torch.nn.functional.interpolate(im_tensor, scale_factor=2).squeeze(0)
        im_tensor = im_tensor * 2.0 - 1.0
        return im_tensor
    
    def get_factor(self, index):
        """ Converts indices to factors np array shape [6,batch_size]
        Args:
          index:   int
                   indices[i] in range(num_data)

        Returns:
          factors: np array shape [6,1].
                   factors[i]=factors[i,:] takes integer values in 
                   range(_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[i]]).
        """
        factors = np.zeros(6)
        base = self.n_samples
        for factor, name in enumerate(self._FACTORS_IN_ORDER):
            base /= self._NUM_VALUES_PER_FACTOR[name]
            factors[factor] = index // base
            index %= base
        return factors
    # methods for sampling unconditionally/conditionally on a given factor
    def get_index(self, factors):
        """ Converts factors to indices in range(num_data)
        Args:
          factors: np array shape [6,batch_size].
                   factors[i]=factors[i,:] takes integer values in 
                   range(_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[i]]).

        Returns:
          indices: np array shape [batch_size].
        """
        indices = 0
        base = 1
        for factor, name in reversed(list(enumerate(self._FACTORS_IN_ORDER))):
            indices += factors[factor] * base
            base *= self._NUM_VALUES_PER_FACTOR[name]
        return int(indices)

    def sample_random_batch(self, batch_size):
        """ Samples a random batch of images.
            Args:
                batch_size: number of images to sample.

            Returns:
                batch: images shape [batch_size,64,64,3].
        """
        indices = np.random.choice(self.n_samples, batch_size)
        ims = []
        for ind in indices:
            im = self.images[ind]
            im = np.asarray(im)
            ims.append(im)
        ims = np.stack(ims, axis=0)
        ims = ims / 255. # normalise values to range [0,1]
        ims = ims.astype(np.float32)
        return ims.reshape([batch_size, 64, 64, 3])

    def sample_batch(self, batch_size, fixed_factor, fixed_factor_value):
        """ Samples a batch of images with fixed_factor=fixed_factor_value, but with
            the other factors varying randomly.
        Args:
            batch_size: number of images to sample.
            fixed_factor: index of factor that is fixed in range(6).
            fixed_factor_value: integer value of factor that is fixed 
            in range(_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[fixed_factor]]).

        Returns:
            batch: images shape [batch_size,64,64,3]
        """
        factors = np.zeros([len(self._FACTORS_IN_ORDER), batch_size],
                           dtype=np.int32)
        for factor, name in enumerate(self._FACTORS_IN_ORDER):
            num_choices = self._NUM_VALUES_PER_FACTOR[name]
            factors[factor] = np.random.choice(num_choices, batch_size)
        factors[fixed_factor] = fixed_factor_value
        indices = get_index(factors)
        ims = []
        for ind in indices:
            im = images[ind]
            im = np.asarray(im)
            ims.append(im)
        ims = np.stack(ims, axis=0)
        ims = ims / 255. # normalise values to range [0,1]
        ims = ims.astype(np.float32)
        return ims.reshape([batch_size, 64, 64, 3])

class init_dataset(data.Dataset):
    '''# 1
    /video_data_20180804/jig_KYT01/20180723_KYT/dw_2018_07_23_15-46-55_000000_20fps_bae/camera_2_center_fov60.h264/15_00404.png
    3 1208 1920 1
    0
    4
    1 745.1 620.9 955.2 719.8
    1 960.5 648.3 1026.6 706.7
    1 1093.0 661.8 1125.5 692.6
    1 1081.3 659.5 1104.3 688.9

    # index
    # img path
    # img size: channel height width num
    # 0
    # number of bounding boxes
    # category_label1 x1 y1 x2 y2 (For categories: 1-Car; 2-Speed Limit Sign; 3-Person)
    # category_label2 x1 y1 x2 y2
    # category_label3 x1 y1 x2 y2
    # category_label4 x1 y1 x2 y2
    '''
    
    CLASSES = { (255,0,0):'car',
                (0,255,0): 'rider',
                (0,0,255):'person',
                }
    @staticmethod
    def num_classes():
        return len(init_dataset.CLASSES.keys())
    def __init__(self, train, opts):
        self.train = train
        self.opts = opts
        pathA = os.path.join(os.sep, 'sinergia', 'ozaydin', 'init', 'batch1_anno', 'sunny1.txt')
        pathB = os.path.join(os.sep, 'sinergia', 'ozaydin', 'init', 'batch1_anno', 'night1.txt')
        self.A = self.read_data(pathA)
        self.B = self.read_data(pathB)
        self.A_size = len(self.A)
        self.B_size = len(self.B)
        self.dataset_size = max(self.A_size, self.B_size)
        # A = self.A
        # print(len(A), A[0]['path'], A[0]['num_boxes'], A[0]['boxes']['classes'], A[0]['boxes']['coords'])
    
    def read_data(self, path):
        root = '/sinergia/ozaydin/init/'
        folder = os.path.splitext(os.path.basename(path))[0][:-1]
        data = []
        with open(path) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if '#' in line:
                    new_data = {}
                    new_data['idx'] = int(line.split()[-1])
                    new_data['path'] = root + folder + f.readline()[:-1]
                    f.readline();f.readline();
                    new_data['num_boxes'] = int(f.readline())
                    
                    new_data['boxes'] = {'classes': [], 'coords': []}
                    for i in range(new_data['num_boxes']):
                        line = f.readline().split()
                        new_data['boxes']['classes'].append(int(line[0]))
                        new_data['boxes']['coords'].append(tuple(float(c) for c in line[1:]))
                    data.append(new_data)
        return data
        
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, index):
        try:
            idx_a = index if self.dataset_size == self.A_size else random.randint(0, self.A_size - 1)
            idx_b = index if self.dataset_size == self.B_size else random.randint(0, self.B_size - 1)
            # unpack path and coords
            path_a = self.A[idx_a]['path']
            path_b = self.B[idx_b]['path']
            coords_a = torch.tensor(self.A[idx_a]['boxes']['coords'])
            coords_b = torch.tensor(self.B[idx_b]['boxes']['coords'])
            cls_a = torch.tensor(self.A[index]['boxes']['classes'])
            cls_b = torch.tensor(self.B[index]['boxes']['classes'])
            data_A, anno_A = self.load_img(path_a, cls_a, coords_a)
            data_B, anno_B = self.load_img(path_b, cls_b, coords_b)
        except(IOError, OSError, IndexError, AssertionError) as e:
            #print(e)
            return self.__getitem__(random.randrange(0, self.dataset_size-1))
        return data_A, anno_A, data_B, anno_B
  

    def load_img(self, path, clss, coords):
        img = Image.open(path).convert('RGB')
        masks = self.make_mask(img, clss, coords)
        t_img, t_masks = self.transform(img, masks, self.opts)
        
        return t_img, t_masks
    
    def make_mask(self, img, clss, box):
        w, h = img.size
        mask1 = Image.new('L', (w, h), 0)
        mask2 = Image.new('L', (w, h), 0)
        mask3 = Image.new('L', (w, h), 0)

        clss_ = np.asarray(clss).astype('int')
        box_ = np.round(box / 4) * 4
        box_ = np.asarray(box_).astype('int')
        w = box_[:,2] - box_[:,0]
        h = box_[:,3] - box_[:,1]

        for idx in range(clss_.size):
            if clss_[idx] == 1:
                sub = Image.new('L', (w[idx], h[idx]), 255)
                mask1.paste(sub, (box_[idx, 0], box_[idx, 1]))
            elif clss_[idx] == 2:
                sub = Image.new('L', (w[idx], h[idx]), 255)
                mask2.paste(sub, (box_[idx, 0], box_[idx, 1]))
            else: # cls[idx] == 3:
                sub = Image.new('L', (w[idx], h[idx]), 255)
                mask3.paste(sub, (box_[idx, 0], box_[idx, 1]))
        mask = []
        mask.append(mask1)
        mask.append(mask2)
        mask.append(mask3)
        return mask
    
    def __len__(self):
        return self.dataset_size
  

    def transform(self, input, target, opts):
        resize_size = 360
        crop_size=360
        input = F.resize(input, resize_size, transforms.InterpolationMode.BICUBIC)
        target = [F.resize(tgt, resize_size, transforms.InterpolationMode.NEAREST) for tgt in target]
        if self.train:
            i, j, h, w = transforms.RandomCrop.get_params(input, (crop_size, crop_size))
        else:
            i, j, h, w = 0, 0, 8.0/8 * crop_size, 12.0/8 * crop_size
        input = F.crop(input, i, j, h, w)
        target = [F.crop(tgt, i, j, h, w) for tgt in target]
        if random.random() > 0.5 and self.train:
            input = F.hflip(input)
            target = [F.hflip(tgt) for tgt in target]
        input = F.to_tensor(input)
        input = F.normalize(input, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        target = [F.to_tensor(tgt) for tgt in target]
            
        target = torch.cat(target, dim=0)
        s_target = torch.sum(target, 0, keepdims=True)
        m = torch.where(s_target == 0, torch.ones(1, *target.shape[-2:]),
                        torch.zeros(1, *target.shape[-2:]))
        target = torch.cat((target, m), 0)
        
        return input, target
    '''
    def transform(self, input, coords, opts):
        W, H = input.size
        r = opts['resize_size']
        c = opts['crop_size']
        ratio = r / min(H, W)
        coords = coords * r
        input = F.resize(input, r, transforms.InterpolationMode.BICUBIC)
        if self.train:
            i, j, h, w = transforms.RandomCrop.get_params(input, (c, c))
        else:
            i, j, h, w = 0, 0, 7.0/8 * c, 14.0/8 * c
        input = F.crop(input, i, j, h, w)
        
        x_coords = coords[:,0::2] - j
        y_coords = coords[:,1::2] - i
        x_coords[x_coords>=w] = -1
        x_coords[x_coords<0] = -1
        y_coords[y_coords>=h] = -1
        y_coords[y_coords<0] = -1
        if random.random() > 0.5 and self.train:
            input = F.hflip(input)
            y_coords = w - y_coords
        coords = torch.index_select(torch.cat([x_coords, y_coords], dim=1), 1, torch.LongTensor([0,2,1,3]))
        input = F.to_tensor(input)
        input = F.normalize(input, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return input, coords
    '''
    
    def get_image(self, index, domain='A'):
        dom = self.A if domain == 'A' else self.B
        path = dom[index]['path']
        coords = torch.tensor(dom[index]['boxes']['coords'])
        clss = torch.tensor(dom[index]['boxes']['classes'])
        data, anno = self.load_img(path, clss, coords)
        return data, anno
    
    def get_random_image(self, domain='A'):
        idx = random.randint(0, self.A_size - 1) if domain == 'A' \
         else random.randint(0, self.B_size - 1)
        return self.get_image(idx, domain)

if __name__ == "__main__":
    from utils import get_config
    config_path = 'configs/synthia2cityscape_folder.yaml'
    config = get_config(config_path)['munit']
    dataset = init_dataset(train=False, opts=config)
    data, anno = dataset.get_image(2)
    print(data.shape, anno.shape)
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    #batch = next(iter(dataloader))

    

    

