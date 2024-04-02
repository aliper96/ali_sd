import glob
import os
import pickle

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
def load_latents(latent_path):
    r"""
    Simple utility to save latents to speed up ldm training
    :param latent_path:
    :return:
    """
    latent_maps = {}
    for fname in glob.glob(os.path.join(latent_path, '*.pkl')):
        s = pickle.load(open(fname, 'rb'))
        for k, v in s.items():
            latent_maps[k] = v[0]
    return latent_maps

class AssemblyDataset(Dataset):
    def __init__(self, split, im_path, mask_path, caption_path, im_size, im_channels=3, use_latents=False, latent_path=None,
                 condition_config=None):
        """
        Inicializa las propiedades del conjunto de datos.
        """
        self.split = split
        self.im_size = im_size
        self.mask_h = im_size  # Altura de la máscara, ajustar según sea necesario
        self.mask_w = im_size  # Ancho de la máscara, ajustar según sea necesario
        self.im_channels = im_channels
        self.condition_types = ['text'] if condition_config is None else condition_config['condition_types']

        self.idx_to_cls_map = {}
        self.cls_to_idx_map = {}

        if 'image' in self.condition_types:
            self.mask_channels = 1
            self.mask_h = 256
            self.mask_w = 256

        self.images, self.masks, self.captions = self.load_images(im_path, mask_path, caption_path)


        # Asegurarse de que la cantidad de imágenes y máscaras coincida si se usa la condición 'image'
        if 'image' in self.condition_types:
            assert len(self.masks) == len(self.images), "Condition Type Image but could not find masks for all images"
        if 'text' in self.condition_types:
            assert len(self.captions) == len(self.images), "Condition Type Text but could not find captions for all images"

        print(f'Found {len(self.images)} images')
        print(f'Found {len(self.masks)} masks')
        print(f'Found {len(self.captions)} captions')

        # Cargar latentes si es necesario
        self.latent_maps = None
        self.use_latents = False
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) == len(self.images):
                self.use_latents = True
                self.latent_maps = latent_maps
                print(f'Found {len(self.latent_maps)} latents')
            else:
                print('Latents not found')

    def get_mask(self, index):
        """
        Procesa la máscara para convertirla en un formato de múltiples canales,
        donde cada canal representa una clase diferente.
        """
        mask_im = Image.open(self.masks[index])
        mask_im = mask_im.resize((self.mask_h, self.mask_w), Image.NEAREST)  # Redimensionar si es necesario
        mask_im = np.array(mask_im)

        im_base = np.zeros((self.mask_h, self.mask_w, self.mask_channels))
        for orig_idx in range(self.mask_channels):
            # Asumiendo que los valores en las máscaras empiezan en 1 y corresponden al índice en idx_to_cls_map
            im_base[mask_im == (orig_idx + 1), orig_idx] = 1

        mask = torch.from_numpy(im_base).permute(2, 0, 1).float()  # Convertir a tensor y ajustar dimensiones
        return mask

    def load_images(self, im_path, mask_path, caption_path):
        """
        Carga todas las imágenes y máscaras del directorio especificado y las agrupa.
        """
        assert os.path.exists(im_path), f"Image path {im_path} does not exist"
        assert os.path.exists(mask_path), f"Mask path {mask_path} does not exist"
        assert os.path.exists(caption_path), f"Caption path {caption_path} does not exist"

        ims = []
        mks = []
        captions = []

        for im_file in tqdm(glob.glob(os.path.join(im_path, '*.jpg'))):  # Asume extensión .jpg para imágenes
            base_name = os.path.basename(im_file).split('.')[0]
            mask_file = os.path.join(mask_path, f'{base_name}.png')  # Asume extensión .png para máscaras
            caption_file = os.path.join(caption_path, f'{base_name}.txt')  # Asume extensión .txt para captions
            #read captions line by line
            with open(caption_file, 'r') as f:
                caption = f.readline()
            # Asegurar que captions no esté vacío y al menos hay una linea
            assert caption, f"Caption not found for {im_file}"


            if os.path.exists(mask_file):
                ims.append(im_file)
                mks.append(mask_file)
                captions.append(caption)

        return ims, mks, captions

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Obtiene un elemento del conjunto de datos.
        """
        im = Image.open(self.images[index]).convert('L')
        transform = transforms.Compose([
            transforms.Resize(self.im_size),  # Resize the image to the desired size
            transforms.ToTensor(),  # Convert the image to a tensor
        ])
        im_tensor = transform(im)
        im.close()
        # im_tensor = (2 * im_tensor) - 1

        cond_inputs = {}
        if 'image' in self.condition_types:
            mask = self.get_mask(index)  # Usar el método get_mask para obtener la máscara procesada
            cond_inputs['image'] = mask
        if 'text' in self.condition_types:
            cond_inputs['text'] = self.captions[index]

        if self.use_latents:
            latent = self.latent_maps[self.images[index]]
            return latent, cond_inputs if len(self.condition_types) > 0 else latent
        else:
            im_tensor = (2 * im_tensor) - 1  # Normalizar a [-1, 1]

            return im_tensor, cond_inputs if len(self.condition_types) > 0 else im_tensor
