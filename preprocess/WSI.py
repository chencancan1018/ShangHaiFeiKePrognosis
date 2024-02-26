import os
import cv2
import numpy as np
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
from openslide import PROPERTY_NAME_MPP_X, PROPERTY_NAME_MPP_Y, PROPERTY_NAME_VENDOR

import utils

class WSI():
    """
    A reading whole-slide image class
    API: openslide
    """
    def __init__(self, wsi_path):
        if not os.path.exists(wsi_path):
            raise Exception(f"{wsi_path} not exists!")
        file_extension = os.path.splitext(wsi_path)[-1]
        if file_extension not in ['.tif', '.svs', '.ndpi']:
            raise Exception(f"Unsupported file extension for {wsi_path}!")
        try:
            self.slide = OpenSlide(wsi_path)
        except OpenSlideUnsupportedFormatError:
            raise OpenSlideUnsupportedFormatError()
    
    def get_level_count(self):
        return self.slide.level_count
    
    def get_level_dimensions(self, level):
        level = int(level)
        return  self.slide.level_dimensions[level]

    def read_region(self, location, size, level):
        """Return a cv2.Image containing the contents of the region.

        location: (x, y) tuple giving the top left pixel in the level 0
                  reference frame.
        level:    the level number.
        size:     (width, height) tuple giving the region size.

        Unlike in the C interface, the image data returned by this
        function is not premultiplied."""
        if len(list(location)) == 2 and len(list(size)) == 2:
            loc = (int(location[0]), int(location[1]))
            size = (int(size[0]), int(size[1]))
            level = int(level)
            wsi_pil = self.slide.read_region(loc, level, size)
            wsi_cv = utils.rgba2bgr(wsi_pil)
            return wsi_cv
        else:
            raise TypeError('The type of location is not int!')
    
    def get_wsi_data(self, level):
        level = int(level)
        if level < self.slide.level_count - 1:
            width, height = self.slide.level_dimensions[level]
            wsi_pil = self.slide.read_region((0,0), level, (width, height))
            wsi_cv = utils.rgba2bgr(wsi_pil)
            return wsi_cv
        else:
            raise Exception(f"The level {level} is beyond all wsi levels!") 

    def get_resolution(self): # (0.092045000000000002, 0.092045000000000002)
        mpp = [0, 0]
        if PROPERTY_NAME_MPP_X in self.slide.properties and PROPERTY_NAME_MPP_Y in self.slide.properties:
            mpp = [self.slide.properties[PROPERTY_NAME_MPP_X], self.slide.properties[PROPERTY_NAME_MPP_Y]]
        elif PROPERTY_NAME_VENDOR in self._slide.properties and self.slide.properties[PROPERTY_NAME_VENDOR] == 'generic-tiff':
            if self.slide.properties['tiff.ResolutionUnit'] == 'centimeter':
                mpp = [10000/float(self.slide.properties['tiff.XResolution']), 10000/float(self.slide.properties['tiff.YResolution'])]
            elif self.slide.properties['tiff.ResolutionUnit'] == 'inch':
                mpp = [2.54*10000/float(self.slide.properties['tiff.XResolution']), 2.54*10000/float(self.slide.properties['tiff.YResolution'])]
        return mpp
    
    def get_propertise(self):
        if not self.slide:
            raise OpenSlideUnsupportedFormatError()            
        result = {}
        for key in self.slide.properties.keys():
            value = self.slide.properties.get(key)
            result[key] = value
        return result
