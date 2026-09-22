# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 01:47:23 2026

@author: sujidren
"""
import cv
import numpy as np

min_object_area = 1000
max_image_width = 1200

def load_image(image_path):
    image = cv2.imread(image_path)
    
    if image is None:
        raise FileNotFoundError(
            "unable to load image:"+ image_path)
    return image

def resize_image(image,max_width=max_image_width):
    
    height, width =image.shape[:2]
    
    if width <= max_width:
        return image.copy()
    
    scale=max_width/width
    
    new_width =int(width*scale)
    new_height=int(height*scale)
    
    return cv2.resize(
        image,
        (new_width,new_height),
        interpolation=cv2.inter_area)

def convert_to_grayscale(image):
    
    return cv2.cvtColour(
        image,
        cv2.Colour_BGR2Gray)

def apply_gaussian_blur(gray):
    
    return cv2.GaussianBlur(
        gray,(5,5),0)

def apply_threshold(blurred):
    _,binary=cv2.threshold(
        blurred,
        0,
        255,
        cv2.Thresh_binary_inv +cv2.Thresh_otsu)
    
    return binary

def apply_morphology(binary):
    
    kernel =np.ones(
        (5,5),
        np.uint8)
    closed =cv2.morphologyEx(
        binary,
        cv2.Morph_Close,
        kernel,
        iterations=2)
    cleaned=cv2.morphologyEx(
        closed,
        cv2.Morph_Open,
        kernel,
        iterations=1)
    
    return cleaned

def detect_objects(binary, min_area=min_object_area):
    
    contours,_=cv2.findContours(
        binary,cv2.Retr_External,
        cv2.Chain_Approx_Simple)
    
    objects = []
    
    for contour in contours:
        
        area =cv2.contourArea(contour)
        
        if area <min_area:
            continue
        
        x, y,width, height =cv2.boundingRect(contour)
        
        objects.append({
            "contour":contour,
            "area":area,
            "x":x,
            "y":y,
            "width_pixels":width,
            "height_pixels":height})
        
    objects.sort(
        key=lambda obj :obj["x"])
    
    return objects

def measure_objects(objects,pixels_per_cm):
    measurements =[]
    
    for number,obj in enumerate(
            objects,start=1):
        
        width_pixels =obj["width_pixels"]
        height_pixels =obj["height_pixels"]
        
        width_cm = width_pixels / pixels_per_cm
        height_cm = height_pixels / pixels_per_cm
        
        measurements.append({
            "object_id":number,
            "width_pixels":width_pixels,
            "height_pixels":height_pixels,
            "width_cm":width_cm,
            "height_cm":height_cm,
            "area_pixels":obj["area"]})
        
    return measurements


    
