"""
@Name: 1.批量修改名称.py
@Auth: Huohuo
@Date: 2023/4/14-14:21
@Desc: 
@Ver : 
code_idea
"""
import os

# path = r"D:\WorkDirectory\mywork\yolov5-master\datasets\VOC2007\img_label_xml/"
# path = r"D:\WorkDirectory\mywork\myself_OCR\paper_used_data\train\img_en/"
# path = r"D:\WorkDirectory\mywork\myself_OCR\paper_used_data\train\txt_en/"
path = r"D:\WorkDirectory\mywork\myself_OCR\paper_used_data\test\txt_en/"


for filename in os.listdir(path):
	print(filename)
	newname = filename.replace('.txt', '_msr.txt')
	# newname = filename.replace('.jpg', '_msr.jpg')
	os.rename(path+filename, path+newname)
