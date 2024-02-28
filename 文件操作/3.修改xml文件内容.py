import os
import xml.etree.ElementTree as ET

# 定义要修改的目标文件夹路径
folder_path = r'E:\datasets\Nonvehicle\Nonvehicle\Annotations_new'

# 遍历文件夹中的所有XML文件
for filename in os.listdir(folder_path):
    if filename.endswith('.xml'):
        xml_file_path = os.path.join(folder_path, filename)

        # 解析XML文件
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # 遍历XML树中的所有<name>元素，并将其内容修改为"driver"
        for name_elem in root.iter('name'):
            name_elem.text = 'driver'

        # 保存修改后的XML文件
        tree.write(xml_file_path)

        print(f"已修改文件：{filename}")
