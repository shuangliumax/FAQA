import os
import xml.etree.ElementTree as ET

# 要修改的XML文件所在的文件夹路径
xml_folder = '/home/shuangliumax/dataset/gao/VOC2012/Annotations/'

# 包含XML文件名称的txt文件路径
xml_names_txt = '/home/shuangliumax/dataset/gao/VOC2012/ImageSets/Main/val.txt'

# 读取txt文件中的XML文件名称列表
with open(xml_names_txt, 'r') as f:
    xml_names = f.read().splitlines()

# 遍历每个XML文件，修改<Difficult>标签为<difficult>
for xml_name in xml_names:
    xml_path = os.path.join(xml_folder, xml_name + '.xml')

    if os.path.isfile(xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall('object'):
            difficult_elem = obj.find('Difficult')
            if difficult_elem is not None:
                difficult_elem.tag = 'difficult'

        tree.write(xml_path)

    else:
        print(f"XML file {xml_name}.xml not found.")

print("Conversion completed.")
