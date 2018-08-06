import os
import shutil
import xml.etree.ElementTree as et
from glob import glob
import fnmatch

#tree = et.parse('rain.xml')
#root = tree.getroot()

#print(root)
curr_path = 'Annotations/'
save_dir = 'motorbike_annotations/'
curr_path_img = 'JPEGImages/'
save_dir_img = 'motorbike_images/'

def locate(pattern,root=os.curdir):
    for path, dirs, files in os.walk(os.path.abspath(root)):
        for filename in fnmatch.filter(files,pattern):
            yield os.path.join(path,filename)

saari = []

for files in locate('*.xml', r'C:\Users\sjb5cob\Desktop\smartcity\VOC\VOCdevkit\VOC2012\Annotations'):
	saari.append(files)
##print(saari)
#print(
bhandi = next(os.walk(r"C:\Users\sjb5cob\Desktop\smartcity\VOC\VOCdevkit\VOC2012\Annotations"))[2]
kaash = saari[0]

motu = []
chotu = []
for i in range(len(saari)):
	hagga = saari[i]
	tree = et.parse(hagga)

	root = tree.getroot()
	motu.append(root)
kaash = []

for j in range(len(motu)):
	
	kaash.append(bhandi[j])
			

for i in range(len(motu)):
	for child in motu[i]:
		
		if ((child.tag == 'folder')) :
			child.text = []
			child.text = 'Anything'
for k in range(len(bhandi)):
	output_path = os.path.join(r"C:\Users\sjb5cob\Desktop\smartcity\VOC\VOCdevkit\VOC2012\Annotations1",  bhandi[k])
	tree.write(output_path)
		
			
