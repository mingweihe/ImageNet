import warnings
warnings.filterwarnings("ignore")
from xml.dom import minidom
import pandas as pd
import os
from os.path import basename
import pathlib
import sys

def xmlToTxt(xml_path, target_forder, isCreatePfolder):
    mydoc=minidom.parse(xml_path)
#     folder=mydoc.getElementsByTagName('folder')[0].firstChild.data
#     filername=mydoc.getElementsByTagName('filename')[0].firstChild.data
    folder=os.path.basename(os.path.dirname(xml_path))
    filername=os.path.splitext(basename(xml_path))[0]
    width=float(mydoc.getElementsByTagName('width')[0].firstChild.data)
    height=float(mydoc.getElementsByTagName('height')[0].firstChild.data)
    objects=mydoc.getElementsByTagName('object')
    to_save=target_forder
    if(isCreatePfolder):
        to_save = os.path.join(target_forder, folder)
    pathlib.Path(to_save).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(to_save, filername+'.txt'), 'a') as the_file:
        for obj in objects:
            name=obj.getElementsByTagName('name')[0].firstChild.data
            name_index=str(df.loc[df[0]==name].index[0])
            xmin=float(obj.getElementsByTagName('xmin')[0].firstChild.data)
            ymin=float(obj.getElementsByTagName('ymin')[0].firstChild.data)
            xmax=float(obj.getElementsByTagName('xmax')[0].firstChild.data)
            ymax=float(obj.getElementsByTagName('ymax')[0].firstChild.data)
            w=xmax-xmin
            h=ymax-ymin
            x=xmax-w/2
            y=ymin+h/2
            the_file.write(name_index+' '+str(round(x/width, 5))+
            ' '+str(round(y/height, 5))+' '+str(round(w/width,5))+
            ' '+str(round(h/height,5))+'\n')

if __name__=='__main__':
    if(len(sys.argv)<5):
        print('Usage: python', sys.argv[0], '[map_file]', '[xml_folder]', '[output_folder]', '[isCreatePfolder]')
        sys.exit(1)

    map_file=sys.argv[1]
    xml_folder=sys.argv[2]
    output_folder=sys.argv[3]
    isCreatePfolder=int(sys.argv[4])

    one_lalel=pd.DataFrame
    # map_file='../../backup/LOC_synset_mapping.txt'
    df=pd.read_csv(map_file, sep='\t', header=None)
    df[1]=df[0].str[10:]
    df[0]=df[0].str[0:9]

    # xml_folder='../../backup/ILSVRC/Annotations/CLS-LOC/train'
    # output_folder='labels'
    for target, dirs, files in os.walk(xml_folder):
        for file in files:
            if file.endswith(".xml"):
                xmlToTxt(os.path.join(target, file), output_folder, isCreatePfolder)