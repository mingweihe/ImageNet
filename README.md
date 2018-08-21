Trial on kaggle imagenet object localization by yolov3 in google cloud
project:https://www.kaggle.com/c/imagenet-object-localization-challenge
yolo:https://pjreddie.com/darknet/yolo/
nice yolo explanation:https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088
My operation system is Mac OS, although it doesn't matter much, because we do most steps on google cloud.
Let's do it step by step:
1.basic environment preparation:
I.Apply a google cloud account, we can have a $300 credit acount if it's out first time to do it.
II.Create a ubuntu 16.04 compute engine instance on google cloud, with 500G SSD disk, 4 cores' cpu and, 15G memories. 
we will change it a little bit later coz GPU/CPU extention or other reasons, but now it's enough.
III.Apply quotas increasing on Nvidia tesla K80/P100/V100, coz we don't have permission to use gpu default. 
GPUs cost credit so fast, so we can choose it by needed. For me, I just increase 1 K80s for test, 4 P100s for training our model, haven't tried on V100 yet.
IV.SSH connection by RSA keys.
#:ssh-keygen -t rsa -f ~/.ssh/gc_rsa -C anynamehere
No pass word is easy for login.
#:cd ~/.ssh
#:vi gc_rsa.pub
then go to google cloud, copy everything in gc_rsa.pub to ubuntu instance SSH key part.
#:chmod 400 gc_rsa
#:ssh -i gc_rsa anynamehere@your google cloud external ip
we can also connect by 'FileZilla', no more words here.
V.pip installation
#:sudo apt-get -y install python-pip
#:sudo apt-get -y install python3-pip
VI.kaggle-cli installation
#:pip install kaggle-cli
2.dataset download
#:kg download -u <username> -p <password> -c imagenet-object-localization-challenge
// dataset is about 160G, so it will cost about 1 hour if your instance download speed is around 42.9 MiB/s.
// let's open another ssh connection, same terminal command, with (step 1->IV->last command)
3.opencv-3.4.0 installation(we will turn on opencv option in yolo project later for better image processing)
execute all the steps in the following url.
http://www.python36.com/how-to-install-opencv340-on-ubuntu1604/
4.cuda 9.0 with cudnn 7.0 installation
we can use the fallowing bash script, download it and execute it in instance.
https://gist.github.com/ashokpant/5c4e9481615f54af4025ab2085f85869#file-cuda_9-0_cudnn_7-0-sh
5.cudnn library configuration
go to https://developer.nvidia.com/rdp/cudnn-download to download cuDNN v7.0.5 Library for Linux CUDA 9.0
it's name should be cudnn-9.0-linux-x64-v7.tgz, we use scp command or filezilla to move this package from local machine to remote instance.
#:scp -i ~/.ssh/gc_rsa Downloads/cudnn-9.0-linux-x64-v7.tgz anynamehere@your google cloud external ip:~/
// come to instance window
#:tar zxvf cudnn-9.0-linux-x64-v7.tgz
#:cd cuda
#:sudo cp include/* /usr/local/cuda-9.0/include/
#:sudo cp lib64/* /usr/local/cuda-9.0/lib64/
#:echo 'export PATH=/usr/local/cuda-9.0/bin:$PATH' >> ~/.bashrc
#:echo 'export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64/:$LD_LIBRARY_PATH' >> ~/.bashrc
#:source ~/.bashrc
6.Add 1 piece of K80 GPU we applied before on our instance when it's power off, then start it.
Maybe the external ip changed after restart, but way to connect it is same as before.
7.Yolo installation
#:git clone https://github.com/pjreddie/darknet
#:cd darknet
#:make
8.X11 installatoin both of instance and our local machine, so that we can see our predicted image remotely.
#:sudo apt-get install xorg openbox
// what I need on my mac is XQuartz.
// install feh, so that we can see any picture remotely.
#:sudo apt install feh
// test it
#:feh predictions.jpg
9.test yolov3
// Actually we've done a good job until now, but we still can't see expected result if we won't change Makefile a little bit,
// I haven't figured out the reason, although let's just change it now. 
#:sed -i 's/CUDNN=1/CUDNN=0/g' Makefile
#:make
#:wget https://pjreddie.com/media/files/yolov3.weights
#:./darknet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights data/dog.jpg
10.Now let's come to the main part - train yolo on kaggle imagenet object localization
I.training data preprocessing.
#:cd ~
#:tar zxvf imagenet_object_localization.tar.gz
#:unzip LOC_synset_mapping.txt.zip
#:mkdir ILSVRC/Data/CLS-LOC/train/images
#:mv ILSVRC/Data/CLS-LOC/train/n* ILSVRC/Data/CLS-LOC/train/images/
#:mv ILSVRC/Data/CLS-LOC/val/ ILSVRC/Data/CLS-LOC/images
#:mkdir ILSVRC/Data/CLS-LOC/val/
#:mv ILSVRC/Data/CLS-LOC/images ILSVRC/Data/CLS-LOC/val/images
#:git pull https://github.com/mingweihe/ImageNet
#:pip3 install pandas
#:pip3 install pathlib
#:cd ImageNet
#:python3 generate_labels.py ../LOC_synset_mapping.txt ../ILSVRC/Annotations/CLS-LOC/train ../ILSVRC/Data/CLS-LOC/train/labels 1
#:python3 generate_labels.py ../LOC_synset_mapping.txt ../ILSVRC/Annotations/CLS-LOC/val ../ILSVRC/Data/CLS-LOC/val/labels 0
#:cd ~
#:find `pwd`/ILSVRC/Data/CLS-LOC/train/labels/ -name \*.txt > darknet/data/inet.train.list
#:sed -i 's/\.txt/\.JPEG/g' darknet/data/inet.train.list
#:sed -i 's/labels/images/g' darknet/data/inet.train.list

#:find `pwd`/ILSVRC/Data/CLS-LOC/val/labels/ -name \*.txt > darknet/data/inet.val.list
#:sed -i 's/\.txt/\.JPEG/g' darknet/data/inet.val.list
#:sed -i 's/labels/images/g' darknet/data/inet.val.list
II.pretrained weights preparation.
#:cd darknet
#:wget https://pjreddie.com/media/files/darknet53.conv.74
III.cfg files preparation
#:cp ~/ImageNet/yolov3-ILSVRC.cfg cfg/
#:cp ~/ImageNet/ILSVRC.data cfg/
IV.Traning
#:./darknet detector train cfg/ILSVRC.data cfg/yolov3-ILSVRC.cfg darknet53.conv.74
// we can also restart training from a checkpoint:
#:./darknet detector train cfg/ILSVRC.data cfg/yolov3-ILSVRC.cfg backup/yolov3.backup
V.Traininguse with multiple GPUs
// shutdown instance, configure GPU from 1 piece of K80 to 4 piece of P100, with 8 CPUs.
// boot instance, start training using following command
#:./darknet detector train cfg/ILSVRC.data cfg/yolov3-ILSVRC.cfg darknet53.conv.74 -gpus 0,1,2,3
// continue from checkpoints we can replace darknet53.conv.74 with backup file.
VI.
VII.
VIII.
11.Prediction
#:
12.transfer predcitions to CSV file.
13.submit our predictions.
Good luck and thanks for your attention.













