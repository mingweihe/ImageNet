<h1>Trial on kaggle imagenet object localization by yolov3 in google cloud</h1>
project:https://www.kaggle.com/c/imagenet-object-localization-challenge<br>
yolo:https://pjreddie.com/darknet/yolo/<br>
nice yolo explanation:https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088<br>
My operation system is Mac OS, although it doesn't matter much, because we do most steps on google cloud.<br>
Let's do it step by step:<br>
<h2>1.Basic environment preparation:</h2>
<h4>I.Apply a google cloud account.</h4>
ps:Google provide $300 credit trial time for first time sign up account.
<h4>II.Create a ubuntu 16.04 compute engine instance on google cloud, with 400G SSD disk, 4 cores' cpu and, 15G memories. </h4>
we will change it a little bit later coz GPU/CPU extention or other reasons, but now it's enough.<br>
<h4>III.Apply quotas increasing on Nvidia tesla K80/P100/V100, coz we don't have permission to use gpu default. </h4>
GPUs cost credit so fast, so we can choose it by needed. For me, I just increase 1 K80s for test, 4 P100s for training our model, haven't tried on V100 yet.<br>
<h4>IV.SSH connection by RSA keys.</h4>
&#35;:ssh-keygen -t rsa -f ~/.ssh/gc_rsa -C anynamehere<br>
No pass word is easy for login.<br>
&#35;:cd ~/.ssh<br>
&#35;:vi gc_rsa.pub<br>
then go to google cloud, copy everything in gc_rsa.pub to ubuntu instance SSH key part.<br>
&#35;:chmod 400 gc_rsa<br>
&#35;:ssh -i gc_rsa anynamehere@your google cloud external ip<br>
we can also connect by 'FileZilla', no more words here.<br>
<h4>V.pip installation</h4>
&#35;:sudo apt update<br>
&#35;:sudo apt upgrade<br>
&#35;:sudo apt-get -y install python-pip<br>
&#35;:sudo apt-get -y install python3-pip<br>
<h4>VI.kaggle-cli installation</h4>
&#35;:pip install kaggle-cli<br>
<h2>2.Dataset download</h2>
&#35;:kg download -u &lt;your kaggle username&gt; -p &lt;your kaggle password&gt; -c imagenet-object-localization-challenge<br>
// dataset is about 160G, so it will cost about 1 hour if your instance download speed is around 42.9 MiB/s.<br>
// let's open another ssh connection to do next step when it's doing the download process.<br>
<h2>3.Opencv-3.4.0 installation(we will turn on opencv option in yolo project later for better image processing)</h2>
execute all the steps in the following url.<br>
http://www.python36.com/how-to-install-opencv340-on-ubuntu1604/<br>
<h2>4.Cuda 9.0 with cudnn 7.0 installation</h2>
we can use the fallowing bash script, download it and execute it in instance.<br>
https://gist.github.com/ashokpant/5c4e9481615f54af4025ab2085f85869#file-cuda_9-0_cudnn_7-0-sh<br>
<h2>5.Cudnn library configuration</h2>
go to https://developer.nvidia.com/rdp/cudnn-download to download cuDNN v7.0.5 Library for Linux CUDA 9.0<br>
it's name should be cudnn-9.0-linux-x64-v7.tgz, we use scp command or filezilla to move this package from local machine to remote instance.<br>
&#35;:scp -i ~/.ssh/gc_rsa Downloads/cudnn-9.0-linux-x64-v7.tgz anynamehere@your google cloud external ip:~/<br>
// come to instance window<br>
&#35;:tar zxvf cudnn-9.0-linux-x64-v7.tgz<br>
&#35;:cd cuda<br>
&#35;:sudo cp include/* /usr/local/cuda-9.0/include/<br>
&#35;:sudo cp lib64/* /usr/local/cuda-9.0/lib64/<br>
&#35;:echo 'export PATH=/usr/local/cuda-9.0/bin:$PATH' >> ~/.bashrc<br>
&#35;:echo 'export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64/:$LD_LIBRARY_PATH' >> ~/.bashrc<br>
&#35;:source ~/.bashrc<br>
<h2>6.Shutdown instance, add 1 piece of K80 GPU, then boot instance again.</h2>
ps:For frugality, we can revise number of cpu cores from 4 to 2<br>
// view GPUs detailed info<br>
&#35;: nvidia-smi<br>
// view the number of CPUs<br>
&#35;:nproc<br>
<h2>7.X11 installatoin both of instance and our local machine, so that we can see our predicted image remotely.</h2>
&#35;:sudo apt-get install xorg openbox<br>
// what I need on my mac is XQuartz.<br>
// install feh, so that we can see any picture remotely.<br>
&#35;:sudo apt install feh<br>
// logout from instance, connect it with additional parameter, then test it<br>
&#35;:ssh -Y -i ~/.ssh/gc_rsa anynamehere@your google cloud external ip<br>
&#35;:feh darknet/data/dog.jpg<br>
<h2>8.Yolo installation</h2>
&#35;:git clone https://github.com/pjreddie/darknet<br>
&#35;:cd darknet<br>
&#35;:make<br>
<h2>9.Test yolov3</h2>
// Actually we've done a good job until now, but we still can't see expected result if we won't change Makefile a little bit,<br>
// I haven't figured out the reason, although let's just change it now. <br>
&#35;:cd darknet<br>
&#35;:sed -i 's/GPU=./GPU=1/' Makefile<br>
&#35;:sed -i 's/CUDNN=./CUDNN=0/' Makefile<br>
&#35;:sed -i 's/OPENCV=./OPENCV=1/' Makefile<br>
&#35;:sed -i 's/OPENMP=./OPENMP=1/' Makefile<br>
&#35;:sed -i 's/DEBUG=./DEBUG=0/' Makefile<br>
&#35;:make<br>
&#35;:wget https://pjreddie.com/media/files/yolov3.weights<br>
&#35;:./darknet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights data/dog.jpg<br>
<h2>10.Now let's train it.</h2>
<h4>I.training data preprocessing.</h4>
&#35;:cd ~<br>
&#35;:tar zxvf imagenet_object_localization.tar.gz<br>
// delete package so that we'll have enough disk space.<br>
&#35;:rm imagenet_object_localization.tar.gz<br>
// view disk space info.<br>
&#35;: df -h<br>
// Data preparation<br>
&#35;:unzip LOC_synset_mapping.txt.zip<br>
&#35;:mkdir ILSVRC/Data/CLS-LOC/train/images<br>
&#35;:mv ILSVRC/Data/CLS-LOC/train/n* ILSVRC/Data/CLS-LOC/train/images/<br>
&#35;:mv ILSVRC/Data/CLS-LOC/val/ ILSVRC/Data/CLS-LOC/images<br>
&#35;:mkdir ILSVRC/Data/CLS-LOC/val/<br>
&#35;:mv ILSVRC/Data/CLS-LOC/images ILSVRC/Data/CLS-LOC/val/images<br>
&#35;:git clone https://github.com/mingweihe/ImageNet<br>
&#35;:pip3 install pandas<br>
&#35;:pip3 install pathlib<br>
&#35;:cd ImageNet<br>
// generating all training formatted label files costs about 20 minutes<br>
&#35;:python3 generate_labels.py ../LOC_synset_mapping.txt ../ILSVRC/Annotations/CLS-LOC/train ../ILSVRC/Data/CLS-LOC/train/labels 1<br>
// generating all validation formatted label files<br>
&#35;:python3 generate_labels.py ../LOC_synset_mapping.txt ../ILSVRC/Annotations/CLS-LOC/val ../ILSVRC/Data/CLS-LOC/val/labels 0<br>
&#35;:cd ~<br>
&#35;:find `pwd`/ILSVRC/Data/CLS-LOC/train/labels/ -name \*.txt > darknet/data/inet.train.list<br>
&#35;:sed -i 's/\.txt/\.JPEG/g' darknet/data/inet.train.list<br>
&#35;:sed -i 's/labels/images/g' darknet/data/inet.train.list<br>
&#35;:find `pwd`/ILSVRC/Data/CLS-LOC/val/labels/ -name \*.txt > darknet/data/inet.val.list<br>
&#35;:sed -i 's/\.txt/\.JPEG/g' darknet/data/inet.val.list<br>
&#35;:sed -i 's/labels/images/g' darknet/data/inet.val.list<br>
<h4>II.pretrained weights preparation.</h4>
&#35;:cd darknet<br>
&#35;:wget https://pjreddie.com/media/files/darknet53.conv.74<br>
<h4>III.Traning</h4>
&#35;:./darknet detector train ~/ImageNet/ILSVRC.data ~/ImageNet/yolov3-ILSVRC.cfg darknet53.conv.74<br>
// we can also restart training from a checkpoint:<br>
&#35;:./darknet detector train ~/ImageNet/ILSVRC.data ~/ImageNet/yolov3-ILSVRC.cfg backup/yolov3-ILSVRC.backup<br>
<h4>IV.Training with multiple GPUs</h4>
// shutdown instance, increase number of GPUs from 1 piece's K80 to 4 pieces' P100, with 6 CPUs.<br>
// boot instance, start training using following command<br>
&#35;:./darknet detector train ~/ImageNet/ILSVRC.data ~/ImageNet/yolov3-ILSVRC.cfg backup/yolov3-ILSVRC.backup -gpus 0,1,2,3<br>
// continue from checkpoints we can replace darknet53.conv.74 with backup file.<br>
<h4>V.Training without ssh connection</h4>
&#35;:screen<br>
&#35;:./darknet detector train ~/ImageNet/ILSVRC.data ~/ImageNet/yolov3-ILSVRC.cfg backup/yolov3-ILSVRC.backup -gpus 0,1,2,3<br>
// press Keys of &lt;Ctrl+a&gt;<br>
// then press &lt;d&gt; key<br>
// now we have our task done detached from our local machine<br>
// If we wanna put task back, we can connect ssh, then:<br>
&#35;:screen -r<br>
// For more detailed instruction, just google "linux screen detach".<br>
<h2>11.Prediction and transfer predictions into CSV file.</h2>
&#35;:unzip LOC_sample_submission.csv.zip<br>
&#35;:mkdir ~/submissions<br>
&#35;:python3 ~/ImageNet/predict.py<br>
<h2>12.Submit our predictions.</h2>
&#35;kg submit &lt;submission-file&gt; -u &lt;your kaggle username&gt; -p &lt;your kaggle password&gt; -c imagenet-object-localization-challenge -m "my submission"<br>
(optional way is submitting it on kaggle website by using any web browser.)<br>
<h2>13.Accuracy improvement.</h2>
<h4>I.Cross-validation & Ensembling.</h4>
TODO.<br>
<h4>II.Training validation dataset for a few more epochs before final submission.</h4>
Good luck, thanks for attentions.<br>