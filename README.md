# Traditional Inspired Network
This repository contains the implementation details of our paper:

"**[TRADITIONAL METHOD INSPIRED DEEP NEURAL NETWORK FOR EDGE DETECTION](https://ieeexplore.ieee.org/document/9190982)**"  
by Jan Kristanto Wibisono , Hsueh-Ming Hang      

![image](https://github.com/jannctu/TIN/blob/master/img/result_BSDS.png)
![image](https://github.com/jannctu/TIN/blob/master/img/result_NYUD.png)

# Dependencies
* Python 3.7 
* Pytorch 1.4   

# Network Structure
Our systems contain three basic modules: Feature Extractor, Enrichment, and Summarizer, which roughly correspond to gradient, low pass filter, and pixel connection in the traditional edge detection schemes.   

![image](https://github.com/jannctu/TIN/blob/master/img/TIN1.png)

# Evaluation
![image](https://github.com/jannctu/TIN/blob/master/img/ODS.png)
Comparison of complexity and accuracy performance among various edge detection schemes. Our proposed methods (Green). BDCN family (Red). Other methods (Blue). ODS (Transparent label). Number of Parameter (Orange label)   

# Todo:

#### Testing

        python inference.py
		
#### Manual Input

        python test.py

![image](https://github.com/jannctu/TIN/blob/master/img/lenna.png)
![image](https://github.com/jannctu/TIN/blob/master/img/result_lenna.png)
![image](https://github.com/jannctu/TIN/blob/master/img/mri_brain.jpg)
![image](https://github.com/jannctu/TIN/blob/master/img/result_mri_brain.png)

# Citing 
@INPROCEEDINGS{9190982,
  author={J. K. {Wibisono} and H. -M. {Hang}},
  booktitle={2020 IEEE International Conference on Image Processing (ICIP)}, 
  title={Traditional Method Inspired Deep Neural Network For Edge Detection}, 
  year={2020},
  volume={},
  number={},
  pages={678-682},
}
