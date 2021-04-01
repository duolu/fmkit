# FMKit - A Library and Data Repository for In-Air-Handwriting Analysis

FMKit is a library and data repository for finger motion based user login and in-air-handwriting analysis. We have also built a demo system based on them. See this [introduction video](https://youtu.be/O3Jqq9yqJSE) and a longer [demo](https://www.youtube.com/watch?v=asxqpF7dH10). Also, see the [documents](https://duolu-fmkit.github.io/). Currently, the "word-210" dataset is openly available [here](https://www.thothlab.com/getgooglefile/1RXj0t8NMYt_Jr5lW-BIeAIxw4aX5VshV). If you would like to use other datasets related to in-air-handwriting of ID and passcode, please fill this [application](https://docs.google.com/document/d/1AHX3lj1mjm4ZZEZTHNdm3xDmJAAWi7P6bIdLlBNZ8PA/edit?usp=sharing) send us an email (see the "authors" section below).

[![FMKit Demo](https://img.youtube.com/vi/O3Jqq9yqJSE/0.jpg)](https://www.youtube.com/watch?v=O3Jqq9yqJSE)


## The FMKit Library

The FMKit library contains a set of Python code and scripts to model and process in-air-handwriting signals. See the "code" folder. An overview is shown as follows.

* code_core ---> The main library code. 
  * The "fmsignal" module contains the classes modeling the finger motion signal. 
  * The "fmsignal_vis" module contains functions to plot and animate the signal and the trajectory.
  * The "fmsignal_demo" module contains contains demo code.
  * The "pyrotation" and "pyrotation_demo" modules are copied from the [pyrotation](https://github.com/duolu/pyrotation) project, which is necessary for finger motion signal preprocessing.
* code_utilities ---> Utility code written in C to speed up the Dynamic Time Warping calculation.
* data_demo ---> Some example in-air-handwriting signals, collected using two devices.
* meta_demo ---> Metadata of the datasets.

The FMKit code library requires the following software packages

* Python 3 (tested with Python 3.6.9)
* NumPy (tested with NumPy 1.19.5)
* Matplotlib (tested with 3.1.2)

To use this code library, just download the Python modules under the "code_fmkit" folder and incorporate into your project.

## The Data Repository

Here is a description of the dataset we have collected to facilitate our research on in-air-handwriting. We have IRB approval of this data collection (Arizona State University STUDY00008279 and STUDY00010539). Currently, the "word-210" dataset is openly available [here](https://drive.google.com/drive/folders/1RXj0t8NMYt_Jr5lW-BIeAIxw4aX5VshV?usp=sharing). If you would like to use other datasets related to in-air-handwriting of ID and passcode, please fill this [application](https://docs.google.com/document/d/1AHX3lj1mjm4ZZEZTHNdm3xDmJAAWi7P6bIdLlBNZ8PA/edit?usp=sharing) send us an email (see the "authors" section below). 

Two devices are used (shown in the following figure): a wearable device (a custom-made data glove with inertial sensors) and a contactless 3D camera (the Leap Motion controller). The data repository contains the following five datasets.

![Device illustration.](pics/devices.png)

**(1) Sign-up and Sign-in**: We asked each participating user to create two distinct meaningful strings and write them in-the-air, one as an account ID and the other as an account passcode. Such a string may include alphanumeric letters, characters in a language other than English, or meaningful symbols such as five-pointed stars. The content of the string is determined by the user. Hence, each string can be used as either an ID or a passcode. For each string, we asked the user to write it 5 repetitions as registration and another 5 repetitions as login attempts. This simulates the normal sign-up and sign-in procedure (shown in the following figure). In total 180 users participated the data collection and 360 strings are obtained. In total, there are 360 (strings) * 10 (repetitions) * 2 (devices) = 7,200 signals.

![Sign-in illustruction.](pics/sign-in.png)

**(2) Spoofing with Semantic Leakage**: We asked 10 skilled imposters to imitate the in-air-handwriting of the ID and password generated by the users in the first dataset. In this setting, the imposters know the semantic meaning of the strings written by the legitimate users, but the imposters have not seen the legitimate users writing the ID and password in the air, which simulates spoofing attack with semantic leakage. All strings in the first dataset are spoofed and each imposter wrote every string with 5 repetitions using both two devices for data collection. In total, there are 360 (strings) * 10 (imposters) * 5 (repetitions) * 2 (devices) = 36,000 signals.

**(3) Spoofing with both Semantic and Visual Leakage**: We asked 10 skilled imposters to imitate the in-air-handwriting of the ID and password generated by the users in the first dataset. In this setting, the imposters can watch the recorded video of the in-air-handwriting and they will be informed with the semantic meaning of the in-air-handwriting. 180 strings in the first dataset are spoofed and each imposter wrote the string with 5 repetitions using both two devices for data collection. In total, there are 360 (strings) * 10 (imposters) * 5 (repetitions) * 2 (devices) = 18,000 signals.

**(4) Long-Term Persistence Study**: We kept collecting the sign-in in-air-handwriting of a subset of the users in the first dataset for a period of about 4 weeks, which simulates login activity in the long term. In the first dataset, the user wrote each string 5 repetitions as registration. In this dataset, the users wrote the strings for the account ID and the account passcode 5 repetitions consecutively as a session, and 10 sessions in total. 40 users participated in this dataset. In total there are 80 (strings) * 10 (imposters) * 5 (repetitions) * 2 (devices) = 8,000 data samples.

**(5) In-Air-Handwriting Words (word-210)**: We asked 10 users to write 210 English words and 210 Chinese words with 5 repetitions for each word with both devices. In total, there are 2 (languages) * 210 (strings) * 10 (writers) * 5 (repetitions) * 2 (devices) = 42,000 data samples.

**(6) Usability Survey**: We asked the participating users to fill a survey on the usability of gesture sign-in system with various sensors and different types of gestures. 72 users responded to the survey.



## Authors

* **Duo Lu < duolu@asu.edu >** - main contributor, current maintainer of the project.
* **Yuli Deng < ydeng19@asu.edu >** - contributor.
* **Linzhen Luo < lluo21@asu.edu >** - contributor.
* **Dijiang Huang < dijiang.huang@asu.edu >** - our academic advisor and sponsor.

## Papers

* Duo Lu, Linzhen Luo, Dijiang Huang, Yezhou Yang, "**FMKit: An In-Air-Handwriting Analysis Library and Data Repository.**" *CVPR Workshop on Computer Vision for Augmented and Virtual Reality, 2020.* [[pdf]](/papers/fmkit.pdf) [[link]](https://mixedreality.cs.cornell.edu/workshop/2020/papers#block-93cead2afaf5f6895a67) [[video]](https://youtu.be/O3Jqq9yqJSE)
* Dijiang Huang, Duo Lu, "**Three-Dimensional In-The-Air Finger Motion based User Login Framework for Gesture Interface**", *US Patent 10,877,568, 2020.* [[link]](https://patents.google.com/patent/US10877568B2/en)
* Duo Lu, Dijiang Huang, "**FMCode: A 3D In-the-Air Finger Motion Based User Login Framework for Gesture Interface.**" *arXiv preprint arXiv:1808.00130, 2018.* [[pdf]](/papers/fmcode.pdf) [[link]](https://arxiv.org/abs/1808.00130)
* Duo Lu, Dijiang Huang, "**FMHash: Deep Hashing of In-Air-Handwriting for User Identification.**" *in Proceedings of the International Conference on Communication (ICC), 2019* [[pdf]](/papers/fmhash.pdf) [[link]](https://arxiv.org/abs/1806.03574) [[slides]](/papers/fmhash_slides.pdf) [[video]](https://www.youtube.com/watch?v=MyaWe7RX8oE)
* Duo Lu, Dijiang Huang, Yuli Deng, and Adel Alshamrani. "**Multifactor user authentication with in-air-handwriting and hand geometry.**" *In 2018 International Conference on Biometrics (ICB), 2018.* [[pdf]](/papers/multifactor.pdf) [[link]](https://ieeexplore.ieee.org/document/8411230) [[slides]](/papers/multifactor_slides.pdf) [[poster]](/papers/multifactor_poster.pdf)
* Duo Lu, Kai Xu, and Dijiang Huang, "**A Data Driven In-Air-Handwriting Biometric Authentication System.**", *in Proceedings of the International Joint Conference on Biometrics (IJCB), 2017.* [[pdf]](/papers/data-driven.pdf) [[link]](https://ieeexplore.ieee.org/document/8272739) [[slides]](/papers/data-driven_slides.pdf)

## License

The code of this project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

We would like to thank all participants and volunteers who helped us collecting the data.

This project is supported by NSF CCRI award [#1925709](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1925709).
