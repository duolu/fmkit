# FMKit - A Library and Data Repository for In-Air-Handwriting Analysis

FMKit is a library and data repository for finger motion based user login and in-air-handwriting analysis. We have also built a demo system based on them. See this [introduction video](https://youtu.be/O3Jqq9yqJSE) and a longer [demo](https://www.youtube.com/watch?v=asxqpF7dH10) on in-air-handwriting word recognition. Also, see the [documents](https://duolu-fmkit.github.io/). Currently, the "word-210" dataset is openly available [here](https://drive.google.com/drive/folders/1RXj0t8NMYt_Jr5lW-BIeAIxw4aX5VshV?usp=sharing) without the need of application.

[![FMKit Demo](https://img.youtube.com/vi/O3Jqq9yqJSE/0.jpg)](https://www.youtube.com/watch?v=O3Jqq9yqJSE)


## The FMKit Library

The FMKit library contains a set of Python code and scripts to model and process in-air-handwriting signals. See the "code" folder. An overview is shown as follows.

* code_fmkit ---> The main library code. 
  * The "fmsignal" module contains the classes modeling the finger motion signal. 
  * The "fmsignal_vis" module contains functions to plot and animate the signal and the trajectory.
  * The "fmsignal_demo" module contains contains demo code.
  * The "pyrotation" and "pyrotation_demo" modules are copied from the [pyrotation](https://github.com/duolu/pyrotation) project, which is necessary for finger motion signal preprocessing.
* code_utilities ---> Utility code written in C to speed up the Dynamic Time Warping calculation.
* data_demo ---> Some example in-air-handwriting signals, collected using two devices.

The FMKit code library requires the following software packages

* Python 3 (tested with Python 3.6.9)
* NumPy (tested with NumPy 1.19.5)
* Matplotlib (tested with Matplotlib 3.1.2)

To use this code library, please download the Python modules under the "code_fmkit" folder and incorporate them into your project. Please also check the [user manual](https://duolu-fmkit.github.io/manual_signal/).

## The Data Repository

Here is a description of the dataset we have collected to facilitate our research on in-air handwriting. We have IRB approval for this data collection (Arizona State University STUDY00008279 and STUDY00010539). Currently, the "word-210" dataset is openly available [here](https://drive.google.com/drive/folders/1RXj0t8NMYt_Jr5lW-BIeAIxw4aX5VshV?usp=sharing). 

Two devices are used (shown in the following figure): a wearable device (a custom-made data glove with inertial sensors) and a contactless 3D camera (the Leap Motion controller). The data repository contains the following five datasets.

![Device illustration.](pics/devices.png)

**(1) ID-passcode**: We asked each participating user to create two distinct meaningful strings and write them in the air, one as an account ID and the other as an account passcode. Such a string may include alphanumeric letters, characters in a language other than English, or meaningful symbols such as five-pointed stars. The content of the string is determined by the user. Hence, each string can be used as either an ID or a passcode. For each string, we asked the user to write it 5 repetitions as registration and another 5 repetitions as login attempts. This simulates the normal sign-up and sign-in procedure (shown in the following figure). In total, 180 users participated in the data collection, and 360 strings were obtained. In total, there are 360 (strings) * 10 (repetitions) * 2 (devices) = 7,200 signals.

![Sign-in illustruction.](pics/sign-in.png)

**(2) ID-passcode-collision**: We asked 10 skilled imposters to imitate the in-air-handwriting of the ID and password generated by the users in the first dataset. In this setting, the imposters know the semantic meaning of the strings written by the legitimate users, but the imposters have not seen the legitimate users writing the ID and password in the air, which simulates a spoofing attack with semantic leakage. All strings in the first dataset are spoofed, and each imposter wrote every string with 5 repetitions using both two devices for data collection. In total, there are 360 (strings) * 10 (imposters) * 5 (repetitions) * 2 (devices) = 36,000 signals.

**(3) ID-passcode-spoofing**: We asked 10 skilled imposters to imitate the in-air-handwriting of the ID and password generated by the users in the first dataset. In this setting, the imposters can watch the recorded video of the in-air handwriting, and they will be informed of the semantic meaning of the in-air handwriting. 180 strings in the first dataset are spoofed (i.e., 90 users with both IDs and passcodes), and each imposter wrote the string with 5 repetitions using both two devices for data collection. In total, there are 360 (strings) * 10 (imposters) * 5 (repetitions) * 2 (devices) = 18,000 signals.

**(4) ID-passcode-persistence**: We kept collecting the sign-in in-air-handwriting of a subset of the users in the first dataset for a period of about 4 weeks, which simulates login activity in the long term. In the first dataset, the user wrote each string 5 repetitions as registration. In this dataset, the users wrote the strings for the account ID and the account passcode 5 repetitions consecutively as a session, and 10 sessions in total. 40 users participated in this dataset. In total, there are 80 (strings) * 10 (imposters) * 5 (repetitions) * 2 (devices) = 8,000 data samples.

**(5) In-Air-Handwriting Words (word-210)**: We asked 10 users to write 210 English words and 210 Chinese words with 5 repetitions for each word with both devices. In total, there are 2 (languages) * 210 (strings) * 10 (writers) * 5 (repetitions) * 2 (devices) = 42,000 data samples.

**(6) Usability Survey**: We asked the participating users to fill out a survey on the usability of a gesture sign-in system with various sensors and different types of gestures. 72 users responded to the survey.



## Authors

* **Duo Lu `<duolu@asu.edu>`** - main contributor, current maintainer of the project.
* **Yuli Deng `<ydeng19@asu.edu>`** - contributor.
* **Linzhen Luo `<lluo21@asu.edu>`** - contributor.
* **Dijiang Huang `<dijiang.huang@asu.edu>`** - our academic advisor and sponsor.


## Publications

Please cite the following work if you would like to use any of the resources provided by FMKit.

* Duo Lu, Linzhen Luo, Dijiang Huang, Yezhou Yang, "**FMKit: An In-Air-Handwriting Analysis Library and Data Repository.**" *CVPR Workshop on Computer Vision for Augmented and Virtual Reality, 2020.* [[pdf]](/papers/fmkit.pdf) [[link]](https://mixedreality.cs.cornell.edu/workshop/2020/papers#block-93cead2afaf5f6895a67) [[video]](https://youtu.be/O3Jqq9yqJSE)

Other publications:

* Duo Lu, Dijiang Huang,"**Systems and Methods for a Multifactor User Identification and Authentication Framework for In-Air-Handwriting with Hand Geometry and Deep Hashing**", *US Patent 11,120,255, 2021.* [[link]](https://patents.google.com/patent/US20200250413A1/en)
* Dijiang Huang, Duo Lu, "**Three-Dimensional In-The-Air Finger Motion based User Login Framework for Gesture Interface**", *US Patent 10,877,568, 2020.* [[link]](https://patents.google.com/patent/US10877568B2/en)
* Duo Lu, "**3D In-Air-Handwriting based User Login and Identity Input Method**", *Arizona State University Ph.D. Dissertation, 2021.* [[link]](https://keep.lib.asu.edu/items/161976)
* Duo Lu, Yuli Deng, Dijiang Huang, "**Global Feature Analysis and Comparative Evaluation of Freestyle In-Air-Handwriting Passcode for User Authentication**", *Annual Computer Security Applications Conference (ACSAC), 2021* [[link]](https://www.openconf.org/acsac2021/modules/request.php?module=oc_program&action=summary.php&id=260) [[video]](https://www.youtube.com/watch?v=kbh44gBlFNU)
* Duo Lu, Dijiang Huang, "**FMCode: A 3D In-the-Air Finger Motion Based User Login Framework for Gesture Interface.**" *arXiv preprint arXiv:1808.00130, 2018.* [[pdf]](/papers/fmcode.pdf) [[link]](https://arxiv.org/abs/1808.00130)
* Duo Lu, Dijiang Huang, "**FMHash: Deep Hashing of In-Air-Handwriting for User Identification.**" *in Proceedings of the International Conference on Communication (ICC), 2019* [[pdf]](/papers/fmhash.pdf) [[link]](https://arxiv.org/abs/1806.03574) [[slides]](/papers/fmhash_slides.pdf) [[video]](https://www.youtube.com/watch?v=MyaWe7RX8oE)
* Duo Lu, Dijiang Huang, Yuli Deng, and Adel Alshamrani. "**Multifactor user authentication with in-air-handwriting and hand geometry.**" *In 2018 International Conference on Biometrics (ICB), 2018.* [[pdf]](/papers/multifactor.pdf) [[link]](https://ieeexplore.ieee.org/document/8411230) [[slides]](/papers/multifactor_slides.pdf) [[poster]](/papers/multifactor_poster.pdf)
* Duo Lu, Kai Xu, and Dijiang Huang, "**A Data Driven In-Air-Handwriting Biometric Authentication System.**", *in Proceedings of the International Joint Conference on Biometrics (IJCB), 2017.* [[pdf]](/papers/data-driven.pdf) [[link]](https://ieeexplore.ieee.org/document/8272739) [[slides]](/papers/data-driven_slides.pdf)

## License

The code of this project is released under the MIT License (see the [LICENSE.md](LICENSE.md) file for details).

The "word-210" dataset is released under the [CC-BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).

Other datasets, including in-air-handwriting of user ID and passcode, are released under the [FMKit License](https://docs.google.com/document/d/1AHX3lj1mjm4ZZEZTHNdm3xDmJAAWi7P6bIdLlBNZ8PA/edit?usp=sharing).

## Acknowledgments

We would like to thank all participants and volunteers who helped us collect the data.

This project is supported by the NSF CCRI award [#1925709](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1925709).

This project is also supported by the Arizona State University Fulton School of Engineering micro infrastructure grant.
