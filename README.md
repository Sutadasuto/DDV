# DDV
Extra dependencies needed:

OpenFace (https://github.com/TadasBaltrusaitis/OpenFace)
SyntaxNet (https://github.com/tensorflow/models/tree/master/research/syntaxnet)
- Both can be installed in any desired folder, but both locations must be specified inside openface.yaml and syntaxnet.yaml in DDV/config
- In ~/models/research/syntaxnet/syntaxnet/models, for Spanish, you need to add parsey_universal and the Spanish model. Also, you need to replace the demo.sh in ~/models/research/syntaxnet/syntaxnet/demo.sh (basically, you need to comment the last 4 lines, that convert the output from conll to tree)

For Automatic Speech Recognition, IBM's Watson is used remotely (https://www.ibm.com/watson/services/speech-to-text/). You should create your own account and put your login data in "credentials.txt" inside DDV/audio (dummy data is included currently in the file as an example).

For audio extraction, you should have FFmpeg installed on your system (https://www.ffmpeg.org/).

For audio analysis, we used a slightly modified version of COVAREP (https://covarep.github.io/covarep/). This modified repository is included inside the folder "covarep" in DDV/audio.

There are three different Python scripts for execution, all of their names begin by "main". All user variables are hardcoded after the imports.

Depending on the database you want to use for analysis, you should create/modify a folder inside DDV/config (use the included folders as an example). Inside the "main" scripts, you can modify which folder is called by modifying the "dataset_name" variable.

Remember this code is not intended to work out of the box, it is just the code resulting from my thesis research work. The implementation was tested on Ubuntu 18.04 using Python 3.6 with Anaconda. Neural Network implementations were tested using a GeForce GTX 1050 GPU with Driver Version 396.54. "environment.yml" contains the dependencies installed in the Python environment used for testing. For further reference, you can check this article: http://openaccess.thecvf.com/content_CVPRW_2019/papers/CFS/Rill-Garcia_High-Level_Features_for_Multimodal_Deception_Detection_in_Videos_CVPRW_2019_paper.pdf 
