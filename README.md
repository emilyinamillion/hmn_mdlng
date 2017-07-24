# hmn_mdlng 1.0

I wrote a module 🤘🎉.

Throughout my time developing machine learning models for Legal Server / Houston.ai, I've been developing an ipython notebook full of functions that help me quickly assess the results of my NLP models. Eventually I grew tired of passing parameters into these functions and decided to write a class to handle these functions. Then I realized it might be a useful module to introduce to the ecosystem. 

### WHAT IS HMN_MDLNG?
hmn_mdlng is a module written on top of sci-kit learn that assists with analyzing modeling results. It would also be great for a beginner natural language modeler to explore the process of NLP, as the auto settings will take your clean text and return a basic classifier without much hassle. hmn_mdlng is currently only available for supervised natural language problems but will be available for a wider range of problems in future releases. 

### WHAT DOES HMN_MDLNG OFFER THAT SCI-KIT LEARN DOESN'T?
hmn_mdlng mostly just provides a smoother process for modeling utilizing sci-kit learn. Its offerings include:

1. Takes your dataframe, splits into test and train
2. Vectorizes your corpus
3. Passes it into a model
4. Prints model analysis results from sci-kit learn
5. Prints false assignments

### WHAT DOES HMN_MDLNG NOT DO?
Provide moral support during particularly difficult modeling tasks. 

## How do I install this little thing?
I'll eventually get this into the pypi ecosystem once it's bulked up enough, but for the moment I've included a setup.py file and a test.py modeling off 20 NewsGroups. So anyway, git clone this repo, cd into the folder, and run this in your terminal.

``` python 
 python setup.py install
 ```

## This is the beginning

hmn_mdlng is still very much under construction. I wrote this module for a very specific and NLP-centric purpose as it has been a work project, but I recognize that it could be useful for more general modeling purposes. Future developments include, but are not limited to, the list below. < I take requests 🎤 >. If you have any, please get in touch with me.

1. standard supervized learning modeling
2. more direct comparisons between models themselves
