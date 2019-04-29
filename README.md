# DGABot Microservice

The main goals of this project are to:
* Provide a powerful tool for detecting urls created with Domain Generation Algorithms
* Train and evaluate a classifier aimed at discriminating between legitimate and DGA domains.
* Deploy the production model by providing an inference API as a REST service. 
* Build a docker image containing the application, production model and any resources to run as a micro-service.

These goals are achieved through the use a Long short-term memory (LSTM) Artificial Recurrent Neural Network, derived from the field of deep learning. As well as Flask; a micro web framework written in Python.


DGABot Class Usage Example
--------------------------
Import, instantiate, predict:
      
      from dgabot import DGABot
      dgabot = DGABot() 
      dgabot.load_model()
      dgabot.predict('www.googlexx111.com') 
      
Note: dgabot.predict() returns a dictionary of the form {'class': 0} or {'class': 1}, depending on the predicted class of the input. 


DGABot Microservice Usage Example
---------------------------------
Build, run, predict: 

      $ cd dgabot
      $ docker build --tag=dgabot .
      $ docker run -p 5000:5000 dgabot
      > Running on http://127.0.0.1:5000/
      
      In a web the browser: 
      http://127.0.0.1:5000/predict/www.googlexx111.com


More Information
----------------
Python 3.6 + is fully supported.  The application was fully tested with Python 3.6.2


License
-------
* Free software: MIT License
