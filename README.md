# DGABot Dockerized Microservice

The main goals of this project are to:
* Provide a powerful tool for detecting urls created with Domain Generation Algorithms
* 
* Build a docker image containing the application, production model and any resources to run as a micro-service.

These goals are achieved through the use a Long short-term memory (LSTM) Artificial Recurrent Reural Network, derived from the field of deep learning. As well as Flask; a micro web framework written in Python.

DGABot Class Usage Example
--------------------------
Import, instantiate, predict::
      
      from dgabot import DGABot
      dgabot = DGABot() 
      dgabot.load_model()
      dgabot.predict('www.googlexx111.com') 
      
Note: dgabot.predict() return a dictionary of the form {'class': 0} or {'class': 1}, depending on the predited class of the input. 

DGABot Microservice Usage Example
---------------------------------
