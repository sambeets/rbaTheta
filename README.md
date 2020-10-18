# <h1> rbaTheta: Ramping Behaviour Analysis

Definition: A sudden change in time-varying data is termed as an event. For example a day (24 h) can be summed up with a few
events than 24 points. This algorithm identifies the events and classifies them into stationary or significant events. 
An abrupt change is understood as significant event while a persistent event is a stationary. Rest of the data have little
significace in context of decision making.


#### Directory organization

| Directory | Files | Description |
| --- | --- | --- |
| Core | event_extraction.py , helpers.py , model.py | Contains the model and associated files
| input_data | QGISfiles, Wind files | Input excel and shape files
| plots| plotted figures | plotting scripts and figures
| simulations| test results and script to run tests | fast_test.py is for a simple test

#### How to run?

To execute simply execute the files in simulation folder. For example:
> fast_test.py


#### Please cite the below publication as the source. The source code has a MIT liscence meaning users are free to modify and use only with a permission or citation of the publication.




