# <h1> rbaTheta: Ramping Behaviour Analysis

Definition: A sudden change in time-varying data is termed as an event. For example a day (24 h) can be summed up with a few
events than 24 points. This algorithm identifies the events and classifies them into stationary or significant events.
An abrupt change is understood as significant event while a persistent event is a stationary. Rest of the data have little
significace in context of decision making.

![rbaTheta](/plots/plotted_figures/RBAevents_new.png?raw=true)

#### Directory organization

| Directory   | Files                                       | Description                             |
| ----------- | ------------------------------------------- | --------------------------------------- |
| core        | event_extraction.py , helpers.py , model.py | Contains the model and associated files |
| input_data  | QGISfiles, Wind files                       | Input excel and shape files             |
| plots       | plotted figures                             | plotting scripts and figures            |
| simulations | test results and script to run tests        | fast_test.py is for a simple test       |

#### How to run?

To run an experiment execute the main.py file. It calls the model and helpers from core folder and exectues as a parallel process. To verify that the execution was successful, look out for the a message stating how long it took to execute on terminal.

```
main.py
```

#### **Please cite the below publication as the source. The source code has a MIT liscence meaning users are free to modify and use only with a explicit written permission or citation of the publication.**

> [Mishra S, Ören E, Bordin C, Wen F, Palu I. Features extraction of wind ramp events from a virtual wind park. Energy Reports. 2020 Nov 1;6:237-49.](https://doi.org/10.1016/j.egyr.2020.08.047)

```
@article{mishra2020features,
  title={Features extraction of wind ramp events from a virtual wind park},
  author={Mishra, Sambeet and {\"O}ren, Esin and Bordin, Chiara and Wen, Fushuan and Palu, Ivo},
  journal={Energy Reports},
  volume={6},
  pages={237--249},
  year={2020},
  publisher={Elsevier}
}
```
