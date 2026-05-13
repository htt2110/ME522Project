## Heading 1 File Description

## Heading 2 activateCmeraScriptOnButtonPress.py

This file reads the signal from Arduino, which is activated on a button press 
telling the system the camera is ready to detect the trash being presented


## Heading 2 CameraInterface.py

This script is running a trained neural network, which is detecting whether the trash if
it is organic or recyclable. Based on the pin detected a GPIO pin is activated, and that sends a 
signal to arduino to run the motor to open the respective door
