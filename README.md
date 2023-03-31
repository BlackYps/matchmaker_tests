This repository holds some tools to run simulations with the Forged Alliance Forever matchmaker.

This assumes you have set up the [server](https://github.com/FAForever/server/) repository.  
The `matchmaker_performance_test.py` file is supposed to go into the root directory of that repo.  
You need to additionally install the numpy and matplotlib packages. I recommend adding them to the 
pipfile of the server repo so you don't have to install them manually. 
Once that is done you can run `test_matchmaker_performance()` in `matchmaker_performance_test.py` as 
a normal unit test for a matchmaker run with simulated data. This can take some seconds as by default
it runs a lot of iterations. At the end a window should open with some graphs about the run. Have a 
look at the log output as well. If you want to make any changes, change the code of the test directly.

The `api queries` folder holds some scripts to query the api for data that I used to tune the generated
player data to what we have in the real world.

`faf_client_cli.py` is a command line tool to interact with a running server instance. It is not needed if
you just want to make simulation runs. Credit and copyright for this file goes to Askaholic.
