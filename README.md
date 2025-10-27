Overview: 

This branch adds a PI control loop and a fixed supervisory mode to check that the PI loop works as expected in the simplified building environment.
It’s mainly for testing the PI loop before connecting the RL actions later during training.

Files:

continuous_building_environment.py: Includes the updated PI loop and supervisory control actions to test system response.
test_pi_heating.py: A simple test script to run the new environment and make sure the PI loop is behaving correctly.

No other files are needed — these two are all you need to run the test.
